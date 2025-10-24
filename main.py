# filename: app_with_rag.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import json
import boto3
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import PyPDF2
import math
import numpy as np
import faiss
import time

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv(dotenv_path=".env.local")
openai_key = os.getenv("OPENAI_API_KEY")

# Initialize clients (keeps same names you already had)
openaiclient = OpenAI(api_key=openai_key)
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

ALLOWED_TABLE = os.getenv("ALLOWED_TABLE", "agentic_program_data")
DEFAULT_MAX_ROWS = int(os.getenv("DEFAULT_MAX_ROWS", "1000"))
N8N_WEBHOOK_URL = "https://n8n.altius.cc/webhook/agentic-sql"
#N8N_WEBHOOK_URL = "https://n8n.altius.cc:5678/webhook/agentic-sql"

# -------------------------------
# RAG / Embeddings config
# -------------------------------
PDF_PATH = "./docs/QAT_Guidance.pdf"
CHUNKS_JSON = "./docs/qat_chunks.json"
FAISS_INDEX_PATH = "./docs/qat_index.faiss"
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model
EMBED_DIM = 1536  # embedding dim for text-embedding-3-small (if it differs, it will be validated)
CHUNK_SIZE = 1000  # approx characters per chunk
CHUNK_OVERLAP = 200
TOP_K = 3  # number of chunks to retrieve for RAG

# -------------------------------
# Technical Context (small fallback)
# -------------------------------
def get_technical_context():
    """
    Keep a compact technical context (short) — used as a fallback and appended.
    This is unchanged from your original code (keeps your business rules brief).
    """
    return """
# QAT_Guidance.pdf (compact reference)

Key rules:
- Stock status: CASE on MOS vs MIN_STOCK_MOS and MAX_STOCK_MOS.
- AMC uses months_in_past + months_in_future (only non-zero months).
- Unmet Demand when actual consumption available: (Actual consumption * Days stocked out)/(Days in Month - Days stocked out).
- Expired stock: sum of opening balances of batches expiring this month.
"""
# -------------------------------
# PDF extraction & embedding helpers
# -------------------------------

def pdf_to_text(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    text = []
    for p in reader.pages:
        page_text = p.extract_text()
        if page_text:
            text.append(page_text)
    return "\n\n".join(text)

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Simple character-based chunking. Returns list of chunk strings.
    """
    text = text.replace("\r", "")
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # advance by chunk_size - overlap
        start += (chunk_size - overlap)
    return chunks

def build_qat_embeddings(force_rebuild=False):
    """
    Build embeddings for the QAT guidance PDF and save:
    - FAISS index at FAISS_INDEX_PATH
    - chunks metadata JSON at CHUNKS_JSON

    This function is idempotent: if files exist and force_rebuild is False, it returns.
    """
    if not Path(PDF_PATH).exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    if Path(FAISS_INDEX_PATH).exists() and Path(CHUNKS_JSON).exists() and not force_rebuild:
        print("RAG index and chunks already exist — skipping build.")
        return

    print("Extracting text from PDF...")
    text = pdf_to_text(PDF_PATH)
    print("Chunking text...")
    chunks = chunk_text(text)

    print(f"{len(chunks)} chunks created — creating embeddings...")

    # create embeddings in batches to avoid timeouts
    embeddings = []
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        # call OpenAI embeddings
        resp = openaiclient.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        # resp.data is expected as list
        for item in resp.data:
            embeddings.append(item.embedding)
        time.sleep(0.1)  # tiny pause to be kind to API

    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    print(f"Embedding dim: {dim}, total vectors: {embeddings.shape[0]}")

    # build FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index written to {FAISS_INDEX_PATH}")

    # write chunks metadata
    metadata = {"chunks": [{"id": idx, "text": chunks[idx]} for idx in range(len(chunks))]}
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Chunk metadata written to {CHUNKS_JSON}")

def load_faiss_and_chunks():
    """
    Load FAISS index and chunk metadata into memory.
    Returns (index, chunks_list)
    """
    if not Path(FAISS_INDEX_PATH).exists() or not Path(CHUNKS_JSON).exists():
        raise FileNotFoundError("FAISS index or chunks metadata missing. Run build_qat_embeddings() first.")

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    chunks = [c["text"] for c in metadata["chunks"]]
    return index, chunks

def retrieve_relevant_context(prompt, top_k=TOP_K):
    """
    Given a prompt, embed it and query FAISS for top_k chunks.
    Returns concatenated chunk text (safely trimmed).
    """
    # ensure index exists
    if not Path(FAISS_INDEX_PATH).exists() or not Path(CHUNKS_JSON).exists():
        # Try to build automatically if missing
        build_qat_embeddings()

    index, chunks = load_faiss_and_chunks()

    # embed the prompt
    emb_resp = openaiclient.embeddings.create(input=[prompt], model=EMBEDDING_MODEL)
    q_vec = np.array(emb_resp.data[0].embedding).astype("float32").reshape(1, -1)

    # search
    D, I = index.search(q_vec, top_k)
    idxs = I[0].tolist()
    # sometimes faiss returns -1 if insufficient vectors; filter
    idxs = [i for i in idxs if i >= 0 and i < len(chunks)]

    retrieved = []
    for i in idxs:
        retrieved.append(chunks[i].strip())

    # small header and concatenation
    combined = "\n\n---\n\n".join(retrieved)
    header = f"Relevant guidance snippets (from QAT guidance) — top {len(retrieved)} matches:\n\n"
    return header + combined

# -------------------------------
# Build index at startup (if missing) — optional
# -------------------------------
try:
    if not (Path(FAISS_INDEX_PATH).exists() and Path(CHUNKS_JSON).exists()):
        print("QAT embeddings/index not found — building at startup (this may take a minute)...")
        build_qat_embeddings()
    else:
        print("QAT RAG index present.")
except Exception as e:
    # don't crash the app if build fails; log and continue
    print("Warning: failed to build/load QAT embeddings at startup:", e)

# -------------------------------
# FastAPI setup (unchanged)
# -------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_json_from_markdown(text: str) -> str:
    """Remove markdown code blocks from JSON response"""
    text = text.strip()
    # Remove a leading code fence (``` or ```lang) if present
    if text.startswith("```"):
        newline_idx = text.find("\n")
        if newline_idx != -1:
            text = text[newline_idx + 1 :]
        else:
            text = text[3:]
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text.strip()

# -------------------------------
# Main endpoint (keeps your original flow)
# -------------------------------
@app.post("/generate-dashboard")
async def generate_dashboard(req: Request):
    body = await req.json()
    prompt = body["prompt"]
    print(f"Received prompt: {prompt}")

    # -------------------------------
    # RAG retrieval: get top relevant chunks for this prompt
    # -------------------------------
    try:
        rag_context = retrieve_relevant_context(prompt, top_k=TOP_K)
        print("Retrieved RAG context (trimmed):", rag_context[:500])
    except Exception as e:
        print("RAG retrieval failed — proceeding with fallback technical context. Error:", e)
        rag_context = ""  # fallback to only compact context

    # -------------------------------
    # Step 1: Generate SQL using OpenAI
    # We inject the retrieved RAG context into the system prompt to ground SQL generation.
    # -------------------------------
    system_content = (
        "You are a MySQL 8.0.43 expert specializing in supply chain analytics for the QAT (Quantification Analytics Tool) system.\n\n"
        # Insert RAG-retrieved chunks first (keeps it concise)
        f"{rag_context}\n\n"
        # Then append the compact technical context (unchanged)
        f"{get_technical_context()}\n\n"
        "# Available Columns (USE ONLY THESE - EXACT NAMES):\n"
                    "PROGRAM_NAME, PROGRAM_CODE, COUNTRY_CODE, COUNTRY, HEALTH_AREA, HEALTH_AREA_CODE, "
                    "ORGANISATION_CODE, ORGANISATION_NAME, PLANNING_UNIT_NAME, FORECASTING_UNIT_NAME, "
                    "PLANNING_UNIT_ID, REORDER_FREQUENCY_IN_MONTHS, MIN_MONTHS_OF_STOCK, "
                    "LOCAL_PROCUREMENT_LEAD_TIME, SHELF_LIFE, CATALOG_PRICE, MONTHS_IN_FUTURE_FOR_AMC, "
                    "MONTHS_IN_PAST_FOR_AMC, DISTRIBUTION_LEAD_TIME, FORECAST_ERROR_THRESHOLD, "
                    "TRANS_DATE, AMC, AMC_COUNT, MOS, MIN_STOCK_QTY, MIN_STOCK_MOS, "
                    "MAX_STOCK_QTY, MAX_STOCK_MOS, OPENING_BALANCE, SHIPMENT_QTY, "
                    "FORECASTED_CONSUMPTION_QTY, ACTUAL_CONSUMPTION_QTY, EXPIRED_STOCK, "
                    "CLOSING_BALANCE, UNMET_DEMAND, PLAN_BASED_ON, VERSION_STATUS, VERSION_TYPE, VERSION_ID\n\n"
                  '''
                    DB_SCHEMA_EXPLANATION
                    ### Database Field Definitions for Supply Chain and Forecasting Expert ### 
                    PROGRAM_NAME - Name of the Program. Generally combination of Country, Health Area and Organisation.
                    PROGRAM_CODE - Code of the Program. Generally combination of Country code. Health Area code and Organisation code.
                    COUNTRY_CODE - 3 digit Code of the Country.
                    COUNTRY - Name of the Country.
                    HEALTH_AREA - Name of Health Area/Technical Area
                    HEALTH_AREA_CODE - 3 to 10 digit code of Health Area/Technical Area.
                    ORGANISATION_CODE - Name of Organisation
                    ORGANISATION_NAME - 3 to 10 digit code of Organisation
                    PLANNING_UNIT_NAME - Name of the Planning Unit (product/SKU).
                    FORECASTING_UNIT_NAME - Name of the Forecasting Unit.
                    REORDER_FREQUENCY_IN_MONTHS - Reorder level settings for this Planning Unit and Program (reorder cycle) i.e How many months between shipments?
                    MIN_MONTHS_OF_STOCK - Minimum MoS settings for this Planning Unit and Program.
                    LOCAL_PROCUREMENT_LEAD_TIME - Local Procurement Lead Time for shipments to be procured if procured by local procurement agent
                    SHELF_LIFE - Shelf Life of the product in months
                    CATALOG_PRICE - Price of the product
                    MONTHS_IN_FUTURE_FOR_AMC - Look-ahead period for Average Monthly Consumption calculation. Including current month.
                    MONTHS_IN_PAST_FOR_AMC - Look-back period for Average Monthly Consumption calculation.
                    DISTRIBUTION_LEAD_TIME - How many months does it take between shipment receipt and the product to be distributed down to the lowest level? Used for suggested shipments ahead of understock.
                    TRANS_DATE - Month for which this record is for (temporal dimension).
                    AMC - Average Monthly Consumption calculated and stored for this Month
                    AMC_COUNT - Number of months of data used for AMC calculation.
                    MOS - Months of Stock available for this Product (Inventory health metric). 0 indicates Out of Stock, Below Min MOS indicates under stock, Above Max MOS (Min Mos + Reorder Level) indicates overstock, Between Min MOS and Max MOS indicates Adaqute stock.
                    MIN_STOCK_QTY - Minimum Stock Quantity (safety stock in units). Any value below this indicates an undersupply risk.
                    MIN_STOCK_MOS - Minimum Months of Stock (safety stock in time). Any value below this indicates an undersupply risk.
                    MAX_STOCK_QTY - Maximum Stock Quantity (inventory ceiling in units). Any value above this indicates an oversupply/obsolescence risk.
                    MAX_STOCK_MOS - Maximum Months of Stock (inventory ceiling in time). Any value above this indicates an oversupply/obsolescence risk.
                    OPENING_BALANCE - Beginning stock level for the period.
                    SHIPMENT_QTY - Quantity of Shipment expected to be received in this month (inbound supply).
                    FORECASTED_CONSUMPTION_QTY - Future expected demand.
                    ACTUAL_CONSUMPTION_QTY - Actual historical demand.
                    EXPIRED_STOCK - Amount of Stock that Expired this month (a measure of waste/obsolescence).
                    CLOSING_BALANCE Ending stock level for the period.
                    UNMET_DEMAND - Amount of demand that was there but was Unmet due to a stock-out. Occurs when the closing balance is '0' and part of consumption was unmet, or when adjusted consumption accounts for days stocked out. Unmet demand can also happen when a negative manual adjustment is larger than the projected ending balance.
                    PLAN_BASED_ON - Should minimum and maximum inventory parameters be based on QTY or months of stock (MOS)? Most products are better planned by MOS, while some low consumption, higher expiry products are better planned by quantity.
                    PLANNING_UNIT_ID - Id for the Planning Unit.
                    VERSION_STATUS - Version Status Desc (Pending Approval, Approved, Needs Revision, No Review Needed)
                    VERSION_TYPE - Version Type Desc (Draft Version, Final Version)
                    VERSION_ID - Version Id

                    Below are the formulas that can be used

                    1) Forecast Error is calculated using the Weighted Absolute Percentage Error (WAPE). WAPE is used over MAPE (Mean Absolute Percentage Error) as it can account for when consumption is intermittent or low.

                    2) The WAPE formula uses the previous 3-12 months of data depending on the selection in the Time Window dropdown. For example, if the ‘Time Window’ selected is 6 months, then 6 months of actual consumption and 6 months of forecasted consumption is used
                    '''
                    "# CRITICAL Column Name Mappings:\n"
                    "- Stock status (derived) → Use CASE with MOS, MIN_STOCK_MOS, MAX_STOCK_MOS\n"
                    "- Minimum months threshold → MIN_STOCK_MOS (NOT MIN_MONTHS_OF_STOCK)\n"
                    "- Maximum months threshold → MAX_STOCK_MOS (NOT MAX_MONTHS_OF_STOCK)\n"
                    "- Current inventory level → MOS\n"
                    "- Expired/expiry stock → EXPIRED_STOCK\n"
                    "- Report period filtering → TRANS_DATE\n"
                    "- Program identifier → PROGRAM_CODE (format: Country-HealthArea-Org)\n\n"
                    
                    "# Business Intelligence Keywords:\n"
                    "- 'expiries', 'expired stock', 'projected expiry' → EXPIRED_STOCK column\n"
                    "- 'stock status', 'inventory health' → Calculate using MOS with CASE statement\n"
                    "- 'consumption data' → ACTUAL_CONSUMPTION_QTY (historical), FORECASTED_CONSUMPTION_QTY (projected)\n"
                    "- 'inventory data' → OPENING_BALANCE, CLOSING_BALANCE, MOS\n"
                    "- 'shipment data' → SHIPMENT_QTY\n"
                    "- 'supply plan' → Include MIN_STOCK_MOS, MAX_STOCK_MOS, MOS, AMC, UNMET_DEMAND\n"
                    "- 'report period' → Filter using TRANS_DATE BETWEEN start AND end\n\n"
                    
                    "# Version Type Keywords:\n"
                    "- 'draft', 'draft version', 'latest draft' → VERSION_TYPE = 'Draft Version'\n"
                    "- 'final', 'final version', 'approved' → VERSION_TYPE = 'Final Version'\n"
                    "- 'latest version' (no type specified) → Use MAX(VERSION_ID) only\n\n"
                    
                    "# Query Rules:\n"
                    "1. Output ONLY the SQL query - no markdown, no explanations\n"
                    "2. Use LIKE for PROGRAM_CODE matching: WHERE PROGRAM_CODE LIKE 'value'\n"
                    "3. For 'latest draft/final', include VERSION_TYPE in subquery:\n"
                    "   VERSION_ID = (SELECT MAX(VERSION_ID) FROM agentic_program_data WHERE PROGRAM_CODE LIKE 'XXX' AND VERSION_TYPE = 'YYY')\n"
                    "4. for latest version without version type, include this subquery:\n"
                    "   VERSION_ID = (SELECT MAX(VERSION_ID) FROM agentic_program_data WHERE PROGRAM_CODE LIKE 'XXX')\n"
                    "4. Date ranges: TRANS_DATE BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'\n"
                    "5. Stock status CASE (use exact column names):\n"
                    "   CASE \n"
                    "     WHEN MOS = 0 THEN 'Out of Stock'\n"
                    "     WHEN MOS < MIN_STOCK_MOS THEN 'Understock'\n"
                    "     WHEN MOS > MAX_STOCK_MOS THEN 'Overstock'\n"
                    "     ELSE 'Adequate Stock'\n"
                    "   END AS stock_status\n"
                    "6. For supply plan queries, default columns:\n"
                    "   TRANS_DATE, PLANNING_UNIT_NAME, MOS, MIN_STOCK_MOS, MAX_STOCK_MOS, AMC, \n"
                    "   UNMET_DEMAND, EXPIRED_STOCK, OPENING_BALANCE, CLOSING_BALANCE, SHIPMENT_QTY,\n"
                    "   ACTUAL_CONSUMPTION_QTY, FORECASTED_CONSUMPTION_QTY\n"
                    "7. NEVER create column names not in the Available Columns list\n"
                    "8. Always ORDER BY TRANS_DATE for time-series queries\n"
                    "9. Do NOT filter VERSION_STATUS unless explicitly mentioned\n"
                    "10. Only SELECT columns directly relevant to user's request\n"
    )

    sql_resp = openaiclient.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=3000,
        temperature=0,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
    )

    sql_query = sql_resp.choices[0].message.content.strip()
    print(f"Generated SQL: {sql_query}")

    # -------------------------------
    # Step 2: Send SQL to N8N webhook (unchanged)
    # -------------------------------
    n8n_resp = requests.post(N8N_WEBHOOK_URL, json={"sql": sql_query})
    n8n_resp.raise_for_status()
    data = n8n_resp.json()
    print(f"N8N response data: {data}")

    # -------------------------------
    # Step 3: Analyze data using Claude via AWS Bedrock (unchanged)
    # -------------------------------
    system_prompt = (
        "You are a world-class Supply Planning Expert specializing in QAT inventory analytics.\n\n"
    f"{get_technical_context()}\n\n"
    f"User's question: {prompt}\n\n"
    
    "# Your Task:\n"
    "Analyze the provided dataset to answer the user's specific question comprehensively.\n"
    "If data contains multiple planning units, analyze each separately with dedicated insights.\n\n"
    
    "# Analysis Guidelines:\n"
    "1. Understand Question Intent: Identify what user is asking (trends, comparisons, forecasting, problems)\n"
    "2. Data-Driven Insights: Base conclusions on actual values, calculate metrics, identify patterns\n"
    "3. Supply Planning Expertise: Interpret MOS vs thresholds, assess inventory health, evaluate consumption\n"
    "4. Contextual Recommendations: Provide specific, actionable advice when user asks 'what should I do'\n"
    "5. Visualization Recommendations: Choose appropriate charts (line for trends, bar for comparisons)\n\n"
    
    "# Response Format - CRITICAL:\n"
    "You MUST return ONLY valid JSON with this EXACT structure (no additional text before or after):\n\n"
    "{\n"
    '  "requestType": "1-Simple English" | "2-Data Table" | "3-Visualization",\n'
    '  "data": [...original dataset or processed data...],\n'
    '  "visualization": {\n'
    '    "summaryCards": [\n'
    '      {\n'
    '        "title": "Metric name",\n'
    '        "value": "Formatted value",\n'
    '        "subtitle": "Additional context",\n'
    '        "status": "success" | "warning" | "error" | "info"\n'
    '      }\n'
    '    ],\n'
    '    "charts": [\n'
    '      {\n'
    '        "type": "line" | "bar" | "composed" | "area",\n'
    '        "title": "Chart title",\n'
    '        "description": "What this chart shows",\n'
    '        "data": [...data rows from dataset...],\n'
    '        "xAxis": "TRANS_DATE or other field",\n'
    '        "yAxisLabel": "Label for Y axis",\n'
    '        "series": [\n'
    '          {\n'
    '            "dataKey": "Column name from data",\n'
    '            "name": "Display name",\n'
    '            "color": "#hexcolor",\n'
    '            "type": "line" | "bar" | "area",\n'
    '            "strokeDasharray": "5 5"\n'
    '          }\n'
    '        ]\n'
    '      }\n'
    '    ],\n'
    '    "recommendations": [\n'
    '      {\n'
    '        "type": "immediate" | "short-term" | "medium-term" | "insight",\n'
    '        "title": "Section title",\n'
    '        "items": ["Recommendation 1", "Recommendation 2"]\n'
    '      }\n'
    '    ]\n'
    '  },\n'
    '  "answer": "Comprehensive narrative analysis"\n'
    "}\n\n"
    
    "# Guidelines for Each Section:\n"
    "summaryCards: Create 3-5 key metrics. Status: success(good), warning(attention), error(critical), info(neutral)\n"
    "charts: Include 1-3 charts. Use TRANS_DATE for xAxis in time-series. Series dataKey must match actual column names.\n"
    "recommendations: Only include if user asks for advice. Group by urgency: immediate, short-term, medium-term\n"
    "answer: Clear narrative explaining data, key findings, and answers to user's question with specific numbers\n\n"
    
    "# Color Guide:\n"
    "Blue #3b82f6: Primary data | Green #10b981: Targets/good | Orange #f59e0b: Warning | Red #ef4444: Critical | Purple #8b5cf6: Forecasts\n\n"
    
    "# Critical Rules:\n"
    "- Output ONLY valid JSON, no text before or after\n"
    "- Use actual column names in dataKey fields\n"
    "- Ensure chart data contains complete rows with all referenced columns\n"
    "- Answer specific question asked, don't provide generic analysis\n"
    "- Use actual data values (cite numbers, dates, planning units)\n"
)

    bedrock_body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8000,  # Reduced from 40000 to 8000
        "temperature": 0.3,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": f"Analyze this dataset and respond with ONLY the JSON format specified (no additional text):\n{json.dumps(data, indent=2)}"
            }
        ]
    })

    try:
        bedrock_response = bedrock_runtime.invoke_model(
            modelId="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            body=bedrock_body
        )

        response_body = json.loads(bedrock_response['body'].read())
        analysis_response = response_body['content'][0]['text'].strip()
        
        # Clean any markdown wrapping
        cleaned_analysis = extract_json_from_markdown(analysis_response)
        
        # Additional cleaning: remove any text before first { or after last }
        start_idx = cleaned_analysis.find('{')
        end_idx = cleaned_analysis.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            cleaned_analysis = cleaned_analysis[start_idx:end_idx+1]
        
        try:
            parsed_analysis = json.loads(cleaned_analysis)
            
            # Validate and add defaults if missing
            if "visualization" not in parsed_analysis:
                parsed_analysis["visualization"] = {
                    "summaryCards": [],
                    "charts": [],
                    "recommendations": []
                }
            else:
                viz = parsed_analysis["visualization"]
                if "summaryCards" not in viz:
                    viz["summaryCards"] = []
                if "charts" not in viz:
                    viz["charts"] = []
                if "recommendations" not in viz:
                    viz["recommendations"] = []
                    
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Cleaned analysis (first 500 chars): {cleaned_analysis[:500]}")
            
            # Fallback structure
            parsed_analysis = {
                "requestType": "1-Simple English",
                "data": data[:10],  # Return first 10 rows
                "visualization": {
                    "summaryCards": [
                        {
                            "title": "Total Records",
                            "value": str(len(data)),
                            "subtitle": "Data rows returned",
                            "status": "info"
                        }
                    ],
                    "charts": [],
                    "recommendations": []
                },
                "answer": f"Received {len(data)} data records. Analysis failed due to response format issue. Raw response: {analysis_response[:200]}..."
            }

    except Exception as e:
        print(f"Bedrock invocation error: {e}")
        # The 'parsed_analysis' is set here on error.
        parsed_analysis = {
            "requestType": "1-Simple English",
            "data": data[:10],
            "visualization": {
                "summaryCards": [],
                "charts": [],
                "recommendations": []
            },
            "answer": f"Error analyzing data: {str(e)}"
        }

    # -------------------------------
    # Step 4: Return SQL + result + analysis
    # -------------------------------
    # *** CORRECTED INDENTATION ***: This is now correctly aligned with the body of the function.
    return { 
        "query": sql_query,
        "data": data[:100] if len(data) > 100 else data,  # Limit raw data
        "analysis": parsed_analysis
    }

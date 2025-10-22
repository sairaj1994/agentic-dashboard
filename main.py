from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import json
import boto3
from openai import OpenAI
from dotenv import load_dotenv

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv(dotenv_path=".env.local")
openai_key = os.getenv("OPENAI_API_KEY")

# Initialize clients
openaiclient = OpenAI(api_key=openai_key)
# Initialize AWS Bedrock client
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv("AWS_REGION", "us-east-1")  # Add AWS_REGION to your .env.local
)

ALLOWED_TABLE = os.getenv("ALLOWED_TABLE", "agentic_program_data")
DEFAULT_MAX_ROWS = int(os.getenv("DEFAULT_MAX_ROWS", "1000"))
N8N_WEBHOOK_URL = "http://localhost:5678/webhook-test/agentic-sql"

# -------------------------------
# Database Schema Explanation
# -------------------------------
DB_SCHEMA_EXPLANATION = """
# Database Field Definitions for Supply Chain and Forecasting Expert

- PROGRAM_NAME: Name of the Program (e.g., Country + Health Area + Organisation)
- PROGRAM_CODE: Program code (same context as PROGRAM_NAME)
- COUNTRY_CODE: 3-digit ISO Country Code
- COUNTRY: Name of the Country
- HEALTH_AREA: Name of Health/Technical Area
- HEALTH_AREA_CODE: 3–10 digit code of Health Area
- ORGANISATION_CODE: Organisation Code
- ORGANISATION_NAME: Organisation Name
- PLANNING_UNIT_NAME: Name of Planning Unit (Product/SKU)
- FORECASTING_UNIT_NAME: Name of Forecasting Unit
- PLANNING_UNIT_ID: Unique ID for the Planning Unit
- REORDER_FREQUENCY_IN_MONTHS: Months between shipments
- MIN_MONTHS_OF_STOCK: Minimum months of stock
- LOCAL_PROCUREMENT_LEAD_TIME: Lead time for local procurement
- SHELF_LIFE: Shelf life of product (months)
- CATALOG_PRICE: Unit catalog price
- MONTHS_IN_FUTURE_FOR_AMC: Look-ahead months for Average Monthly Consumption
- MONTHS_IN_PAST_FOR_AMC: Look-back months for AMC
- DISTRIBUTION_LEAD_TIME: Time between receipt and lowest-level distribution
- FORECAST_ERROR_THRESHOLD: Acceptable forecast error margin
- TRANS_DATE: Month for which record applies
- AMC: Average Monthly Consumption (demand measure)
- AMC_COUNT: Months used to calculate AMC
- MOS: Months of Stock available
- MIN_STOCK_QTY: Minimum stock quantity (safety stock)
- MIN_STOCK_MOS: Minimum months of stock
- MAX_STOCK_QTY: Maximum stock quantity
- MAX_STOCK_MOS: Maximum months of stock
- OPENING_BALANCE: Beginning stock for period
- SHIPMENT_QTY: Quantity of shipment expected this month
- FORECASTED_CONSUMPTION_QTY: Expected consumption
- ACTUAL_CONSUMPTION_QTY: Actual historical consumption
- EXPIRED_STOCK: Stock expired during this month
- CLOSING_BALANCE: Ending stock level
- UNMET_DEMAND: Unfulfilled demand due to stockouts
- PLAN_BASED_ON: Whether plan is based on QTY or MOS
- VERSION_STATUS: Version status (Pending Approval, Approved, etc.)
- VERSION_TYPE: Version type (Draft, Final)
"""

# -------------------------------
# FastAPI setup
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
        # If there's a newline after the opening fence, remove everything up to that newline
        newline_idx = text.find("\n")
        if newline_idx != -1:
            text = text[newline_idx + 1 :]
        else:
            # No newline, just remove the three backticks
            text = text[3:]

    # Remove a trailing closing fence if present
    if text.endswith("```"):
        text = text[:-3].rstrip()

    return text.strip()


@app.post("/generate-dashboard")
async def generate_dashboard(req: Request):
    body = await req.json()
    prompt = body["prompt"]
    print(f"Received prompt: {prompt}")

    # -------------------------------
    # Step 1: Generate SQL using OpenAI
    # -------------------------------
    sql_resp = openaiclient.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=2500,
        messages=[
            {
                "role": "system",
            "content": (
                "You are a MySQL 8.0.43 expert. Generate SELECT queries for table `agentic_program_data` containing supply chain data.\n\n"
                "# Available Columns (USE ONLY THESE):\n"
                "PROGRAM_NAME, PROGRAM_CODE, COUNTRY_CODE, COUNTRY, HEALTH_AREA, HEALTH_AREA_CODE, "
                "ORGANISATION_CODE, ORGANISATION_NAME, PLANNING_UNIT_NAME, FORECASTING_UNIT_NAME, "
                "PLANNING_UNIT_ID, REORDER_FREQUENCY_IN_MONTHS, MIN_MONTHS_OF_STOCK, "
                "LOCAL_PROCUREMENT_LEAD_TIME, SHELF_LIFE, CATALOG_PRICE, MONTHS_IN_FUTURE_FOR_AMC, "
                "MONTHS_IN_PAST_FOR_AMC, DISTRIBUTION_LEAD_TIME, FORECAST_ERROR_THRESHOLD, "
                "TRANS_DATE, AMC, AMC_COUNT, MOS, MIN_STOCK_QTY, MIN_STOCK_MOS, "
                "MAX_STOCK_QTY, MAX_STOCK_MOS, OPENING_BALANCE, SHIPMENT_QTY, "
                "FORECASTED_CONSUMPTION_QTY, ACTUAL_CONSUMPTION_QTY, EXPIRED_STOCK, "
                "CLOSING_BALANCE, UNMET_DEMAND, PLAN_BASED_ON, VERSION_STATUS, VERSION_TYPE, VERSION_ID\n\n"
                "# IMPORTANT Column Name Clarifications:\n"
                "- For minimum months of stock threshold → Use: MIN_STOCK_MOS\n"
                "- For maximum months of stock threshold → Use: MAX_STOCK_MOS\n"
                "- For minimum months of stock setting → Use: MIN_MONTHS_OF_STOCK\n"
                "- For current months of stock value → Use: MOS\n"
                "- For projected expiry → Use: EXPIRED_STOCK\n\n"

                "# Field Descriptions:\n"
                "- PROGRAM_CODE: Program identifier (Country-HealthArea-Org format)\n"
                "- PLANNING_UNIT_ID: Product/SKU identifier\n"
                "- PLANNING_UNIT_NAME: Product/SKU name\n"
                "- TRANS_DATE: Transaction month (DATE type)\n"
                "- MOS: Months of Stock available - current inventory health metric\n"
                "- MIN_STOCK_MOS: Minimum MoS threshold (use this column, NOT 'MIN_MONTHS_OF_STOCK')\n"
                "- MAX_STOCK_MOS: Maximum MoS threshold (use this column, NOT 'MAX_MONTHS_OF_STOCK')\n"
                "- MIN_MONTHS_OF_STOCK: Minimum MoS setting from program configuration\n"
                "- CLOSING_BALANCE: Ending inventory quantity\n"
                "- OPENING_BALANCE: Starting inventory quantity\n"
                "- SHIPMENT_QTY: Incoming shipment quantity\n"
                "- ACTUAL_CONSUMPTION_QTY: Historical demand\n"
                "- FORECASTED_CONSUMPTION_QTY: Projected demand\n"
                "- EXPIRED_STOCK: Quantity expired (this is 'projected expiry' in user language)\n"
                "- UNMET_DEMAND: Stockout quantity\n"
                "- AMC: Average Monthly Consumption\n"
                "- VERSION_ID: Data version (higher = newer)\n"
                "- VERSION_STATUS: Approval status\n"
                "- VERSION_TYPE: 'Draft Version' or 'Final Version'\n\n"

                "# Query Rules:\n"
                "1. Output ONLY the SQL query - no markdown, no explanations\n"
                "2. Use fuzzy matching (LIKE 'value') for PROGRAM_CODE and text fields\n"
                "3. Match on both CODE and NAME fields when available\n"
                "4. For 'latest version' without type specification:\n"
                "   WHERE VERSION_ID = (SELECT MAX(VERSION_ID) FROM agentic_program_data WHERE PROGRAM_CODE LIKE 'XXX')\n"
                "5. For 'latest draft' or 'latest final', include VERSION_TYPE in both outer query and subquery:\n"
                "   WHERE VERSION_ID = (SELECT MAX(VERSION_ID) FROM agentic_program_data WHERE PROGRAM_CODE LIKE 'XXX' AND VERSION_TYPE = 'Draft Version')\n"
                "6. Do NOT filter VERSION_STATUS unless user explicitly mentions approval status\n"
                "7. For date ranges: TRANS_DATE BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'\n"
                "8. To determine stock status using CASE statements, use these EXACT column names:\n"
                "   CASE \n"
                "     WHEN MOS = 0 THEN 'Out of Stock'\n"
                "     WHEN MOS < MIN_STOCK_MOS THEN 'Understock'\n"
                "     WHEN MOS > MAX_STOCK_MOS THEN 'Overstock'\n"
                "     ELSE 'Adequate Stock'\n"
                "   END AS stock_status\n"
                "   NOTE: Use MIN_STOCK_MOS and MAX_STOCK_MOS (not MIN_MONTHS_OF_STOCK or MAX_MONTHS_OF_STOCK)\n"
                "9. Always include relevant columns for the analysis, use GROUP BY/ORDER BY for reports\n"
                "10. When user asks for 'consumption, inventory and shipment data', include:\n"
                "    ACTUAL_CONSUMPTION_QTY, FORECASTED_CONSUMPTION_QTY, OPENING_BALANCE, CLOSING_BALANCE, SHIPMENT_QTY\n"
                "11. NEVER use columns not in the Available Columns list above"

                )
            },
            {"role": "user", "content": prompt}
        ]
    )

    sql_query = sql_resp.choices[0].message.content.strip()
    print(f"Generated SQL: {sql_query}")

    # -------------------------------
    # Step 2: Send SQL to N8N webhook
    # -------------------------------
    n8n_resp = requests.post(N8N_WEBHOOK_URL, json={"sql": sql_query})
    n8n_resp.raise_for_status()
    data = n8n_resp.json()
    print(f"N8N response data: {data}")

    # -------------------------------
    # Step 3: Analyze data using Claude via AWS Bedrock
    # -------------------------------
    system_prompt = (
        "You are a world-class Forecasting & Supply Planning Expert with deep knowledge of production planning, "
        "demand forecasting, inventory optimization, and supply chain analytics. "
        f"The user prompt is: {prompt}\n\n"
        "Analyze the provided dataset and respond concisely.\n"
        "Think step-by-step like a planning consultant.\n\n"
        "You MUST respond with a valid JSON object only:\n"
        "{\n"
        '  "requestType": "1-Simple English" or "2-Data Table" or "3-Visualization",\n'
        '  "data": "..." or [...],\n'
        '  "visualization": "...",\n'
        '  "answer": "..."\n'
        "}"
    )

    # Prepare request body for Bedrock
    bedrock_body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4000,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": f"Analyze this dataset and respond in JSON only:\n{json.dumps(data, indent=2)}"
            }
        ]
    })

    # Invoke Claude via Bedrock
    bedrock_response = bedrock_runtime.invoke_model(
        modelId="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        body=bedrock_body
    )

    # Parse response
    response_body = json.loads(bedrock_response['body'].read())
    analysis_response = response_body['content'][0]['text'].strip()
    cleaned_analysis = extract_json_from_markdown(analysis_response)

    try:
        parsed_analysis = json.loads(cleaned_analysis)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        parsed_analysis = {
            "requestType": "1-Simple English",
            "data": "Error parsing analysis response",
            "visualization": "",
            "answer": analysis_response
        }

    # -------------------------------
    # Step 4: Return SQL + result + analysis
    # -------------------------------
    return {
        "query": sql_query,
        "data": data,
        "analysis": parsed_analysis
    }

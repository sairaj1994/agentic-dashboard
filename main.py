from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import json
from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv(dotenv_path=".env.local")
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize clients
openaiclient = OpenAI(api_key=openai_key)
claudeclient = Anthropic(api_key=anthropic_key)

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
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
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
                    f"You are a Database expert with access to MySQL 8.0.43. "
                    f"You have access to table `{ALLOWED_TABLE}` containing Supply Plan information. "
                    f"Use this schema to generate SQL queries:\n\n{DB_SCHEMA_EXPLANATION}\n\n"
                    "Users will write prompts in plain English. Your job is to generate a single **safe MySQL SELECT statement** "
                    "that accurately retrieves data from the provided table. "
                    "If version status or type is not mentioned, do not include them in WHERE clause. "
                    "Always output only the SQL query (no markdown, no explanation). "
                    "When a user refers to ‘months of stock’, use `MOS`. "
                    "When a user refers to ‘maximum months of stock’, use `MAX_STOCK_MOS`. "
                    "Avoid creating non-existent fields like MAX_MONTHS_OF_STOCK. "
                    "Use existing fields from schema only. "
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
    # Step 3: Analyze data using Claude
    # -------------------------------
    analysis = claudeclient.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=(
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
        ),
        messages=[
            {"role": "user", "content": f"Analyze this dataset and respond in JSON only:\n{json.dumps(data, indent=2)}"}
        ]
    )

    analysis_response = analysis.content[0].text.strip()
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

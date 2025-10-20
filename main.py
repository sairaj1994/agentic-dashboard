from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import json
from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env.local")
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize clients
openaiclient = OpenAI(api_key=openai_key)
claudeclient = Anthropic(api_key=anthropic_key)

ALLOWED_TABLE = os.getenv("ALLOWED_TABLE", "agentic_program_data")
DEFAULT_MAX_ROWS = int(os.getenv("DEFAULT_MAX_ROWS", "1000"))

# FastAPI setup
app = FastAPI()
N8N_WEBHOOK_URL = "http://localhost:5678/webhook-test/agentic-sql"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
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
                    "You are a Database expert. You have been provided with access to a table `agentic_program_data`. The database version is MySQL 8.0.43. "
                    "This table contains Supply Plan information for different Country Health Care Programs from around the world.  Each Program consists of Planning Units or Products. Data is provided for different Months for each Planning Unit. The DB_SCHEMA_EXPLANATION gives you an explanation of what each field is for. Use only these fields when referring to the table."
                    "Each Program consists of Planning Units or Products. Data is provided for different Months for each Planning Unit. "
                    "The DB_SCHEMA_EXPLANATION gives you an explanation of what each field is for. Use only these fields when referring to the table.\n\n"
                    "DB_SCHEMA_EXPLANATION = '# Database Field Definitions for Supply Chain and Forecasting Expert ### "
                    "- PROGRAM_NAME - Name of the Program. Generally combination of Country, Health Area and Organisation."
                    "-PROGRAM_CODE - Code of the Program. Generally combination of Country. Health Area and Organisation."
                    "- COUNTRY_CODE - 3 digit Code of the Country."
                    "-COUNTRY - Name of the Country."
                    "-HEALTH_AREA - Name of Health Area/Technical Area"
                    "-HEALTH_AREA_CODE - 3 to 10 digit code of Health Area/Technical Area."
                    "-ORGANISATION_CODE - Name of Organisation"
                    "-ORGANISATION_NAME - 3 to 10 digit code of Organisation."
                    "-PLANNING_UNIT - Name of the Planning Unit (product/SKU)."
                    "- FORECASTING_UNIT_NAME: Name of the Forecasting Unit. "
                    "- PLANNING_UNIT_ID: Id for the Planning Unit. "
                    "- REORDER_FREQUENCY_IN_MONTHS - Reorder level settings for this Planning Unit and Program (reorder cycle) i.e How many months between shipments? "
                    "- MIN_MONTHS_OF_STOCK: Minimum MoS setting. "
                    "-MIN_MONTHS_OF_STOCK - Minimum MoS settings for this Planning Unit and Program."
                    "-LOCAL_PROCUREMENT_LEAD_TIME - Local Procurement Lead Time for shipments to be procured if procured by local procurement agent"
                    "-SHELF_LIFE - Shelf Life of the product in months"
                    "- CATALOG_PRICE: Price of the product."
                    "- MONTHS_IN_FUTURE_FOR_AMC: Look-ahead period for Average Monthly Consumption calculation. Including current month."
                    "- MONTHS_IN_PAST_FOR_AMC: Look-back period for AMC (Average Monthly Consumption calculation)."
                    "- DISTRIBUTION_LEAD_TIME: How many months does it take between shipment receipt and the product to be distributed down to the lowest level? Used for suggested shipments ahead of understock. "
                    "- FORECAST_ERROR_THRESHOLD: Threshold for forecast error. "
                    #"- TRANS_DATE - Month for which this record is for (temporal dimension)."
                    "- MONTH: Month for which this record is for (temporal dimension). "
                    "- AMC: Average Monthly Consumption calculated and stored for this Month (a measure of baseline demand). "
                    "- AMC_COUNT: Number of months of data used for AMC (Average Monthly Consumption calculation). "
                    "- MOS: Months of Stock available for this Product (Inventory health metric). 0 indicates Out of Stock, Below Min MOS indicates under stock, Above Max MOS (Min Mos + Reorder Level) indicates overstock, Between Min MOS and Max MOS indicates Adaqute stock. "
                    "- MIN_STOCK_QTY: Minimum Stock Quantity (safety stock in units). Any value below this indicates an undersupply risk. "
                    "- MIN_STOCK_MOS: Minimum Months of Stock (safety stock in time). Any value below this indicates an undersupply risk. "
                    "- MAX_STOCK_QTY: Maximum Stock Quantity (inventory ceiling in units). Any value above this indicates an oversupply/obsolescence risk."
                    "- MAX_STOCK_MOS: Maximum Months of Stock (inventory ceiling in time). Any value above this indicates an oversupply/obsolescence risk."
                    "- OPENING_BALANCE: Beginning stock level for the period. "
                    "- SHIPMENT_QTY: Quantity of Shipment expected to be received in this month (inbound supply). "
                    "- FORECASTED_CONSUMPTION_QTY: Future expected demand. "
                    "- ACTUAL_CONSUMPTION_QTY: Actual historical demand. "
                    "- EXPIRED_STOCK: Amount of Stock that Expired this month (a measure of waste/obsolescence)."
                    "- CLOSING_BALANCE: Ending stock level for the period."
                    "- UNMET_DEMAND: Amount of demand that was there but was Unmet due to a stock-out. Occurs when the closing balance is '0' and part of consumption was unmet, or when adjusted consumption accounts for days stocked out. Unmet demand can also happen when a negative manual adjustment is larger than the projected ending balance."
                    "-PLAN_BASED_ON - Should minimum and maximum inventory parameters be based on QTY or months of stock (MOS)? Most products are better planned by MOS, while some low consumption, higher expiry products are better planned by quantity."
                    "PLANNING_UNIT_ID - Id for the Planning Unit."
                    "-VERSION_STATUS - Version Status Desc (Pending Approval, Approved, Needs Revision, No Review Needed)"
                    "-VERSION_TYPE - Version Type Desc (Draft Version, Final Version)"
                    "VERSION_ID - Version Id"
                    "Below are the formulas that can be used 1) Forecast Error is calculated using the Weighted Absolute Percentage Error (WAPE). WAPE is used over MAPE (Mean Absolute Percentage Error) as it can account for when consumption is intermittent or low. 2) The WAPE formula uses the previous 3-12 months of data depending on the selection in the Time Window dropdown. For example, if the ‘Time Window’ selected is 6 months, then 6 months of actual consumption and 6 months of forecasted consumption is used"
        
                    "Users will write out prompts in simple English, your job is to generate a single **safe MySQL SELECT statement** which will pull the relevant data from the provided table based on these prompts. "

                    "Create queries that can be used in mysql 8.0.4 database. In case you have both code and name for something like organisation code and name then try matching with code and name both and if possible use fuzzy match for matching in where conditions. "
                    "You should generate a complex sql query using GROUP BY ORDER BY etc, in case the user prompt intents for a report "
                    "If version status is not specified do not add that in the where clause." 
                    "If version Type is not specified then do not add that in the where clause. "
                    "Give direct results and related columns do not only intermediate columns or final columns."
                    "Be careful of the version that is asked in the prompt. For latest version use data that is there for max version Id. If no version is specified use latest version"
                    "generate a query which will show the month wise records for that product. "
                    "Mandatory : Always output only the SQL query, without explanations or extra text and without any markdown code blocks. "
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
        # model="claude-sonnet-4-20250514",
        model="claude-sonnet-4-20250514",
        max_tokens=4000,  # ✅ More tokens for detailed analysis
        system=(
            "You are a world-class **Forecasting & Supply Planning Expert** with deep knowledge of production planning, "
            "demand forecasting, inventory optimization, and supply chain analytics. "
            "You are analyzing data extracted from a MySQL table that contains historical data for Planning units from specific programs.\n\n"
            f"The prompt asked by the user is: {prompt}\n\n. You job is to analyze the data and provide insights to the user query.Keep your answer short and precise.\n\n"
            "Think step-by-step like a supply chain planning consultant, not a generic analyst. "
            "Respond to the user prompt query with a precise answer by using the attached dataset.\n\n"
            "CRITICAL: You MUST respond with ONLY a valid JSON object. "
            "Do NOT wrap your response in markdown code blocks (```json or ```). "
            "Do NOT include any explanatory text before or after the JSON.\n\n"
            "The JSON must have this exact structure:\n"
            "{\n"
            '  "requestType": "1-Simple English" or "2-Data Table" or "3-Visualization",\n'
            '  "data": "..." or [...],  // either simple english string or data table array\n'
            '  "visualization": "...",  // description of visualization for the data\n'
            '  "answer": "..."  // detailed analysis of the visualized data\n'
            "}\n\n"
            f"Here is the dataset to analyze:\n{json.dumps(data, indent=2)}"
        ),
        messages=[
            {"role": "user", "content": f"Analyze this dataset and respond with JSON only:\n\n{json.dumps(data, indent=2)}"}
        ]
    )
    
    print(f"Claude analysis response: {analysis}")
    
    # ✅ Extract and parse JSON response
    analysis_response = analysis.content[0].text.strip()
    cleaned_analysis = extract_json_from_markdown(analysis_response)
    
    try:
        parsed_analysis = json.loads(cleaned_analysis)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Raw analysis response: {analysis_response}")
        # Return a default structure if parsing fails
        parsed_analysis = {
            "requestType": "1-Simple English",
            "data": "Error parsing analysis response",
            "visualization": "",
            "answer": analysis_response
        }
    
    # ✅ Step 4: Return SQL + result + analysis back to frontend/CLI
    return {
        "query": sql_query, 
        "data": data,
        "analysis": parsed_analysis
    }


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import json

client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1",
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InvoiceRequest(BaseModel):
    document_id: str
    text: str
    schema: dict


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/api")
def extract_invoice(req: InvoiceRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
You are an expert invoice information extraction system.

Extract the invoice into JSON that STRICTLY matches the supplied JSON Schema.

Rules:
- Return ONLY JSON.
- Follow the schema exactly.
- Do not invent values.
- vendor: preserve the name exactly, but omit trailing punctuation such as '.', ',', ';', or ':'.
- currency: output ISO 4217 code (USD, EUR, GBP, INR, JPY).
- total_amount: integer in the main currency unit.
- invoice_date: YYYY-MM-DD.
- due_in_days: integer.
- is_paid: boolean.
- priority: one of low, normal, high, urgent.
- contact_email: lowercase.
- line_items: preserve order.
- item_count: equals len(line_items).
""",
                },
                {
                    "role": "user",
                    "content": req.text,
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "invoice_schema",
                    "schema": req.schema,
                    "strict": True,
                },
            },
        )

        result = json.loads(response.choices[0].message.content)

        # ----------------------------
        # Normalization
        # ----------------------------

        if "vendor" in result and isinstance(result["vendor"], str):
            result["vendor"] = result["vendor"].rstrip(".,;: ").strip()

        if "contact_email" in result and isinstance(result["contact_email"], str):
            result["contact_email"] = result["contact_email"].strip().lower()

        if "currency" in result and isinstance(result["currency"], str):
            result["currency"] = result["currency"].strip().upper()

        if "priority" in result and isinstance(result["priority"], str):
            result["priority"] = result["priority"].strip().lower()

        if "line_items" not in result or result["line_items"] is None:
            result["line_items"] = []

        # Normalize numeric fields
        result["item_count"] = len(result["line_items"])

        for item in result["line_items"]:
            if "quantity" in item:
                item["quantity"] = int(item["quantity"])
            if "unit_price" in item:
                item["unit_price"] = int(item["unit_price"])

        if "total_amount" in result:
            result["total_amount"] = int(result["total_amount"])

        if "due_in_days" in result:
            result["due_in_days"] = int(result["due_in_days"])

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
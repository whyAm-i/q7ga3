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


@app.post("/extract")
def extract_invoice(req: InvoiceRequest):
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert invoice extraction system.\n"
                        "Extract the requested fields from the invoice.\n"
                        "Return ONLY valid JSON.\n"
                        "Follow the supplied JSON schema exactly.\n"
                        "Do not include markdown or explanations."
                    ),
                },
                {
                    "role": "user",
                    "content": req.text,
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "invoice_schema",
                    "schema": req.schema,
                    "strict": True,
                }
            },
        )

        return json.loads(response.output_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
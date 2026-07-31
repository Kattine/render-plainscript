import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import asyncio

HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are a medical text simplifier. Rewrite the following medical text "
    "into plain language that a patient with no medical background can "
    "understand. Preserve all key findings and conclusions. Do not add "
    "information not present in the original text. Reply with the rewrite only."
)

app = FastAPI()

async def call_hf(text: str, system: str) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.post(API_URL, 
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={
                "model": "mistralai/Mistral-7B-Instruct-v0.3",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": text},
                ],
                "max_tokens": 400,
                "temperature": 0.7,
            }
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()

@app.post("/rewrite")
async def rewrite(request: Request):
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "No text"}, status_code=400)
    try:
        before, after = await asyncio.gather(
            call_hf(text, "Rewrite the following medical text in simpler words. Reply with the rewrite only."),
            call_hf(text, SYSTEM_PROMPT),
        )
        return {"before": before, "after": after}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, encoding="utf-8") as f:
        return f.read()
# km

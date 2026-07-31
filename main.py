"""
PlainScript — Medical Rewriter
FastAPI backend for Render deployment.
Loads Qwen2-1.5B-Instruct + LoRA adapter from HF Hub.
"""

import os
import torch
import torch.distributed
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import snapshot_download

# ── DTensor patch ────────────────────────────────────────────────────────────
if not hasattr(torch.distributed, "tensor"):
    class _FakeTensorModule:
        class DTensor:
            pass
    torch.distributed.tensor = _FakeTensorModule()

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen2-1.5B-Instruct"
ADAPTER_ID  = os.environ.get("ADAPTER_ID", "zkmine/plainscript-adapter")
HF_TOKEN    = os.environ.get("HF_TOKEN")          # set in Render environment

SYSTEM_PROMPT = (
    "You are a medical text simplifier. Rewrite the following medical text "
    "into plain language that a patient with no medical background can "
    "understand. Preserve all key findings and conclusions. Do not add "
    "information not present in the original text."
)

# ── Load model at startup ─────────────────────────────────────────────────────
print("Downloading adapter from Hub...")
adapter_dir = snapshot_download(
    repo_id=ADAPTER_ID,
    token=HF_TOKEN,
    local_dir="/tmp/adapter",
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base, adapter_dir)
model.eval()
print("Model ready.")

DEVICE = next(model.parameters()).device

# ── Generation ────────────────────────────────────────────────────────────────
def _generate(text: str, max_new_tokens: int = 400) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.post("/rewrite")
async def rewrite(request: Request):
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)
    try:
        with model.disable_adapter():
            before = _generate(text)
        after = _generate(text)
        return {"before": before, "after": after}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", encoding="utf-8") as f:
        return f.read()

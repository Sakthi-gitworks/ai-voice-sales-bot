import json, os, re
from fastapi import FastAPI
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI()
client = OpenAI(api_key=os.getenv("LLM_API_KEY", ""),
                base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"))
MODEL = os.getenv("LLM_MODEL", "gemini-flash-latest")
STATUSES = ["Converted", "Interested", "Follow-up Required", "Not Interested", "Demo Requested", "No Response"]


class Chat(BaseModel):
    product: str
    language: str
    history: list[dict]


class Summ(BaseModel):
    product: str
    transcript: str


@app.get("/")
def home():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.post("/api/chat")
def chat(c: Chat):
    system = (f"You are a friendly, persuasive sales agent on a phone call, selling this SaaS product:\n{c.product}\n"
              f"Speak ONLY in {c.language}. Keep every reply to 1-2 short spoken sentences, no lists, no emojis. "
              "Greet, learn the customer's needs, explain relevant benefits, answer questions, handle objections "
              "politely, and try to close by offering a demo or sign-up. Never invent prices or facts beyond the "
              "product info. You cannot send emails, messages or links, so never promise to. If asked for details, "
              "say the team will follow up, or offer to book a demo during this call. "
              "If the customer wants to end the call, say a brief goodbye.")
    r = client.chat.completions.create(model=MODEL, messages=[{"role": "system", "content": system}] + c.history[-20:])
    return {"reply": r.choices[0].message.content.strip()}


@app.post("/api/summary")
def summary(s: Summ):
    prompt = (f"Product: {s.product}\n\nSales call transcript:\n{s.transcript}\n\n"
              "Return ONLY a JSON object with keys: summary (2-3 sentences), requirements (string), "
              "objections (string), next_steps (string), status (exactly one of: "
              + ", ".join(STATUSES) + ").")
    t = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}]).choices[0].message.content
    try:
        d = json.loads(re.search(r"\{.*\}", t, re.S).group(0))
    except Exception:
        d = {"summary": t[:500], "requirements": "", "objections": "", "next_steps": ""}
    if d.get("status") not in STATUSES:
        d["status"] = "Follow-up Required"
    return d

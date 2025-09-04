import uuid
import os
import sys, pathlib
from typing import Dict, Any
from urllib.error import HTTPError

from fastapi import FastAPI, HTTPException, Depends, Header, status, Request
from fastapi.responses import JSONResponse
from api.schemas import ChatRequest, ChatResponse, ConfirmRequest

# API Key
API_KEY = os.getenv("API_KEY")

def require_api_key(x_api_key: str = Header(default = "")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code = status.HTTP_401_UNAUTHORIZED, detail = "API key is invalid")

# Allow importing repo root modules
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

# In-memory store
PENDING: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title = "Warranty Crew")

# Lazy orchestrator
_orch = None
def get_orchestrator():
    global _orch
    if _orch is None:
        from orchestrator import WarrantyOrchestrator, CONFIDENCE_GOOD
        from self_help_agent import HomeownerHelpAgent
        from hubspot_tool import HubSpotTool
        _orch = {
            "inst": WarrantyOrchestrator(HomeownerHelpAgent(), HubSpotTool()),
            "CONFIDENCE_GOOD": float(CONFIDENCE_GOOD),
        }
    return _orch["inst"], _orch["CONFIDENCE_GOOD"]

# JSON errors handling
@app.exception_handler(Exception)
async def all_exceptions(_: Request, exc: Exception):
    return JSONResponse(
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
        content = {
            "answered": False,
            "answer": "",
            "ticket_id": None,
            "confidence": 0.0,
            "need_confirmation": True,
            "interaction_id": str(uuid.uuid4()),
            "error": str(exc),
        }
    )

# Normalized
def _normalize_result(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        answer = result.get("answer") or result.get("content") or result.get("text") or ""
        return {
            "answer": str(answer).strip(),
            "confidence": float(result.get("confidence", 0.0)),
            "resolved": bool(result.get("resolved", False)),
            "ticket_id": result.get("ticket_id"),
        }
    if result is None:
        return {"answer": "", "confidence": 0.0, "resolved": False, "ticket_id": ""}
    return {"answer": str(result).strip(), "confidence": 0.0, "resolved": False, "ticket_id": None}

# Check
@app.get("/health")
def health():
    return {"ok": True}

# Routes
@app.post("/chat", dependencies = [Depends(require_api_key)], response_model = ChatResponse)
def chat(request: ChatRequest):
    iid = str(uuid.uuid4())

    ctx = {
        "user_id": request.user_id,
        "session_id": request.session_id,
        "job_number": request.job_number,
        "last_name": request.last_name,
    }

    # Orchestrator entrypoint
    orch, CONFIDENCE_GOOD = get_orchestrator()
    raw = orch.call_self_help(request.message, ctx)
    result = _normalize_result(raw)

    # Solid answer
    if result["resolved"] or result["confidence"] >= CONFIDENCE_GOOD:
        return ChatResponse(
            answered = True,
            answer = result["answer"],
            confidence = result["confidence"],
            need_confirmation = False,
            interaction_id = iid,
            ticket_id = None,
        )

    PENDING[iid] = {"message": request.message, "ctx": ctx, "help_res": result}
    return ChatResponse(
        answered = True,
        answer = result["answer"],
        confidence = result["confidence"],
        need_confirmation = True,
        interaction_id = iid,
        ticket_id = None,
    )

@app.post("/chat/confirm", response_model = ChatResponse)
def confirm(request: ConfirmRequest):
    data = PENDING.pop(request.interaction_id, None)
    if not data:
        raise HTTPException(status_code = 404, detail = "No such interaction")

    orch, _ = get_orchestrator()

    # Helpful
    if request.helpful:
        hr = data["help_res"]
        return ChatResponse(
            answered = True,
            answer = hr["answer"]   ,
            confidence = float(hr["confidence"] ),
            need_confirmation = False,
            interaction_id = request.interaction_id,
            ticket_id = None,
        )

    # Not helpful
    ticket = orch.open_ticket(data["message"], data["help_res"], data["ctx"])
    tid = ticket.get("id") or ticket.get("ticket_id") or ticket.get("hs_object_id") or str(ticket)
    return ChatResponse(
        answered = False,
        answer = data["help_res"]["answer"],
        ticket_id = tid,
        confidence = float(ticket.get("confidence", 0.0)),
        need_confirmation = False,
        interaction_id = request.interaction_id,
    )

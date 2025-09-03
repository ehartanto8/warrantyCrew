import uuid
import sys, pathlib
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from api.schemas import ChatRequest, ChatResponse, ConfirmRequest
from orchestrator import WarrantyOrchestrator, CONFIDENCE_GOOD
from self_help_agent import HomeownerHelpAgent
from hubspot_tool import HubSpotTool

# Allow importing repo root modules
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

# In-memory store
PENDING: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title = "Warranty Crew")

orchestrator = WarrantyOrchestrator(HomeownerHelpAgent(), HubSpotTool())

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat", response_model = ChatResponse)
def chat(request: ChatRequest):
    iid = str(uuid.uuid4())

    ctx = {
        "user_id": request.user_id,
        "session_id": request.session_id,
        "job_number": request.job_number,
        "last_name": request.last_name,
    }

    # Orchestrator entrypoint
    result = orchestrator.call_self_help(request.message, ctx)

    PENDING[iid] = {"message": request.message, "ctx": ctx, "help_res": result}

    if result.get("resolved") or float(result.get("confidence", 0.0)) >= float(CONFIDENCE_GOOD):
        return ChatResponse(answered = True, answer = str(result.get("answer", "")), confidence = float(result.get("confidence", 0.0)))

    ticket = orchestrator.open_ticket(request.message, result, ctx)

    tid = ticket.get("id") or ticket.get("ticket_id") or ticket.get("hs_object_id") or str(ticket)

    return ChatResponse(
        answered = True,
        answer = str(result.get("answer", "")),
        confidence = float(result.get("confidence", 0.0)),
        need_confirmation = True,
        interaction_id = iid
    )

@app.post("/chat/confirm", response_model = ChatResponse)
def confirm(request: ConfirmRequest):
    data = PENDING.pop(request.interaction_id, None)
    if not data:
        raise HTTPException(status_code = 404, detail = "No such interaction")

    # Helpful
    if request.helpful:
        hr = data["help_res"]
        return ChatResponse(
            answered = True,
            answer = str(hr.get("answer", "")),
            confidence = float(hr.get("confidence", 0.0)),
            need_confirmation = False,
            interaction_id = request.interaction_id,
        )

    # Not helpful
    ticket = orchestrator.open_ticket(data["message"], data["help_res"], data["ctx"])
    tid = ticket.get("id") or ticket.get("ticket_id") or ticket.get("hs_object_id") or str(ticket)
    return ChatResponse(
        answered = False,
        ticket_id = tid,
        confidence = float(ticket.get("confidence", 0.0)),
        need_confirmation = False,
        interaction_id = request.interaction_id,
    )

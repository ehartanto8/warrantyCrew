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
    """
    Make agent outputs predictable:
    - returns {'answer', 'answers', 'confidence', 'resolved', 'ticket_id'}
    - pulls text from many common fields; joins lists when needed
    """
    def first_nonempty_str(*vals):
        for v in vals:
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    answers_obj = None
    answer_text = ""

    if isinstance(result, dict):
        # 1) Direct string fields commonly used by LLM libs
        answer_text = first_nonempty_str(
            result.get("answer"),
            result.get("content"),
            result.get("text"),
            result.get("message"),
            result.get("reply"),
            result.get("response"),
            result.get("output"),
            result.get("final"),
            result.get("final_answer"),
            result.get("result"),
        )

        # 2) OpenAI-style
        if not answer_text:
            try:
                msg = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                if isinstance(msg, str) and msg.strip():
                    answer_text = msg.strip()
            except Exception:
                pass

        # 3) LangChain / custom libs: a list under "answers"
        if "answers" in result and result["answers"] is not None:
            answers_obj = result["answers"]
            # If list/dict, turn into readable text if we still don't have any
            if not answer_text:
                if isinstance(answers_obj, list):
                    # pull text-ish bits out of list items
                    parts = []
                    for item in answers_obj:
                        if isinstance(item, str) and item.strip():
                            parts.append(item.strip())
                        elif isinstance(item, dict):
                            parts.append(first_nonempty_str(
                                item.get("answer"), item.get("text"), item.get("content")
                            ))
                    answer_text = "\n\n".join([p for p in parts if p]) or ""
                elif isinstance(answers_obj, dict):
                    answer_text = first_nonempty_str(
                        answers_obj.get("answer"),
                        answers_obj.get("text"),
                        answers_obj.get("content"),
                    )

        confidence = float(result.get("confidence", 0.0))
        resolved = bool(result.get("resolved", False))
        ticket_id = result.get("id") or result.get("ticket_id") or result.get("hs_object_id")
    else:
        # Unknown shape
        answer_text = str(result).strip() if result is not None else ""
        confidence = 0.0
        resolved = False
        ticket_id = None

    # Final safety net: never return None/empty silently
    if not answer_text and answers_obj is not None:
        # show a compact JSON as text if you prefer a single-string `answer`
        import json
        try:
            answer_text = json.dumps(answers_obj, separators=(",", ":"))
        except Exception:
            answer_text = str(answers_obj)

    return {
        "answer": answer_text or "",
        "answers": answers_obj,
        "confidence": confidence,
        "resolved": resolved,
        "ticket_id": ticket_id,
    }

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
        answers = result["answers"],
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
            answer = hr["answer"],
            answers = hr["answers"],
            confidence = float(hr["confidence"]),
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

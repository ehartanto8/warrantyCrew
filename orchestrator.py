from __future__ import annotations
from typing import Any, Dict

CONFIDENCE_GOOD = 0.75

def normalize_help_result(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        base = {
            "answer": "",
            "resolved": False,
            "confidence": 0.5,
            "followups": [],
            "title": "Homeowner issue"
        }
        base.update(raw)

        # Check exist
        base["answer"] = str(base.get("answer", "") or "")
        base["resolved"] = bool(base.get("resolved", False))
        base["confidence"] = float(base.get("confidence", 0.5) or 0.0)
        base["followups"] = base.get("followups") or []
        base["title"] = str(base.get("title") or "Homeowner issue")

        return base
    return {
        "answer": str(raw) if raw is not None else "",
        "resolved": False,
        "confidence": 0.5,
        "followups": [],
        "title": "Homeowner issue"
    }

class WarrantyOrchestrator:
    """
    1) Asks self-help to answer
    2) If unresolved/low confidence, opens a ticket via HubSpotTool
    """
    def __init__(self, self_help_agent, hubspot_tool):
        self.self_help = self_help_agent
        self.hubspot = hubspot_tool

    def call_self_help(self, message: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        try:
            raw = self.self_help.run(user_input = message, context = ctx)
        except TypeError:
            try:
                raw = self.self_help.run(message)
            except Exception:
                raw = self.self_help(message) if callable(self.self_help) else {"answer": "", "resolved": False}
        return normalize_help_result(raw)

    def open_ticket(self, message: str, help_res: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        job = ctx.get("job_number")
        last = ctx.get("last_name")
        title = help_res.get("title")
        body = f"Homeowner message: {message}\n\nAnswered notes: {help_res.get('answer', '')}".strip()

        return self.hubspot._run(
            action = "create_ticket",
            job_number = str(job),
            last_name = str(last),
            ticket_description = str(title),
            description = str(body),
        )
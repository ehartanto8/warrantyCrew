from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    job_number: Optional[str] = None
    last_name: Optional[str] = None

class ChatResponse(BaseModel):
    answered: bool
    answer: Optional[str] = None
    ticket_id: Optional[str] = None
    confidence: Optional[float] = None
    need_confirmation: bool = False
    interaction_id: Optional[str] = None

class ConfirmRequest(BaseModel):
    interaction_id: str
    helpful: bool
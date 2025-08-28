import os
from hubspot import HubSpot
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, EmailStr, field_validator, model_validator, PrivateAttr
from typing import Literal, Optional, Type
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()

# Hardcoded for now
TEST_CONTACT_ID = "65929051"

# Hubspot Schema
class HubSpotArgs(BaseModel):
    action: Literal["create_ticket", "update_ticket", "get_ticket"] = Field(..., description = "Which HubSpot operation to perform")

    # Create a ticket
    job_number: Optional[str] = Field(None, description="Ticket for JOB#")
    last_name: Optional[str] = Field(None, description="Homeowner last name")
    ticket_description: Optional[str] = Field(None, description="Ticket description")
    description: Optional[str] = Field(None, description="Long description")
    associate_contact_id: Optional[str] = Field(None, description="HubSpot Contact ID to associate")
    associate_contact_email: Optional[EmailStr] = Field(None, description="HubSpot Contact email to associate")

    # Update ticket
    ticket_id: Optional[str] = Field(None, description="Existing ticket ID (update/get)")
    note: Optional[str] = Field(None, description="Note text to append (update)")

    # Validation
    @field_validator("job_number", "last_name", "ticket_description", "description", "note", mode="before")
    @classmethod
    def _strip_fields(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v or None
        return v

    @field_validator("ticket_id", mode="before")
    @classmethod
    def _strip_id(cls, v):
        return v.strip() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _require_by_action(self):
        if self.action == "create_ticket":
            missing = [
                k for k in ("job_number", "last_name", "ticket_description", "description")
                if getattr(self, k) in (None, "")
            ]
            if missing:
                raise ValueError(f"Missing required fields for create_ticket: {', '.join(missing)}")
        elif self.action in {"update_ticket", "get_ticket"}:
            if not self.ticket_id:
                raise ValueError("ticket_id is required for update_ticket/get_ticket")
        return self


# Helper
def _build_subject(job_number: str, last_name: str, ticket_description: str) -> str:
    # Trim to 255
    return f"R - {job_number} - {last_name} - {ticket_description}"[:255]


def _get_env_ids() -> tuple[str, str]:
    if not os.getenv("HUBSPOT_TICKETS_PIPELINE_ID") or not os.getenv("HUBSPOT_TICKETS_STAGE_ID"):
        raise RuntimeError("Missing Pipeline and Stage ID")
    return os.getenv("HUBSPOT_TICKETS_PIPELINE_ID"), os.getenv("HUBSPOT_TICKETS_STAGE_ID")


def _find_contact_id_by_email(client: Optional[HubSpot], email: str) -> Optional[str]:
    if client is None:
        return None
    req = {
        "filterGroups": [{"filters": [{"propertyName": "email", "operator": "EQ", "value": email}]}],
        "properties": ["email"],
        "limit": 1,
    }
    res = client.crm.contacts.search_api.do_search(public_object_search_request = req)
    results = getattr(res, "results", []) or []
    return results[0].id if results else None


# Implementation
class HubSpotTool(BaseTool):
    name: str = "HubSpotTicket"
    description: str = "Create, update, and query warranty tickets in HubSpot"
    args_schema: type[BaseModel] = HubSpotArgs
    return_direct: bool = False

    # Pydantic model
    _portal_id: str = PrivateAttr(default = "0")
    _client: Optional[HubSpot] = PrivateAttr(default = None)

    def __init__(self, **data):
        super().__init__(**data)
        token = os.getenv("HUBSPOT_API_KEY")
        self._portal_id = os.getenv("HUBSPOT_PORTAL_ID")
        self._client = HubSpot(access_token = token) if token else None

    def _run(
            self,
            action: Literal["create_ticket", "update_ticket", "get_ticket"],
            job_number: Optional[str] = None,
            last_name: Optional[str] = None,
            ticket_description: Optional[str] = None,
            description: Optional[str] = None,
            ticket_id: Optional[str] = None,
            note: Optional[str] = None,
    ):
        if action == "create_ticket":
            return self._create_ticket(
                job_number = job_number,
                last_name = last_name,
                ticket_description = ticket_description,
                description = description,
            )

        if action == "update_ticket":
            return self._update_ticket(ticket_id = ticket_id, note = note)

        if action == "get_ticket":
            return self._get_ticket(ticket_id = ticket_id)
        raise ValueError(f"Unknown action: {action}")

    # Operations
    def _create_ticket(
            self,
            job_number: str,
            last_name: str,
            ticket_description: str,
            description: str,
    ):
        pipeline_id, stage_id = _get_env_ids()
        subject = _build_subject(job_number, last_name, ticket_description)

        # Properties
        props = {
            "subject": subject,
            "content": description,
            "hs_pipeline": pipeline_id,
            "hs_pipeline_stage": stage_id
        }

        # Associations
        associations = [
            {
                "to": {"id": str(TEST_CONTACT_ID)},
                "types": [
                    {
                        "associationCategory": "HUBSPOT_DEFINED",
                        "associationTypeId": str(os.getenv("HUBSPOT_ASSOC_TICKET_TO_CONTACT_TYPE_ID"))
                    }
                ]
            }
        ]

        payload = {"properties": props, "associations": associations}

        # Dry modes: don't call HubSpot — just echo the payload
        if os.getenv("HUBSPOT_MODE") in ("read_only", "dry_run"):
            now = datetime.now(timezone.utc).isoformat()
            print(f"Would CREATE ticket with payload:\n{payload}")
            return {
                "status": os.getenv("HUBSPOT_MODE"),
                "ticket_id": "dryrun-0",
                "url": f"https://app.hubspot.com/contacts/{self._portal_id}/tickets/dryrun-0",
                "payload": payload,
                "createdAt": now,
            }

        if not self._client:
            raise RuntimeError("Missing HUBSPOT TOKEN")

        created = self._client.crm.tickets.basic_api.create(
            simple_public_object_input = {"properties": props, "associations": associations}
        )

        created_at = getattr(created, "createdAt", None) or getattr(created, "createdAt", None)

        out = {
            "ticket_id": created.id,
            "url": f"https://app.hubspot.com/contacts/{self._portal_id}/tickets/{created.id}",
            "properties": props,
            "associations": associations,
        }

        if created_at:
            out["createdAt"] = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
        return out

    def _update_ticket(self, ticket_id: str, note: str):
        if os.getenv("HUBSPOT_MODE") in ("read_only", "dry_run"):
            print(f"Would UPDATE ticket {ticket_id} with note:\n{note}")
            return {"status": os.getenv("HUBSPOT_MODE"), "ticket_id": ticket_id}

        if not self._client:
            raise RuntimeError("Missing HUBSPOT TOKEN")

        self._client.crm.tickets.basic_api.update(
            ticket_id = ticket_id,
            simple_public_object_input = {"properties": {"content": note}},
        )
        return {"status": "updated", "ticket_id": p["ticket_id"]}

    def _get_ticket(self, ticket_id: str):
        if os.getenv("HUBSPOT_MODE") in ("read_only", "dry_run") or str(ticket_id).startswith("dryrun"):
            print(f"[HUBSPOT:{os.getenv('HUBSPOT_MODE')}] Would GET ticket {ticket_id}")
            return {"status": os.getenv("HUBSPOT_MODE"), "ticket_id": ticket_id}

        if not self._client:
            raise RuntimeError("HubSpot client not initialized (missing HUBSPOT_TOKEN).")

        ticket = self._client.crm.tickets.basic_api.get_by_id(ticket_id)
        created_at = getattr(ticket, "createdAt", None) or getattr(ticket, "createdAt", None)
        return {
            "ticket_id": ticket.id,
            "properties": ticket.properties,
            "createdAt": created_at.isoformat() if hasattr(ticket, "isoformat") else str(ticket.createdAt) if created_at else None
        }

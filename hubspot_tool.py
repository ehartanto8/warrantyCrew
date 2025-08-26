import os
from hubspot import HubSpot
from crewai_tools.base_tool import BaseTool

class HubSpotTool(BaseTool):
    name = "HubSpotTicket"
    description = "Create, update, and query warranty tickets in HubSpot"

    schema = {
        "create_ticket": {
            "title": "Create a new warranty ticket",
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "description": {"type": "string"},
                "customer_email": {"type": "string"},
            },
            "required": ["subject", "description", "customer_email"],
        },
        "update_ticket": {
            "title": "Update an existing warranty ticket",
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["ticket_id", "note"],
        },
        "get_ticket": {
            "title": "Get the detail of an existing warranty ticket",
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"},
            },
            "required": ["ticket_id"],
        }
    }

    def __init__(self):
        super().__init__()
        token = os.getenv("HUBSPOT_API_KEY")
        self.portal_id = os.getenv("HUBSPOT_PORTAL_ID")
        self.client = HubSpot(api_key=token)

    def _run(self, action: str, params: dict):
        if action == "create_ticket":
            return self._create_ticket(params)
        if action == "update_ticket":
            return self._update_ticket(params)
        if action == "get_ticket":
            return self._get_ticket(params)
        raise ValueError(f"Unknown action: {action}")

    def _create_ticket(self, p):
        props = {
            "subject": p["subject"],
            "content": p["description"],
            "hs_pipeline": "0",
            "hs_pipeline_stage": "1",
            "email": p["customer_email"],
        }
        ticket = self.client.crm.tickets.basic_api.create(
            simple_public_object_input={"properties": props},
        )
        return {
            "ticket_id": ticket.id,
            "url": f"https://app.hubspot.com/contacts/{self.portal_id}/tickets/{ticket.id}",
        }

    def _update_ticket(self, p):
        self.client.crm.tickets.basic_api.update(
            ticket_id=p["ticket_id"],
            simple_public_object_input={"properties": {"content": p["note"]}},
        )
        return {"status": "updated", "ticket_id": p["ticket_id"]}

    def _get_ticket(self, p):
        ticket = self.client.crm.tickets.basic_api.get_by_id(p["ticket_id"])
        return {
            "ticket_id": ticket.id,
            "properties": ticket.properties,
        }
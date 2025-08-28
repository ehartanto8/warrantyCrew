from hubspot import HubSpot
import os
from dotenv import load_dotenv

load_dotenv()

api = HubSpot(access_token = os.getenv("HUBSPOT_API_KEY"))
types = api.crm.associations.v4.schema.definitions_api.get_all(from_object_type = "tickets", to_object_type = "contacts")

type_id = next(
    (d.type_id for d in getattr(types, "results", []) if getattr(d, "category", "").upper() == "HUBSPOT_DEFINED"), None
)

print("associationTypeId:", type_id)
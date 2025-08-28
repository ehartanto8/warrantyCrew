from dotenv import load_dotenv
load_dotenv()

from hubspot_tool import HubSpotTool

tool = HubSpotTool()

resp = tool._run(
    action = "create_ticket",
    job_number = "TASF0036",
    last_name = "Hartanto",
    ticket_description = "Ticket creation for TASF0036",
    description = "Ticket creation for TASF0036",
)

print("\n== CREATE RESULT ==")
print("ticket_id:", resp.get("ticket_id"))
print("url      :", resp.get("url"))

import json
if "payload" in resp:
    print("\n== PAYLOAD ==")
    print(json.dumps(resp["payload"], indent=2))

# u = tool._run(
#     action = "update_ticket",
#     ticket_id = resp.get("ticket_id", "dryrun-0"),
#     note = "Adding details"
# )
# print("\n== UPDATE RESULT ==", u)

# g = tool._run(
#     action = "get_ticket",
#     ticket_id = resp.get("ticket_id", "dryrun-0"),
# )
# print("\n== GET TICKET ==", g)
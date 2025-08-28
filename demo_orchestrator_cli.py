from dotenv import load_dotenv
import os, json, sys

load_dotenv()

from self_help_agent import HomeownerHelpAgent
from orchestrator import WarrantyOrchestrator, CONFIDENCE_GOOD
from hubspot_tool import HubSpotTool

def prompt(label: str, required: bool = False, default: str | None = None) -> str:
    while True:
        val = input(f"{label}{' ['+default+']' if default else ''}: ").strip()
        if not val and default is not None:
            return default
        if val or not required:
            return val
        print(" (required)")

def yesno(label: str, default: str = "y") -> bool:
    d = default.lower()
    while True:
        val = input(f"{label} (y/n) [{d}]: ").strip().lower()
        if not val:
            return d == "Y"
        if val in ("y", "yes"): return True
        if val in ("n", "no"): return False
        print(" please enter y or n")

def main():
    job_number = prompt("Job number", required = True)
    last_name = prompt("Last name", required = True)
    email = prompt("Email", required = True)

    self_help = HomeownerHelpAgent()
    hubspot = HubSpotTool()
    orchestrator = WarrantyOrchestrator(self_help, hubspot)

    while True:
        msg = prompt("\nDescribe your issue", required = True)
        ctx = {
            "job_number": job_number,
            "last_name": last_name,
            "email": email,
            "open_ticket_if_unresolved": False
        }

        help_res = orchestrator.call_self_help(msg, ctx)

        print("\n--- Assistant answer ---")
        print(help_res["answer"])
        print(f"\n(confidence: {help_res['confidence']:.2f}, resolved flag: {help_res['resolved']}")

        if help_res.get("followups"):
            print("\nFollow-up questions:")
            for i, q in enumerate(help_res["followups"], 1):
                print(f" {i}. {q}")

        # Ask if resolve or not
        if yesno("\nDid this resolve your issue?", default = "n" if help_res["confidence"] < CONFIDENCE_GOOD else "y"):
            print("Great! No ticket needed.")
        else:
            if yesno("Open a support ticket now?", default = "y"):
                ticket = orchestrator.open_ticket(msg, help_res, ctx)
                print("\n---Ticket---")
                print(json.dumps(ticket, indent = 2))
            else:
                print("Sounds good!")

        if not yesno("\nAsk another question?", default = "n"):
            break

    print("\Bye!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
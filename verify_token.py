import os
from dotenv import load_dotenv

load_dotenv()

print("token_prefix:", (os.getenv("HUBSPOT_API_KEY") or "")[:6])
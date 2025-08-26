from dotenv import load_dotenv
import os
from crewai import Agent, Task, Crew
from hubspot_tool import HubSpotTool

load_dotenv()

# Init tool and agent
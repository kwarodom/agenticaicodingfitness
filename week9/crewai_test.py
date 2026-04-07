#creaai_test.py
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
load_dotenv()

# CrewAI reads ANTHROPIC_API_KEY automatically
# Just set the llm parameter using the model name:
my_agent = Agent(
role='Analyst'
,
goal='Analyse data and summarise findings'
,
backstory='Expert data analyst.'
,
llm='claude-haiku-4-5-20251001', # <-- key line
verbose=True,
)

print(my_agent)

# Using z.ai with CrewAI:
# llm='openai/glm-5' + set OPENAI_API_KEY=<your z.ai key>
# and OPENAI_BASE_URL=https://api.z.ai/api/paas/v4/
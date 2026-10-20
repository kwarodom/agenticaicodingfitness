import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
load_dotenv()

# ChatAnthropic reads ANTHROPIC_API_KEY automatically
llm = ChatAnthropic(model='claude-haiku-4-5-20251001')

# Using z.ai with LangGraph:
from langchain_openai import ChatOpenAI
llm_zai = ChatOpenAI(
model='glm-5'
,
api_key=os.environ['ZAI_API_KEY'],
base_url='https://api.z.ai/api/paas/v4/'
,
)

print(llm_zai)
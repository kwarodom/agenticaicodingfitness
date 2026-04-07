import os, autogen
from dotenv import load_dotenv
load_dotenv()
# AutoGen LLM config for Claude
llm_config = {
    'config_list': [{
    'model': 'claude-haiku-4-5-20251001'
    ,
    'api_key': os.environ['ANTHROPIC_API_KEY'],
    'api_type': 'anthropic'
    ,
    }],
    'cache_seed': None,
}
# AutoGen LLM config for z.ai (OpenAI-compatible)
llm_config_zai = {
    'config_list': [{
    'model': 'glm-5'
    ,
    'api_key': os.environ['ZAI_API_KEY'],
    'base_url': 'https://api.z.ai/api/paas/v4/'
    }],
    'cache_seed': None,
}
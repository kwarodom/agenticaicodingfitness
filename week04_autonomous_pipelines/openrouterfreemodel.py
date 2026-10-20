import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()

# First API call with reasoning
response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer " + os.getenv("OPENROUTER_API_KEY"),
    "Content-Type": "application/json",
  },
  data=json.dumps({
    "model": "openrouter/free",
    "messages": [
        {
          "role": "user",
          "content": "How many r's are in the word 'strawberry'?"
        }
      ],
    "reasoning": {"enabled": True}
  })
)

# Extract the assistant message with reasoning_details
response = response.json()
response = response['choices'][0]['message']
print("response {}".format(response.get('content')))

# Preserve the assistant message with reasoning_details
messages = [
  {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
  {
    "role": "assistant",
    "content": response.get('content'),
    "reasoning_details": response.get('reasoning_details')  # Pass back unmodified
  },
  {"role": "user", "content": "Are you sure? Think carefully."}
]

# Second API call - model continues reasoning from where it left off
response2 = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer " + os.getenv("OPENROUTER_API_KEY"),
    "Content-Type": "application/json",
  },
  data=json.dumps({
    "model": "openrouter/free",
    "messages": messages,  # Includes preserved reasoning_details
    "reasoning": {"enabled": True}
  })
)
response2 = response2.json()
response2 = response2['choices'][0]['message']
print("response2 {}".format(response2.get('content')))   
import os
import anthropic
from datetime import datetime
from dotenv import load_dotenv
from duckduckgo_search import DDGS

load_dotenv()

# We configure Anthropic to use the Poe endpoint and key
client = anthropic.Anthropic(
    base_url="https://api.poe.com",
    api_key=os.environ.get("POE_API_KEY")
)

# ==========================================
# 1. Define Local Python Functions (Tools)
# ==========================================
def search_web(query: str) -> str:
    """Uses DuckDuckGo to search the internet for real-time information."""
    print(f"\n[Tool Executing] Searching the web for: '{query}'...")
    with DDGS() as ddgs:
        # Grab the top 3 results from DuckDuckGo
        results = ddgs.text(query, max_results=3)
        
    if not results:
        return "No results found."
        
    # Format the results into a single string summary
    summary = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    return summary

def get_current_time() -> str:
    """Returns the current date and time."""
    print("\n[Tool Executing] Getting current time...")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ==========================================
# 2. Tell Claude what tools exist 
# ==========================================
tools = [
    {
        "name": "search_web",
        "description": "Searches the internet for real-time information, news, and facts using DuckDuckGo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The exact search query to look up on the internet (e.g. 'current weather in Tokyo')"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_current_time",
        "description": "Retrieves the user's current local date and exact time.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    }
]

# ==========================================
# 3. Implement the Chat Agent Logic
# ==========================================
def chat_with_tools(user_input: str):
    messages = [{"role": "user", "content": user_input}]
    print("User:", user_input)
    
    # Send the initial query and the schema of available tools to Claude
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        messages=messages,
        tools=tools
    )
    
    # Check if Claude's response says it needs to use a tool to answer the question
    if response.stop_reason == "tool_use":
        # Find the tool Claude wants to use
        tool_call = next(block for block in response.content if block.type == "tool_use")
        
        # We must append Claude's request to use the tool to the conversation history
        messages.append({"role": "assistant", "content": response.content})
        
        tool_name = tool_call.name
        tool_inputs = tool_call.input
        
        # Execute the appropriate local python function securely
        if tool_name == "search_web":
            tool_result = search_web(tool_inputs["query"])
        elif tool_name == "get_current_time":
            tool_result = get_current_time()
        else:
            tool_result = f"Error: Unknown tool {tool_name}"
            
        # Give the result of the function back to Claude as a new "tool_result" message
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": tool_result
                }
            ]
        })
        
        # Let Claude read the data our python function retrieved and generate the final English answer
        final_response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1000,
            messages=messages,
            tools=tools
        )
        print("\nClaude:", final_response.content[0].text)
        
    else:
        # Claude decided it didn't need to use a tool, it answered normally
        print("\nClaude:", response.content[0].text)

if __name__ == "__main__":
    # Test 1: Claude asks our python script what the time is
    print("--- Test 1 ---")
    chat_with_tools("What time is it right now?")
    
    # Test 2: Claude uses internet search to grab a real-time fact
    print("\n--- Test 2 ---")
    chat_with_tools("Who won the most recent super bowl?")

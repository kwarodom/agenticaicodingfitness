# PYTHON — agent.py — The Core Agent Framework
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()
model = "claude-haiku-4-5"

def add_user_message(messages, text):
    user_message = {"role": "user", "content": text}
    messages.append(user_message)

def add_assistant_message(messages, text):
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)

def chat(messages, temperature=1.0):
    message = client.messages.create(
        model=model,
        max_tokens=1000,
        temperature=temperature,
        messages=messages,
    )
    return message.content[0].text

# Start with an empty message list (this is our short-term memory)
messages = []

print("Chat with Claude (type 'quit' to exit)\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit"):
        print("Bye!")
        break
    if not user_input:
        continue

    add_user_message(messages, user_input)
    answer = chat(messages)
    add_assistant_message(messages, answer)

    print(f"\nClaude: {answer}\n")
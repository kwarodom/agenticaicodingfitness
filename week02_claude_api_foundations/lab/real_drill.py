# real_drill.py — the SAME memory loop as drill.py, but against live Claude.
#
# The only difference from drill.py is these two lines:
#     from dotenv import load_dotenv; load_dotenv()
#     import anthropic; client = anthropic.Anthropic()
# The loop below is byte-for-byte identical. That's the lesson: the mock and
# the real model expose the same surface, so your code doesn't change.

from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY from a .env file next to you
import anthropic

client = anthropic.Anthropic()  # reads the key from the environment

messages = []  # <-- THIS list is the memory


def chat(text):
    messages.append({"role": "user", "content": text})  # 1. add the user turn FIRST
    resp = client.messages.create(
        model="claude-haiku-4-5",  # cheapest model — fractions of a cent per turn
        max_tokens=1000,
        messages=messages,  # 2. resend EVERYTHING each time
    )
    reply = resp.content[0].text
    messages.append({"role": "assistant", "content": reply})  # 3. remember the reply
    print(f"  [{resp.usage.input_tokens} in / {resp.usage.output_tokens} out]")
    return reply


if __name__ == "__main__":
    print(chat("My name is Aom. Remember it."))
    print(chat("What's my name?"))  # the REAL model should answer "Aom"

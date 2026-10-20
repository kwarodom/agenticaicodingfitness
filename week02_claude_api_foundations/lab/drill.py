# drill.py — the multi-turn memory loop, runs at $0 against mock_llm.
#
# The ENTIRE point: the API is stateless. The model remembers nothing between
# calls. The conversation only continues because `messages` (below) is kept by
# YOU and resent in full every turn. That list IS the chatbot's memory.

from mock_llm import client

messages = []  # <-- THIS list is the memory


def chat(text):
    messages.append({"role": "user", "content": text})  # 1. add the user turn FIRST
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        messages=messages,  # 2. resend EVERYTHING each time
    )
    reply = resp.content[0].text
    messages.append({"role": "assistant", "content": reply})  # 3. remember the reply
    print(f"  [{resp.usage.input_tokens} in / {resp.usage.output_tokens} out]")
    return reply


if __name__ == "__main__":
    print(chat("My name is Aom."))
    print(chat("What's my name?"))  # must answer "Aom" — proof memory works

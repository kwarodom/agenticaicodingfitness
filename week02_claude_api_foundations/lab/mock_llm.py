# mock_llm.py — a fake client so the drill runs at $0, no API key needed.
#
# It mimics just enough of the real `anthropic.Anthropic()` surface for the
# memory loop: client.messages.create(...).content[0].text and .usage.
# It "remembers" ONLY because the loop resends the whole messages list each
# call — there is no hidden state here. That's the whole lesson.


class MockUsage:
    def __init__(self, n_in, n_out):
        self.input_tokens, self.output_tokens = n_in, n_out


class MockBlock:
    def __init__(self, text):
        self.text = text


class MockResponse:
    def __init__(self, text, messages):
        self.content = [MockBlock(text)]
        # Fake token count: words in the whole history (in) + words in reply (out).
        n_in = sum(len(m["content"].split()) for m in messages)
        self.usage = MockUsage(n_in, len(text.split()))


class MockMessages:
    def create(self, *, model, max_tokens, messages, system=None):
        # Fake "memory": echo back any name the history ever mentioned.
        name = next(
            (
                m["content"].split("name is ")[1].split(".")[0]
                for m in messages
                if "name is " in m["content"]
            ),
            None,
        )
        last = messages[-1]["content"]
        reply = (
            f"Your name is {name}."
            if "my name" in last.lower() and name
            else f"(mock) You said: {last}"
        )
        return MockResponse(reply, messages)


class MockLLM:
    def __init__(self):
        self.messages = MockMessages()


client = MockLLM()  # later: import anthropic; client = anthropic.Anthropic()

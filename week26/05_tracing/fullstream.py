#!/usr/bin/env python3
"""fullstream — the in-process twin of `nat serve`'s POST /v1/workflow/full (Module 05, lab 05-5 fallback).

Runs under NAT's own Python (week26/.venv-nat/bin/python), not the course venv. It loads a workflow config
and calls the SAME function the FastAPI route calls — nat.front_ends.fastapi.response_helpers.
generate_streaming_response_full — and prints the SSE lines the endpoint would send (`intermediate_data: …`,
`data: …`). Lab 05-5 uses it only when `nat serve` cannot start on this laptop, and says so.

It keeps the event loop alive for a moment after the run, so NAT's exporters (which do not wait for their
background tasks on stop) finish writing the trace file.

    week26/.venv-nat/bin/python week26/05_tracing/fullstream.py <config.yml> "<question>" LLM_END,TOOL_END
"""
import asyncio
import sys

from nat.front_ends.fastapi.response_helpers import generate_streaming_response_full
from nat.runtime.loader import load_workflow


async def main(cfg: str, question: str, filter_steps: str) -> None:
    async with load_workflow(cfg) as session_manager:
        async with session_manager.session() as session:
            async for item in generate_streaming_response_full({"input_message": question}, session=session,
                                                               streaming=True, filter_steps=filter_steps or None):
                sys.stdout.write(item.get_stream_data())
                sys.stdout.flush()
        await asyncio.sleep(2.0)                     # let the file exporter finish its background writes


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""))

#!/usr/bin/env python3
"""Build a NAT workflow's pieces WITHOUT an LLM and call them directly (runs under week26/.venv-nat/bin/python).

    nat_direct.py tool  <config.yml> <function> '<json args>' ['<json args>' …]   → call a function, print its output
    nat_direct.py group <config.yml> <group>                                        → what an agent would get from a group
    nat_direct.py imports                                                           → the 1.9 import paths, side by side

Used by labs 04-2 and 04-4 to get ground truth (0 LLM calls) before asking the agent.
"""
import asyncio
import json
import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)


async def tool(cfg_path: str, name: str, arg_sets: list[str]) -> None:
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.runtime.loader import load_config
    cfg = load_config(cfg_path)
    async with WorkflowBuilder() as b:
        await b.add_function(name, cfg.functions[name])
        fn = await b.get_function(name)
        print(f"description: {fn.description}")
        print(f"input schema: {json.dumps(fn.input_schema.model_json_schema().get('properties', {}))}")
        for a in arg_sets:
            print(f"{name}({a}) -> {await fn.ainvoke(json.loads(a), to_type=str)}")


async def group(cfg_path: str, name: str) -> None:
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.runtime.loader import load_config
    cfg = load_config(cfg_path)
    gc = cfg.function_groups[name]
    print(f"config: include={gc.include} exclude={gc.exclude} "
          f"tool_overrides={sorted((gc.tool_overrides or {}).keys())}")
    async with WorkflowBuilder() as b:
        await b.add_function_group(name, gc)
        g = await b.get_function_group(name)
        every = await g.get_all_functions()
        reach = await g.get_accessible_functions()
        for k, f in sorted(every.items()):
            print(f"{'AGENT SEES' if k in reach else 'hidden    '}  {k}  | {f.description}")


def imports() -> None:
    import nat.plugin_api as api
    from nat.builder.builder import Builder
    from nat.builder.function_info import FunctionInfo
    from nat.cli.register_workflow import register_function
    from nat.data_models.function import FunctionBaseConfig
    for short, obj in (("Builder", Builder), ("FunctionInfo", FunctionInfo), ("register_function", register_function),
                       ("FunctionBaseConfig", FunctionBaseConfig)):
        print(f"nat.plugin_api.{short} is {obj.__module__}.{short}: {getattr(api, short) is obj}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "tool":
        asyncio.run(tool(sys.argv[2], sys.argv[3], sys.argv[4:] or ["{}"]))
    elif cmd == "group":
        asyncio.run(group(sys.argv[2], sys.argv[3]))
    else:
        imports()

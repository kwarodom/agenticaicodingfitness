import json, sys
def read():
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}
def cmd(d):
    return str((d.get("tool_input") or {}).get("command", ""))
def block(msg):
    print("blocked: " + msg, file=sys.stderr); sys.exit(2)

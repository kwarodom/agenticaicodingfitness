#!/usr/bin/env python3
"""Lab 15-4 · Put the Module 14 NAT hotel agent inside an OpenShell sandbox on the Spark.

The sequence the capstone (Module 20) reuses. It combines the OpenShell playbook's Steps 5–7 and 11
(verify vLLM, create a provider, route inference, watch decisions) with a course-designed sandbox for
the NAT agent: `--from base`, our policy file, the module folder uploaded to /sandbox, NAT installed
while the setup-only pypi rule is open, then the rule removed and the agent run with its model at
https://inference.local/v1.

Safety: in DRY mode every command is printed with an EXAMPLE output. With a Spark connected, the
read-only checks run; commands that CREATE or CHANGE something on the Spark run only with
SPARK_APPLY=1 (or --yes). Deleting the sandbox additionally needs --cleanup.

Run: .venv/bin/python week25/15_openshell_sandbox/labs/lab15_4_sandbox_the_nat_agent.py [--yes] [--cleanup]
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "common"))
sys.path.insert(0, str(HERE.parents[1]))
import policykit as pk  # noqa: E402
from sparkkit import banner, note, put, result, sh, step, warn, where  # noqa: E402

MOD = HERE.parents[1]
APPLY = os.environ.get("SPARK_APPLY") == "1" or "--yes" in sys.argv
CLEANUP = "--cleanup" in sys.argv
SB = "hotel-agent"
MODEL = os.environ.get("HOTEL_LLM_MODEL", "nvidia/Qwen3.6-35B-A3B-NVFP4")     # the playbook's DGX Spark handle
REPO = "~/agenticaicodingfitness"                                                 # this repo, cloned on the Spark
ASK = "Room 808 feels hot. Check it and open a ticket if something is wrong."


def change(cmd: str, example: str, timeout: float = 900):
    """A command that creates or changes state on the Spark: DRY → EXAMPLE; LIVE → only with SPARK_APPLY=1/--yes."""
    if where() != "dry" and not APPLY:
        print(f"$ {cmd.splitlines()[0]}   [not run — add --yes or SPARK_APPLY=1]")
        return None
    return sh(cmd, example=example, timeout=timeout)


banner("Lab 15-4 · the NAT hotel agent in an OpenShell sandbox",
       f"sandbox '{SB}' · policy policies/hotel_agent_policy.yaml · model via inference.local · "
       f"{'APPLY' if APPLY else 'read-only unless --yes'}")

step(1, "the policy you are about to push is valid (policykit, offline)")
policy = pk.load(MOD / "policies" / "hotel_agent_policy.yaml")
errs, warns = pk.validate(policy)
if errs:
    for e in errs:
        print(f"✕ {e}")
    sys.exit(1)
print(f"✓ {len(policy['network_policies'])} network groups ({', '.join(policy['network_policies'])}), "
      f"{len(policy['filesystem_policy']['read_write'])} writable paths, runs as {policy['process']['run_as_user']} · "
      f"{len(warns)} warning(s)")

step(2, "vLLM is up and reachable from Docker (playbook Step 5: host IP, not localhost)")
sh('export HARDWARE_IP="$(hostname -I | awk \'{print $1}\')" && curl -sf "http://${HARDWARE_IP}:8000/v1/models" | head -c 300',
   example='{"object":"list","data":[{"id":"nvidia/Qwen3.6-35B-A3B-NVFP4","object":"model",…}]}', timeout=60)

step(3, "provider + inference route (playbook Steps 6–7)")
change('export HARDWARE_IP="$(hostname -I | awk \'{print $1}\')" && openshell provider create --name local-vllm '
       '--type openai --credential OPENAI_API_KEY=not-needed --config OPENAI_BASE_URL="http://${HARDWARE_IP}:8000/v1"',
       example="✓ provider local-vllm created")
change(f"openshell inference set --provider local-vllm --model {MODEL} && openshell inference get",
       example=f"provider: local-vllm\nmodel: {MODEL}")

step(4, "copy the policy to the Spark and create the sandbox (course-designed: --from base + upload)")
if where() == "dry" or APPLY:
    put(MOD / "policies" / "hotel_agent_policy.yaml", "~/w25/hotel_agent_policy.yaml")
else:
    print("$ scp hotel_agent_policy.yaml <spark>:~/w25/hotel_agent_policy.yaml   [not run — add --yes or SPARK_APPLY=1]")
change(f"openshell sandbox create --name {SB} --from base --policy ~/w25/hotel_agent_policy.yaml "
       f"--upload {REPO}/week25/14_nat_agents:/sandbox/14_nat_agents --keep --no-tty -- true",
       example=f"✓ sandbox {SB} created · policy applied · phase: Ready")

step(5, "install NAT inside the sandbox while the setup-only pypi rule is open")
change(f"openshell sandbox exec -n {SB} -- python3 -m venv /sandbox/.venv-nat && "
       f"openshell sandbox exec -n {SB} -- /sandbox/.venv-nat/bin/pip install -q 'nvidia-nat[langchain]~=1.9' greenlet && "
       f"openshell sandbox exec -n {SB} -- /sandbox/.venv-nat/bin/pip install -q --no-deps -e /sandbox/14_nat_agents/hotel_ops_nat",
       example="(pip output) Successfully installed … nvidia-nat-1.9.0 … hotel_ops_nat-0.1.0")
change(f"openshell policy update {SB} --remove-rule pypi --wait && openshell policy get {SB}",
       example="policy revision 2 loaded · network groups: inference")

step(6, "run the agent inside the sandbox — its only way out is inference.local")
change(f"openshell sandbox exec -n {SB} --workdir /sandbox/14_nat_agents "
       f"--env HOTEL_LLM_BASE_URL=https://inference.local/v1 --env HOTEL_LLM_MODEL={MODEL} "
       f"--env HOTEL_TICKET_LOG=/sandbox/tickets.jsonl -- /sandbox/.venv-nat/bin/nat run "
       f"--config_file configs/hotel_agent_spark.yml --input \"{ASK}\"",
       example="Workflow Result:\nRoom 808 is too warm … Created maintenance ticket MT-A83AD2 (high priority).")

step(7, "prove the fence: the same sandbox cannot reach the internet (runbook pattern: these should fail)")
change(f"openshell sandbox exec -n {SB} -- curl -s --max-time 5 https://api.openai.com || echo 'blocked ✓'",
       example="curl: (56) CONNECT tunnel failed, response 403\nblocked ✓")
sh(f"openshell logs {SB} -n 20 --source sandbox", example="… decision=deny dst=api.openai.com:443 …\n"
   "… decision=inspect_for_inference dst=inference.local:443 …", timeout=60)

step(8, "clean up (playbook Step 13) — only with --yes --cleanup")
if CLEANUP and (APPLY or where() == "dry"):
    change(f"openshell sandbox delete {SB} && openshell provider delete local-vllm", example=f"✓ deleted {SB}")
else:
    print(f"$ openshell sandbox delete {SB} && openshell provider delete local-vllm   [not run — add --yes --cleanup]")

if where() == "dry":
    warn("DRY: every output above is an EXAMPLE. This exact sequence has not been run on a Spark yet — "
         "TUTORIAL §6 lists what to check the first time.")
note("Module 20 reuses steps 3–7 unchanged, pointing HOTEL_LLM_MODEL at your fine-tuned model served by vLLM "
     "(or at LiteLLM, Module 08) — the sandbox and policy stay the same.")
result("The agent kept its two tools and its model; it lost the internet, your home directory and root.")

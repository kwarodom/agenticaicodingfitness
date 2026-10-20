#!/usr/bin/env python3
"""Lab 11-4 · Two-Spark FSDP: check both Sparks, write the Accelerate configs, print the launch.

The PyTorch playbook's "Multi-node fine-tuning" tab runs the same recipes across two Sparks
with Docker Swarm (one container per Spark), Accelerate and FSDP. This lab does the parts that
are safe to automate:

  1. read-only preflight on both Sparks: interconnect IP, Docker Swarm state, GPU UUID,
     GPU advertising in /etc/docker/daemon.json, the swarm-resource line, the recipes clone;
  2. write the two Accelerate files (machine_rank 0 on Spark A, 1 on Spark B, the primary's
     interconnect IP and a port) from the playbook's own configs, and validate them;
  3. print the exact playbook commands for the steps that need sudo or change cluster state
     (daemon.json, swarm init/join, stack deploy, launch) — you run those in the ⌨ terminal.

The generated files go to week25/11_pytorch_nemo_two_sparks/.runs/ (gitignored); in LIVE mode they
are also copied into ~/w25/dgx-spark-playbooks/…/configs/ on each Spark (the playbook's "edit the YAML").

Run: .venv/bin/python week25/11_pytorch_nemo_two_sparks/labs/lab04_two_spark_fsdp.py [--config lora|full]
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, check, note, put, result, sh, step, table, warn, where  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", choices=["lora", "full"], default="lora",
                help="lora = config_fsdp_lora.yaml (8B/70B LoRA) · full = config_finetuning.yaml (3B full SFT)")
ap.add_argument("--port", type=int, default=29500, help="main_process_port (course choice; any free port)")
args = ap.parse_args()

OUT = Path(__file__).resolve().parents[1] / ".runs"
ASSETS = "w25/dgx-spark-playbooks/nvidia/playbook-pytorch-fine-tune/assets"
COMPOSE_IF = "enp1s0f1np1"                       # docker-compose.yml's default for the three *_IFNAME variables
CFG_NAME = {"lora": "config_fsdp_lora.yaml", "full": "config_finetuning.yaml"}[args.config]
SCRIPT = {"lora": "Llama3_70B_LoRA_finetuning.py", "full": "Llama3_3B_full_finetuning.py"}[args.config]
# The 70B script defaults to Llama 3.1; the classroom Sparks cache Llama 3.3 (same architecture), so point at it
# rather than pull a second ~140 GB onto each Spark.
SCRIPT_ARGS = " --model_name meta-llama/Llama-3.3-70B-Instruct" if args.config == "lora" else ""

# The playbook's two Accelerate files, byte for byte (assets/configs/). Only the three TODO/rank lines change.
PLAYBOOK_CFG = {
    "config_fsdp_lora.yaml": """compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
machine_rank: 0
main_process_ip: < TODO: specify IP >
main_process_port: < TODO: specify port >
main_training_function: main
mixed_precision: 'bf16'
num_machines: 2
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
""",
    "config_finetuning.yaml": """compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_config:
  fsdp_activation_checkpointing: false
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_reshard_after_forward: false
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: 'LlamaDecoderLayer'
  fsdp_version: 2
machine_rank: 0
main_process_ip: < TODO: specify IP >
main_process_port: < TODO: specify port >
main_training_function: main
mixed_precision: 'bf16'
num_machines: 2
num_processes: 2
parallelism_config:
  parallelism_config_cp_size: 1
  parallelism_config_dp_replicate_size: 1
  parallelism_config_dp_shard_size: 2
  parallelism_config_tp_size: 1
rdzv_backend: c10d
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
""",
}

EX_IPBR = {"a": "lo               UNKNOWN        127.0.0.1/8\nenP7s7           UP             10.0.0.21/24\n"
                 "enp1s0f1np1      UP             192.168.100.10/24\nenP2p1s0f1np1    UP             192.168.101.10/24",
           "b": "lo               UNKNOWN        127.0.0.1/8\nenP7s7           UP             10.0.0.22/24\n"
                 "enp1s0f1np1      UP             192.168.100.11/24\nenP2p1s0f1np1    UP             192.168.101.11/24"}


def link_iface(ibdev: str, ip_br: str) -> str:
    """The first ConnectX-7 netdev that is Up and has an IPv4 (port 0 or 1, whichever is cabled)."""
    for dev in re.findall(r"^\S+ port \d+ ==> (\S+) \(Up\)", ibdev, re.M):
        if iface_ip(ip_br, dev):
            return dev
    return COMPOSE_IF


def iface_ip(ip_br: str, dev: str) -> str:
    m = re.search(rf"^{re.escape(dev)}\s+\S+\s+(\d+\.\d+\.\d+\.\d+)/", ip_br, re.M)
    return m.group(1) if m else ""


def accelerate_for(rank: int, ip: str) -> str:
    text = PLAYBOOK_CFG[CFG_NAME]
    text = re.sub(r"^machine_rank: .*$", f"machine_rank: {rank}", text, flags=re.M)
    text = re.sub(r"^main_process_ip: .*$", f"main_process_ip: {ip}", text, flags=re.M)
    return re.sub(r"^main_process_port: .*$", f"main_process_port: {args.port}", text, flags=re.M)


banner("Lab 11-4 · two-Spark FSDP fine-tuning", f"preflight both Sparks · {CFG_NAME} for rank 0 and 1 · launch plan")

step(1, "read-only preflight on both Sparks")
facts = {}
for which in ("a", "b"):
    print(f"\n── Spark {which.upper()}")
    br = sh("ip -br -4 address", which, timeout=30, example=EX_IPBR[which]).out
    ibdev = sh("ibdev2netdev", which, timeout=30, example=f"rocep1s0f1 port 1 ==> {COMPOSE_IF} (Up)").out
    dev = link_iface(ibdev, br)
    swarm = sh("docker info --format '{{.Swarm.LocalNodeState}}'", which, timeout=30, example="inactive").out.strip()
    uuid = sh("nvidia-smi -a | grep UUID", which, timeout=30,
              example="    GPU UUID                              : GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx").out
    adv = sh("grep -c NVIDIA_GPU /etc/docker/daemon.json 2>/dev/null || echo 0", which, timeout=30, example="0").out
    res = sh("grep -cE '^\\s*swarm-resource' /etc/nvidia-container-runtime/config.toml 2>/dev/null || echo 0", which,
             timeout=30, example="0").out
    clone = sh(f"ls ~/{ASSETS}/{SCRIPT} ~/{ASSETS}/docker-compose.yml 2>/dev/null || echo missing", which, timeout=30,
               example=f"{SCRIPT}\ndocker-compose.yml").out     # not configs/: this lab's own put() creates that
    last = lambda s: (s.strip().splitlines() or ["0"])[-1].strip()  # noqa: E731
    facts[which] = {"dev": dev, "ip": iface_ip(br, dev), "swarm": swarm.splitlines()[-1] if swarm else "?",
                    "uuid": bool(re.search(r"GPU-[0-9a-fx-]+", uuid)), "adv": last(adv) not in ("0", ""),
                    "res": last(res) not in ("0", ""), "clone": "missing" not in clone}

rows = [[f"Spark {w.upper()}", f["dev"], f["ip"] or "—", f["swarm"], "✓" if f["uuid"] else "—",
         "✓" if f["adv"] else "✕ step 3", "✓" if f["res"] else "✕ step 3", "✓" if f["clone"] else "✕ lab 02"]
        for w, f in facts.items()]
table(rows, ["node", "interconnect", "IPv4", "swarm", "GPU UUID", "daemon.json NVIDIA_GPU", "swarm-resource", "recipes"])
for w, f in facts.items():
    if not f["ip"]:
        warn(f"Spark {w.upper()}: no ConnectX-7 interface is Up with an IPv4 address. Finish Module 02 first.")
    elif f["dev"] != COMPOSE_IF:
        warn(f"Spark {w.upper()}: the cable is on {f['dev']}, not the playbook's {COMPOSE_IF}. Edit UCX_NET_DEVICES, "
             f"NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME in docker-compose.yml to {f['dev']}.")

step(2, f"write {CFG_NAME} for each Spark (rank 0 = Spark A, the primary)")
primary = facts["a"]["ip"] or "<PRIMARY_INTERCONNECT_IP>"
OUT.mkdir(parents=True, exist_ok=True)
files = {}
for rank, which in enumerate(("a", "b")):
    text = accelerate_for(rank, primary)
    path = OUT / f"{CFG_NAME.replace('.yaml', '')}.spark-{which}.yaml"
    path.write_text(text, encoding="utf-8")
    files[which] = path
    changed = [ln for ln in text.splitlines() if re.match(r"(machine_rank|main_process_ip|main_process_port):", ln)]
    print(f"── {path.name}: " + " · ".join(changed))
try:
    import yaml
    docs = {w: yaml.safe_load(p.read_text()) for w, p in files.items()}
    check(bool(facts["a"]["ip"]) and all(d["num_machines"] == 2 and d["main_process_ip"] == docs["a"]["main_process_ip"] for d in docs.values())
          and [docs["a"]["machine_rank"], docs["b"]["machine_rank"]] == [0, 1],
          "both files parse, num_machines 2, ranks 0 and 1, the same main_process_ip on both",
          "main_process_ip is a placeholder (Spark A has no interconnect IP) or a file is inconsistent")
    orig = PLAYBOOK_CFG[CFG_NAME].splitlines()
    ndiff = {w: sum(x != y for x, y in zip(orig, p.read_text().splitlines())) for w, p in files.items()}
    check(all(n <= 3 for n in ndiff.values()),
          f"only machine_rank / main_process_ip / main_process_port changed ({ndiff['a']} lines on A, "
          f"{ndiff['b']} on B) — the rest is the playbook's {CFG_NAME}",
          "more than the three expected lines changed — report this as a bug")
except ImportError:
    warn("PyYAML not installed — skipped the parse check")
for which, path in files.items():
    if where(which) != "dry":
        dest = f"{ASSETS}/configs/{CFG_NAME}"
        put(path, f"~/{dest}" if where(which) == "local" else dest, which)

step(3, "the steps that need sudo — run them yourself on BOTH Sparks (playbook Steps 3–4)")
print("$ nvidia-smi -a | grep UUID                    # put YOUR UUID into daemon.json below")
print("$ sudoedit /etc/docker/daemon.json             # add \"default-runtime\": \"nvidia\" and")
print("                                               # \"node-generic-resources\": [\"NVIDIA_GPU=GPU-…\"]")
print("$ sudo sed -i 's/^#\\s*\\(swarm-resource\\s*=\\s*\".*\"\\)/\\1/' /etc/nvidia-container-runtime/config.toml")
print("$ sudo systemctl restart docker")

step(4, "form the swarm, deploy, launch (playbook Steps 4–9)")
print(f"$ docker swarm init --advertise-addr {primary}          # Spark A")
print("$ docker swarm join --token <worker-token> <advertise-addr>:<port>   # Spark B, the line swarm init printed")
print(f"$ cd ~/{ASSETS} && chmod +x pytorch-ft-entrypoint.sh")
print("$ docker stack deploy -c \"$PWD/docker-compose.yml\" finetuning-multinode   # Spark A")
print("$ docker stack ps finetuning-multinode                  # both tasks Running")
print("$ export FINETUNING_CONTAINER=$(docker ps -q -f name=finetuning-multinode)   # on EACH Spark")
print("$ docker exec -e HF_TOKEN -it \"$FINETUNING_CONTAINER\" bash -c '")
print("    bash /workspace/install-requirements;")
print(f"    accelerate launch --config_file=/workspace/configs/{CFG_NAME} /workspace/{SCRIPT}{SCRIPT_ARGS}'   # on EACH Spark")
note("The playbook's run-multi-llama_* helpers wrap that last command. Progress prints on Spark A only; "
     "check Spark B with nvidia-smi. HF_TOKEN must be exported in each Spark's shell first (never in a file you commit).")
note("The stack's containers run sshd on port 2233 with root password 'root' on the host network "
     "(pytorch-ft-entrypoint.sh). Remove it when you are done: docker stack rm finetuning-multinode")

ready = all(f["ip"] and f["adv"] and f["res"] and f["clone"] for f in facts.values())
live = all(where(w) != "dry" for w in ("a", "b"))
if live and ready:
    result("both Sparks pass the preflight and have their Accelerate file. Run the step-4 commands.")
elif live:
    result("fix the ✕ columns above (steps 3–4 or lab 02), then run this lab again.")
else:
    result("DRY: the preflight rows are EXAMPLE shapes; the two Accelerate files are real, generated from the "
           "playbook's config on this machine.")

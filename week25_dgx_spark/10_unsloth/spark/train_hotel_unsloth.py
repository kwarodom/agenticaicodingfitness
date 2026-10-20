"""Week 25 · Module 10 — train the Module 09 hotel dataset with Unsloth, inside the playbook container.

Adapted from the playbook's assets/test_unsloth.py (same imports, same FastModel / get_peft_model /
SFTTrainer / SFTConfig calls). What changed, and why:
  • every hyperparameter comes from hotel_unsloth_config.json, which lab 02 derives from Module 09's YAML,
    so the two frameworks train with the same numbers;
  • the dataset is Module 09's hotel_ops.json, converted by lab 02 into TRL's conversational
    prompt–completion format, so the loss is computed on the answer only (as in LLaMA Factory);
  • at the end it saves the adapter, prints one W25_RESULT JSON line (runtime, loss, peak memory),
    and writes generated_predictions.jsonl for the 60 held-out requests, in LLaMA Factory's format.

Run it through run_unsloth.sh (lab 02 does that for you). Not runnable on a laptop.
"""
import json
import time

from unsloth import FastLanguageModel, FastModel
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

cfg = json.load(open("hotel_unsloth_config.json"))
print("== W25 config:", json.dumps(cfg))

train_ds = load_dataset("json", data_files={"train": "data/hotel_ops_chat.jsonl"}, split="train")
eval_ds = load_dataset("json", data_files={"eval": "data/hotel_ops_eval_chat.jsonl"}, split="eval")

model, tokenizer = FastModel.from_pretrained(
    model_name=cfg["model"],
    max_seq_length=cfg["max_seq_length"],
    load_in_4bit=cfg["load_in_4bit"],    # False = LoRA on a bf16 base (fair vs Module 09); True = QLoRA
    load_in_8bit=False,
    full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=cfg["r"],
    target_modules=cfg["target_modules"],
    lora_alpha=cfg["lora_alpha"],
    lora_dropout=cfg["lora_dropout"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=cfg["seed"],
    max_seq_length=cfg["max_seq_length"],
    use_rslora=False,
    loftq_config=None,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    args=SFTConfig(
        max_seq_length=cfg["max_seq_length"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        optim=cfg["optim"],
        bf16=True,
        seed=cfg["seed"],
        output_dir=cfg["output_dir"],
        report_to="none",
        completion_only_loss=True,       # loss on the answer only, like LLaMA Factory's default
    ),
)

torch.cuda.reset_peak_memory_stats()
stats = trainer.train()
result = {
    "framework": "unsloth",
    "train_runtime_s": round(stats.metrics.get("train_runtime", 0.0), 1),
    "train_samples_per_second": stats.metrics.get("train_samples_per_second"),
    "train_loss": round(stats.training_loss, 4),
    "eval_loss": round(trainer.evaluate().get("eval_loss", float("nan")), 4),
    "peak_mem_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    "peak_mem_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
    "global_step": trainer.state.global_step,
}
print("W25_RESULT " + json.dumps(result))

model.save_pretrained(cfg["adapter_dir"])
tokenizer.save_pretrained(cfg["adapter_dir"])
print("== W25 adapter saved to", cfg["adapter_dir"])

# Answers for the held-out requests, greedy, in LLaMA Factory's generated_predictions.jsonl format.
FastLanguageModel.for_inference(model)
t0 = time.time()
with open(cfg["predictions"], "w", encoding="utf-8") as f:
    for rec in eval_ds:
        text = tokenizer.apply_chat_template(rec["prompt"], tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt").to("cuda")
        out = model.generate(**ids, max_new_tokens=160, do_sample=False)
        pred = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        f.write(json.dumps({"prompt": rec["prompt"][-1]["content"], "predict": pred,
                            "label": rec["completion"][0]["content"]}, ensure_ascii=False) + "\n")
print(f"== W25 wrote {len(eval_ds)} predictions to {cfg['predictions']} in {time.time() - t0:.0f}s")

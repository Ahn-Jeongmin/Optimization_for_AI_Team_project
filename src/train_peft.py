import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    AdaLoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    elif torch.cuda.is_available():
        return torch.float16
    else:
        return torch.float32


# -------------------------
# 데이터 전처리 (GSM8K)
# -------------------------


def get_gsm8k_datasets(tokenizer, max_length: int = 512):
    """
    GSM8K question / answer를
    "Question: ...\\nAnswer: ..." 포맷으로 바꿔서
    Causal LM 학습용 텍스트를 만듬
    """
    raw_ds = load_dataset("openai/gsm8k", "main")

    def preprocess(example):
        q = example["question"].strip()
        a = example["answer"].strip()

        prompt = f"Question: {q}\nAnswer:"
        full_text = prompt + " " + a

        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
        )
        # Causal LM: 전체 토큰에 대해 loss
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    train_ds = raw_ds["train"].map(preprocess, remove_columns=raw_ds["train"].column_names)
    test_ds = raw_ds["test"].map(preprocess, remove_columns=raw_ds["test"].column_names)
    
    # train_ds = train_ds.select(range(800))
    # test_ds = test_ds.select(range(200))
    
    return train_ds, test_ds


@dataclass
class CausalLMCollator:
    tokenizer: AutoTokenizer

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        # input_ids / attention_mask만 모아서 패딩
        input_features = [
            {k: v for k, v in f.items() if k in ["input_ids", "attention_mask"]}
            for f in features
        ]
        labels = [f["labels"] for f in features]

        padded = self.tokenizer.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )

        # labels 패딩: pad 위치는 -100으로 채워서 loss에서 무시
        max_len = padded["input_ids"].size(1)
        labels_tensor = torch.full(
            (len(labels), max_len),
            -100,
            dtype=torch.long,
        )
        for i, lab in enumerate(labels):
            lab = torch.tensor(lab, dtype=torch.long)
            labels_tensor[i, : lab.size(0)] = lab

        padded["labels"] = labels_tensor
        return padded


# -------------------------
# LoRA 모델 구성
# -------------------------


def get_default_target_modules(base_model_name: str):
    # GPT-2는 c_attn / c_proj
    if "gpt2" in base_model_name.lower():
        return ["c_attn", "c_proj"]
    # LLaMA/Mistral 계열: q/k/v/o proj
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def build_lora_model(
    base_model_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules=None,
) -> PeftModel:
    if target_modules is None:
        target_modules = get_default_target_modules(base_model_name)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=get_dtype(),
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    return model


# -------------------------
# AdaLoRA
# -------------------------


def build_adalora_model(
    base_model_name: str,
    init_r: int,
    target_r: int,
    total_step: int,
    lora_alpha: int,
    lora_dropout: float,
    tinit: int,
    tfinal: int,
    deltaT: int,
    orth_reg_weight: float = 0.5,
    target_modules=None,
) -> PeftModel:
    if target_modules is None:
        target_modules = get_default_target_modules(base_model_name)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=get_dtype(),
    )

    adalora_config = AdaLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=init_r,
        init_r=init_r,
        target_r=target_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        total_step=total_step,
        tinit=tinit,
        tfinal=tfinal,
        deltaT=deltaT,
        orth_reg_weight=orth_reg_weight,
    )

    model = get_peft_model(base_model, adalora_config)
    return model


def evaluate(model: PeftModel, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


def train_adalora(
    model: PeftModel,
    train_dataset,
    eval_dataset,
    tokenizer,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    lr: float = 5e-5,
    max_grad_norm: float = 1.0,
    eval_steps: int = 1000,
):
    device = get_device()
    model.to(device)

    collator = CausalLMCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    global_step = 0
    model.train()

    for epoch in range(num_epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # ------- AdaLoRA 핵심: step마다 budget 업데이트 -------
            if hasattr(model, "base_model") and hasattr(
                model.base_model, "update_and_allocate"
            ):
                model.base_model.update_and_allocate(global_step)
            elif hasattr(model, "update_and_allocate"):
                model.update_and_allocate(global_step)

            optimizer.zero_grad()
            global_step += 1

            if global_step % 50 == 0:
                print(f"[Epoch {epoch+1}] step {global_step} | loss = {loss.item():.4f}")

            if eval_steps > 0 and global_step % eval_steps == 0:
                eval_loss = evaluate(model, eval_loader, device)
                print(f"[Eval @ step {global_step}] loss = {eval_loss:.4f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[*] AdaLoRA adapter saved to: {output_dir}")


# -------------------------
# 메인 루프
# -------------------------


def parse_args():
    parser = argparse.ArgumentParser()

    # 모델 / 태스크 기본 설정
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help=(
            "Base Causal LM "
            "(e.g., gpt2, mistralai/Mistral-7B-v0.1, meta-llama/Llama-3.2-1B-Instruct)"
        ),
    )
    parser.add_argument(
        "--adapter_type",
        type=str,
        choices=["lora", "adalora"],
        default="lora",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs/gsm8k_peft")
    parser.add_argument("--max_length", type=int, default=512)

    # 공통 학습 하이퍼파라미터
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)

    # LoRA 설정
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # AdaLoRA 설정
    parser.add_argument("--adalora_init_r", type=int, default=12)
    parser.add_argument("--adalora_target_r", type=int, default=8)
    parser.add_argument(
        "--adalora_tinit_ratio",
        type=float,
        default=0.1,
        help="tinit = ratio * total_steps",
    )
    parser.add_argument(
        "--adalora_tfinal_ratio",
        type=float,
        default=0.2,
        help="tfinal = ratio * total_steps",
    )
    parser.add_argument("--adalora_deltaT", type=int, default=10)
    parser.add_argument("--adalora_orth_reg", type=float, default=0.5)
    parser.add_argument("--eval_steps", type=int, default=1000)

    return parser.parse_args()


def main():
    args = parse_args()

    print("[*] Loading tokenizer and GSM8K datasets...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # 일부 LLM들은 pad_token이 없어서 eos_token을 대신 사용
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, test_ds = get_gsm8k_datasets(tokenizer, max_length=args.max_length)

    if args.adapter_type == "lora":
        # ----------------- LoRA + Trainer -----------------
        print("[*] Building LoRA model...")
        model = build_lora_model(
            base_model_name=args.model_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

        collator = CausalLMCollator(tokenizer)

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            bf16=torch.cuda.is_available()
            and torch.cuda.is_bf16_supported(),
            logging_steps=max(1, args.eval_steps // 10),
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.eval_steps,
            save_total_limit=2,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            data_collator=collator,
            tokenizer=tokenizer,
        )

        print("[*] Start LoRA training...")
        trainer.train()

        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"[*] LoRA adapter saved to: {args.output_dir}")

    else:
        # ----------------- AdaLoRA -----------------
        print("[*] Building AdaLoRA model...")
        total_steps = math.ceil(len(train_ds) / args.batch_size) * args.num_epochs
        tinit = int(args.adalora_tinit_ratio * total_steps)
        tfinal = int(args.adalora_tfinal_ratio * total_steps)

        print(f"    total_steps={total_steps}, tinit={tinit}, tfinal={tfinal}")

        model = build_adalora_model(
            base_model_name=args.model_name,
            init_r=args.adalora_init_r,
            target_r=args.adalora_target_r,
            total_step=total_steps,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            tinit=tinit,
            tfinal=tfinal,
            deltaT=args.adalora_deltaT,
            orth_reg_weight=args.adalora_orth_reg,
        )

        print("[*] Start AdaLoRA training (custom loop)...")
        train_adalora(
            model=model,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            eval_steps=args.eval_steps,
        )


if __name__ == "__main__":
    main()

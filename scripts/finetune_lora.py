"""
LoRA fine-tuning script for the REFUTE R-Agent (DeepSeek-Math-7B).

Fine-tunes any HuggingFace causal-LM using PEFT LoRA + ``trl.SFTTrainer`` on
(conjecture, strategy) → (reasoning, counterexample) triples produced by
``gen_training_data.py``.  Supports 4-bit bitsandbytes quantisation for
A100/H100 nodes.

Typical usage
-------------
::

    python scripts/finetune_lora.py \\
        --config configs/finetune_config.yaml \\
        --data data/training/samples.jsonl \\
        --output models/refuter_lora_v1

Dependencies
------------
::

    pip install transformers peft trl bitsandbytes datasets accelerate
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional-dependency guards — fail fast with clear install instructions
# ---------------------------------------------------------------------------

try:
    import torch
except ImportError as _exc:
    raise ImportError(
        "PyTorch is required. Install with: pip install torch"
    ) from _exc

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        PreTrainedModel,
        PreTrainedTokenizerBase,
        TrainingArguments,
    )
except ImportError as _exc:
    raise ImportError(
        "The 'transformers' package is required: pip install transformers>=4.45.0"
    ) from _exc

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError as _exc:
    raise ImportError(
        "The 'peft' package is required for LoRA: pip install peft>=0.13.0"
    ) from _exc

try:
    from trl import SFTTrainer
except ImportError as _exc:
    raise ImportError(
        "The 'trl' package is required for SFTTrainer: pip install trl>=0.11.0"
    ) from _exc

try:
    from datasets import Dataset, DatasetDict
except ImportError as _exc:
    raise ImportError(
        "The 'datasets' package is required: pip install datasets>=3.0.0"
    ) from _exc


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _detect_device() -> str:
    """
    Return the most capable available device string for logging/reporting.

    Returns:
        ``"cuda"`` if any CUDA GPU is available, ``"mps"`` for Apple Silicon,
        otherwise ``"cpu"``.
    """
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _set_seed(seed: int) -> None:
    """
    Set all relevant random seeds for reproducibility.

    Args:
        seed: Integer seed applied to Python ``random``, ``numpy``, and
            ``torch`` (including CUDA if available).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to %d", seed)


def _load_yaml_config(config_path: str) -> dict:
    """
    Load and parse a YAML configuration file.

    Args:
        config_path: Absolute or relative path to the YAML file.

    Returns:
        Parsed YAML as a nested Python dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Fine-tune config not found: {path.resolve()}"
        )
    with path.open("r", encoding="utf-8") as fh:
        cfg: dict = yaml.safe_load(fh) or {}
    return cfg


# ---------------------------------------------------------------------------
# Core trainer class
# ---------------------------------------------------------------------------


class RefuteLoRATrainer:
    """
    Manages LoRA fine-tuning of the REFUTE R-Agent on counterexample triples.

    Loads configuration from a YAML file (``configs/finetune_config.yaml``),
    prepares the dataset, applies 4-bit quantisation + LoRA adapters via PEFT,
    and delegates training to ``trl.SFTTrainer``.

    Args:
        config_path: Path to the fine-tuning YAML config file.
    """

    def __init__(self, config_path: str) -> None:
        self._cfg: dict = _load_yaml_config(config_path)
        self._validate_config(self._cfg)

        seed: int = self._cfg.get("data", {}).get("seed", 42)
        _set_seed(seed)

        logger.info("RefuteLoRATrainer initialized | device=%s", _detect_device())
        logger.info(
            "Base model: %s", self._cfg["model"]["base_model"]
        )

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_config(cfg: dict) -> None:
        """
        Fail-fast validation of required configuration sections and fields.

        Args:
            cfg: The raw YAML config dictionary.

        Raises:
            ValueError: If any required key is absent or has an invalid value.
        """
        required_top_level = {"model", "lora", "training", "data", "output"}
        missing = required_top_level - set(cfg.keys())
        if missing:
            raise ValueError(
                f"Fine-tune config is missing required top-level keys: {sorted(missing)}"
            )

        model_cfg = cfg["model"]
        if not model_cfg.get("base_model"):
            raise ValueError("config.model.base_model must be a non-empty string")

        lora_cfg = cfg["lora"]
        if not isinstance(lora_cfg.get("r"), int) or lora_cfg["r"] <= 0:
            raise ValueError("config.lora.r must be a positive integer")
        if not lora_cfg.get("target_modules"):
            raise ValueError("config.lora.target_modules must be a non-empty list")

        training_cfg = cfg["training"]
        for field in ("per_device_train_batch_size", "num_train_epochs", "learning_rate"):
            if field not in training_cfg:
                raise ValueError(f"config.training.{field} is required")

        output_cfg = cfg["output"]
        if not output_cfg.get("output_dir"):
            raise ValueError("config.output.output_dir must be a non-empty string")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, data_path: str) -> tuple[Dataset, Dataset]:
        """
        Load training samples from JSONL and split into train/eval sets.

        Each JSONL record must contain at minimum a ``"text"`` key (the full
        prompt+completion string used by ``SFTTrainer``).  Records missing this
        key are skipped with a warning.

        Args:
            data_path: Path to the JSONL file produced by ``gen_training_data.py``.

        Returns:
            A ``(train_dataset, eval_dataset)`` tuple of HuggingFace
            ``Dataset`` objects.

        Raises:
            FileNotFoundError: If ``data_path`` does not exist.
            ValueError: If the file contains no valid records.
        """
        jsonl_path = Path(data_path)
        if not jsonl_path.is_file():
            raise FileNotFoundError(
                f"Training data file not found: {jsonl_path.resolve()}"
            )

        data_cfg = self._cfg.get("data", {})
        max_samples: Optional[int] = data_cfg.get("max_samples")
        seed: int = data_cfg.get("seed", 42)
        train_split: float = data_cfg.get("train_split", 0.8)

        records: list[dict] = []
        skipped = 0

        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(
                tqdm(fh, desc="Loading JSONL", unit="line", dynamic_ncols=True, leave=False),
                start=1,
            ):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    record: dict = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping invalid JSON on line %d: %s", line_no, exc)
                    skipped += 1
                    continue

                if "text" not in record:
                    logger.warning(
                        "Skipping line %d — missing 'text' key (keys present: %s)",
                        line_no,
                        list(record.keys()),
                    )
                    skipped += 1
                    continue

                records.append({"text": record["text"]})

                if max_samples is not None and len(records) >= max_samples:
                    logger.info(
                        "Reached max_samples=%d. Stopping data load.", max_samples
                    )
                    break

        if not records:
            raise ValueError(
                f"No valid training records found in {jsonl_path}. "
                "Verify that gen_training_data.py ran successfully."
            )

        logger.info(
            "Loaded %d records (%d skipped) from %s", len(records), skipped, jsonl_path
        )

        rng = random.Random(seed)
        rng.shuffle(records)

        split_idx = max(1, int(len(records) * train_split))
        train_records = records[:split_idx]
        eval_records = records[split_idx:] or records[:1]  # guarantee non-empty eval

        train_dataset = Dataset.from_list(train_records)
        eval_dataset = Dataset.from_list(eval_records)

        logger.info(
            "Dataset split: train=%d eval=%d", len(train_dataset), len(eval_dataset)
        )
        return train_dataset, eval_dataset

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def setup_model(
        self,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """
        Load the base model with optional 4-bit quantisation and apply LoRA.

        Steps:
        1. Build a ``BitsAndBytesConfig`` if ``model.load_in_4bit`` is true.
        2. Load the base causal-LM with ``device_map="auto"`` for multi-GPU
           distribution.
        3. Construct a ``LoraConfig`` from the YAML ``lora`` section and wrap
           the model via ``peft.get_peft_model``.

        Returns:
            A ``(model, tokenizer)`` tuple where ``model`` has LoRA adapters
            injected and ``tokenizer`` has its padding side set to ``"right"``
            for SFT (avoids loss on padding tokens at sequence start).

        Raises:
            ImportError: If ``bitsandbytes`` is unavailable when 4-bit is
                requested.
        """
        model_cfg = self._cfg["model"]
        lora_cfg = self._cfg["lora"]

        base_model_id: str = model_cfg["base_model"]
        torch_dtype_str: str = model_cfg.get("torch_dtype", "bfloat16")
        load_in_4bit: bool = model_cfg.get("load_in_4bit", False)
        trust_remote_code: bool = model_cfg.get("trust_remote_code", True)
        device_map: str = model_cfg.get("device_map", "auto")

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if torch_dtype_str not in dtype_map:
            raise ValueError(
                f"model.torch_dtype must be one of {list(dtype_map)}, "
                f"got {torch_dtype_str!r}"
            )
        torch_dtype = dtype_map[torch_dtype_str]

        logger.info("Loading tokenizer | model=%s", base_model_id)
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.debug("Set pad_token to eos_token (%s)", tokenizer.eos_token)
        tokenizer.padding_side = "right"

        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }

        if load_in_4bit:
            try:
                import bitsandbytes  # noqa: F401  — import check only
            except ImportError as exc:
                raise ImportError(
                    "bitsandbytes is required for 4-bit quantisation. "
                    "Install with: pip install bitsandbytes>=0.43.0"
                ) from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("4-bit NF4 quantisation enabled")

        logger.info("Loading base model | model=%s", base_model_id)
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            base_model_id, **model_kwargs
        )

        lora_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg.get("lora_alpha", lora_cfg["r"] * 2),
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        logger.info(
            "LoRA applied | r=%d | alpha=%d | target_modules=%s",
            lora_cfg["r"],
            lora_cfg.get("lora_alpha", lora_cfg["r"] * 2),
            lora_cfg["target_modules"],
        )

        return model, tokenizer

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        data_path: str,
        output_dir: Optional[str] = None,
    ) -> None:
        """
        Run the full LoRA fine-tuning loop.

        Orchestrates data loading, model setup, ``SFTTrainer`` construction,
        and checkpoint saving.  After training completes the LoRA adapter
        weights are saved to ``output_dir`` in PEFT format, and the tokenizer
        is saved alongside.

        Args:
            data_path: Path to the JSONL training data file.
            output_dir: Override for ``config.output.output_dir``.  If ``None``,
                the value from the YAML config is used.
        """
        training_cfg = self._cfg["training"]
        output_cfg = self._cfg["output"]
        data_cfg = self._cfg.get("data", {})

        effective_output_dir = output_dir or output_cfg["output_dir"]
        Path(effective_output_dir).mkdir(parents=True, exist_ok=True)

        train_dataset, eval_dataset = self.load_data(data_path)
        model, tokenizer = self.setup_model()

        report_to: str = training_cfg.get("report_to", "none")
        max_seq_length: int = training_cfg.get("max_seq_length", 2048)

        training_args = TrainingArguments(
            output_dir=effective_output_dir,
            per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=training_cfg.get(
                "per_device_eval_batch_size",
                training_cfg["per_device_train_batch_size"],
            ),
            gradient_accumulation_steps=training_cfg.get(
                "gradient_accumulation_steps", 4
            ),
            num_train_epochs=training_cfg["num_train_epochs"],
            learning_rate=float(training_cfg["learning_rate"]),
            warmup_ratio=training_cfg.get("warmup_ratio", 0.05),
            lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
            bf16=training_cfg.get("bf16", True) and torch.cuda.is_available(),
            fp16=False,
            gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
            logging_steps=training_cfg.get("logging_steps", 10),
            save_strategy=training_cfg.get("save_strategy", "epoch"),
            evaluation_strategy=training_cfg.get("evaluation_strategy", "epoch"),
            load_best_model_at_end=training_cfg.get("load_best_model_at_end", True),
            metric_for_best_model=training_cfg.get(
                "metric_for_best_model", "eval_loss"
            ),
            save_total_limit=output_cfg.get("save_total_limit", 3),
            report_to=report_to,
            seed=data_cfg.get("seed", 42),
            remove_unused_columns=False,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            max_seq_length=max_seq_length,
            dataset_text_field="text",
            packing=False,
        )

        logger.info(
            "Starting SFT training | epochs=%d | eff_batch=%d | lr=%.2e",
            training_cfg["num_train_epochs"],
            training_cfg["per_device_train_batch_size"]
            * training_cfg.get("gradient_accumulation_steps", 4),
            float(training_cfg["learning_rate"]),
        )

        trainer.train()

        logger.info("Training complete. Saving LoRA adapter to %s", effective_output_dir)
        trainer.save_model(effective_output_dir)
        tokenizer.save_pretrained(effective_output_dir)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            "Checkpoint saved | output_dir=%s", effective_output_dir
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        eval_dataset: Dataset,
    ) -> dict[str, float]:
        """
        Compute held-out evaluation metrics on the R-Agent fine-tuned model.

        Runs inference over ``eval_dataset`` with greedy decoding and computes:

        * ``counterexample_success_rate``: fraction of samples where the model
          output contains a ``<COUNTEREXAMPLE>`` block.
        * ``reasoning_present_rate``: fraction of samples where a ``<REASONING>``
          block is present.
        * ``mean_reasoning_length``: mean number of characters in the
          ``<REASONING>`` block across all samples.
        * ``mean_output_length``: mean total character length of model outputs.

        Args:
            model: The fine-tuned (PEFT-wrapped) causal-LM.
            tokenizer: The associated tokenizer.
            eval_dataset: HuggingFace ``Dataset`` with a ``"text"`` column.

        Returns:
            A dictionary mapping metric names to float values.
        """
        import re

        training_cfg = self._cfg["training"]
        max_seq_length: int = training_cfg.get("max_seq_length", 2048)
        max_new_tokens: int = 512

        model.eval()
        device = next(model.parameters()).device

        ce_present: list[bool] = []
        reasoning_present: list[bool] = []
        reasoning_lengths: list[int] = []
        output_lengths: list[int] = []

        for record in tqdm(
            eval_dataset,
            desc="Evaluating",
            unit="sample",
            dynamic_ncols=True,
            leave=True,
        ):
            # Extract the prompt (everything up to ### Response:)
            full_text: str = record["text"]
            response_marker = "### Response:\n"
            split_idx = full_text.find(response_marker)
            if split_idx == -1:
                prompt_text = full_text
            else:
                prompt_text = full_text[: split_idx + len(response_marker)]

            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length - max_new_tokens,
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            new_ids = output_ids[0][input_len:]
            decoded = tokenizer.decode(new_ids, skip_special_tokens=True)

            has_ce = bool(re.search(r"<COUNTEREXAMPLE>", decoded, re.IGNORECASE))
            has_reasoning = bool(re.search(r"<REASONING>", decoded, re.IGNORECASE))

            ce_present.append(has_ce)
            reasoning_present.append(has_reasoning)
            output_lengths.append(len(decoded))

            reasoning_match = re.search(
                r"<REASONING>\s*(.*?)\s*</REASONING>",
                decoded,
                re.DOTALL | re.IGNORECASE,
            )
            reasoning_lengths.append(
                len(reasoning_match.group(1)) if reasoning_match else 0
            )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        n = max(len(ce_present), 1)
        metrics: dict[str, float] = {
            "counterexample_success_rate": sum(ce_present) / n,
            "reasoning_present_rate": sum(reasoning_present) / n,
            "mean_reasoning_length": float(np.mean(reasoning_lengths)) if reasoning_lengths else 0.0,
            "mean_output_length": float(np.mean(output_lengths)) if output_lengths else 0.0,
            "n_eval_samples": float(n),
        }

        logger.info("Evaluation metrics: %s", metrics)
        return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for ``finetune_lora.py``.

    Returns:
        A configured ``argparse.ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        prog="finetune_lora",
        description=(
            "LoRA fine-tune DeepSeek-Math-7B (or any HF causal-LM) as the "
            "REFUTE R-Agent using counterexample training triples."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to configs/finetune_config.yaml",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to JSONL training samples from gen_training_data.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="DIR",
        help="Override output directory (default: config.output.output_dir)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "Skip training; load the adapter from --output and run evaluation "
            "on the held-out split."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser


def main() -> None:
    """
    CLI entry point for the LoRA fine-tuning pipeline.

    Parses arguments, configures logging, and dispatches to either the full
    training run or evaluation-only mode.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    trainer = RefuteLoRATrainer(config_path=args.config)

    if args.eval_only:
        logger.info("Eval-only mode: loading adapter from %s", args.output)
        if not args.output:
            parser.error("--output must be specified when --eval-only is set")

        _, eval_dataset = trainer.load_data(args.data)
        model, tokenizer = trainer.setup_model()

        # Load the saved LoRA adapter on top of the freshly-loaded base model.
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("peft is required for eval-only mode: pip install peft") from exc

        model = PeftModel.from_pretrained(model, args.output)
        metrics = trainer.evaluate(model, tokenizer, eval_dataset)
        print(json.dumps(metrics, indent=2))
    else:
        trainer.train(data_path=args.data, output_dir=args.output)


if __name__ == "__main__":
    main()

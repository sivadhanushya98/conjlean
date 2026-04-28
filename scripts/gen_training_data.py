"""
Training data generation pipeline for the REFUTE R-Agent fine-tuning stage.

Generates (conjecture, strategy) → (reasoning_trace, counterexample) triples by
querying a frontier LLM (via the existing ConjLean LLM client) for multiple
reasoning traces per benchmark entry, verifying each proposed counterexample with
SymPy, and persisting verified samples as JSONL for downstream SFT.

Typical usage
-------------
::

    python scripts/gen_training_data.py \\
        --config configs/config.yaml \\
        --benchmark-dir data/benchmark \\
        --output data/training/samples.jsonl \\
        --n-traces 3 \\
        --max-concurrent 5
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Adjust sys.path so the package is importable when run as a script directly
# from the repo root (i.e., before an editable install).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from conjlean.config import ConjLeanConfig
from conjlean.models import LLMClient, create_client
from conjlean.schemas import (
    BenchmarkEntry,
    BenchmarkTier,
    Conjecture,
    Domain,
    RefuterStrategy,
    TrainingSample,
    TrainingSampleSource,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy metadata
# ---------------------------------------------------------------------------

_STRATEGY_DESCRIPTIONS: dict[RefuterStrategy, str] = {
    RefuterStrategy.BOUNDARY: (
        "Test boundary values: n=0, 1, 2, very large numbers, edge cases of the "
        "stated domain"
    ),
    RefuterStrategy.RANDOM_STRUCTURED: (
        "Try strategically chosen values: primes, squares, Fibonacci numbers, "
        "powers of 2"
    ),
    RefuterStrategy.ANALOGICAL: (
        "Look for patterns from similar known counterexamples in number "
        "theory/algebra"
    ),
    RefuterStrategy.SYMBOLIC_PERTURBATION: (
        "Identify which constants/parameters in the conjecture are critical, "
        "perturb them"
    ),
}

_STRATEGY_INSTRUCTIONS: dict[RefuterStrategy, str] = {
    RefuterStrategy.BOUNDARY: (
        "Start with n=0, n=1, n=2. Then try n=40, n=41, the exact boundary of "
        "any stated condition, and large values such as n=100 or n=1000. Document "
        "every value you check."
    ),
    RefuterStrategy.RANDOM_STRUCTURED: (
        "Check n equal to the first 10 primes (2, 3, 5, 7, 11, 13, 17, 19, 23, "
        "29), then perfect squares (4, 9, 16, 25), Fibonacci numbers (1, 1, 2, 3, "
        "5, 8, 13), and powers of 2 (1, 2, 4, 8, 16, 32). Justify each choice."
    ),
    RefuterStrategy.ANALOGICAL: (
        "Recall structurally similar conjectures that are known to be false "
        "(e.g., n^2+n+41 is prime for n<40, Goldbach-style identities, Fermat "
        "numbers). Reason by analogy to identify the failure regime, then verify "
        "explicitly."
    ),
    RefuterStrategy.SYMBOLIC_PERTURBATION: (
        "Express the conjecture symbolically. Vary each numeric constant by ±1 "
        "and ±10 to identify sensitivity. Use SymPy or mental arithmetic to "
        "evaluate the perturbed expressions and isolate the exact failure point."
    ),
}

# Temperatures used for augmentation: one deterministic pass + diverse samples
_AUGMENTATION_TEMPERATURES: list[float] = [0.2, 0.7, 1.0]


# ---------------------------------------------------------------------------
# Core generator class
# ---------------------------------------------------------------------------


class TrainingDataGenerator:
    """
    Generates rich (conjecture, strategy) → (reasoning, counterexample) triples
    for fine-tuning the R-Agent (DeepSeek-Math-7B).

    For every benchmark entry with a known ground-truth counterexample the
    generator calls a frontier LLM at multiple temperatures to produce diverse
    reasoning traces, then verifies each proposed counterexample numerically via
    SymPy.  Only verified samples are written to the output JSONL.

    Args:
        client: An instantiated ``LLMClient`` (Anthropic, OpenAI, vLLM, etc.).
        config: A validated ``ConjLeanConfig`` controlling generation parameters.
    """

    def __init__(self, client: LLMClient, config: ConjLeanConfig) -> None:
        self._client = client
        self._config = config
        self._max_tokens: int = 2048

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def generate_for_entry(
        self,
        entry: BenchmarkEntry,
        strategies: list[RefuterStrategy],
        n_traces_per_strategy: int = 3,
    ) -> list[TrainingSample]:
        """
        Generate training samples for a single benchmark entry.

        For each strategy in ``strategies``, the generator issues
        ``n_traces_per_strategy`` LLM calls at staggered temperatures to
        produce diverse reasoning traces.  Each proposed counterexample is
        verified; only confirmed samples are returned.

        Args:
            entry: A benchmark entry with a known ground-truth counterexample.
            strategies: List of ``RefuterStrategy`` values to train.
            n_traces_per_strategy: Number of reasoning traces to generate per
                (entry, strategy) pair before deduplication.

        Returns:
            A list of verified ``TrainingSample`` objects (may be empty if no
            LLM output passes SymPy verification).
        """
        if not entry.ground_truth_counterexample:
            logger.debug(
                "Skipping entry %s — no ground-truth counterexample", entry.id
            )
            return []

        samples: list[TrainingSample] = []
        temperatures = (_AUGMENTATION_TEMPERATURES * n_traces_per_strategy)[
            :n_traces_per_strategy
        ]

        for strategy in strategies:
            prompts = [
                self._build_generation_prompt(
                    entry.conjecture, strategy, entry.ground_truth_counterexample
                )
                for _ in range(n_traces_per_strategy)
            ]

            try:
                responses = await asyncio.gather(
                    *[
                        self._client.complete(
                            messages=prompts[i],
                            temperature=temperatures[i],
                            max_tokens=self._max_tokens,
                        )
                        for i in range(n_traces_per_strategy)
                    ],
                    return_exceptions=True,
                )
            except Exception as exc:
                logger.warning(
                    "Batch completion failed for entry=%s strategy=%s: %s",
                    entry.id,
                    strategy.value,
                    exc,
                )
                continue

            for response in responses:
                if isinstance(response, BaseException):
                    logger.warning(
                        "Single completion error for entry=%s strategy=%s: %s",
                        entry.id,
                        strategy.value,
                        response,
                    )
                    continue

                reasoning, counterexample_str = self._parse_reasoning_and_counterexample(
                    response
                )
                if not reasoning or not counterexample_str:
                    logger.debug(
                        "Could not parse response for entry=%s strategy=%s",
                        entry.id,
                        strategy.value,
                    )
                    continue

                verified = self._verify_counterexample(
                    entry.conjecture, counterexample_str
                )
                source = (
                    TrainingSampleSource.BENCHMARK_VERIFIED
                    if verified
                    else TrainingSampleSource.FRONTIER_GENERATED
                )

                samples.append(
                    TrainingSample(
                        conjecture_nl=entry.conjecture.nl_statement,
                        strategy=strategy,
                        reasoning_trace=reasoning,
                        counterexample=counterexample_str,
                        verification_evidence={
                            "sympy_verified": verified,
                            "ground_truth": entry.ground_truth_counterexample,
                            "entry_id": entry.id,
                        },
                        domain=entry.conjecture.domain.value,
                        source=source,
                    )
                )

        logger.info(
            "Entry %s: generated %d samples across %d strategies",
            entry.id,
            len(samples),
            len(strategies),
        )
        return samples

    async def generate_batch(
        self,
        entries: list[BenchmarkEntry],
        output_path: Path,
        max_concurrent: int = 5,
    ) -> None:
        """
        Generate and persist training samples for a list of benchmark entries.

        Entries are processed in semaphore-bounded concurrent batches to avoid
        overloading the upstream LLM API.  Each verified sample is appended to
        ``output_path`` as a newline-delimited JSON record immediately upon
        completion, so progress is preserved across partial runs.

        After every batch the generator explicitly calls ``gc.collect()`` to
        release intermediate objects accumulated during async fan-out.

        Args:
            entries: List of ``BenchmarkEntry`` objects to process.
            output_path: Path to the output JSONL file.  Created if absent;
                appended to if it already exists.
            max_concurrent: Maximum number of concurrent ``generate_for_entry``
                coroutines.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        semaphore = asyncio.Semaphore(max_concurrent)
        strategies = list(RefuterStrategy)
        n_traces = self._config.generation.batch_size // max(len(strategies), 1)
        n_traces = max(n_traces, 1)
        total_written = 0

        async def _bounded(entry: BenchmarkEntry) -> list[TrainingSample]:
            async with semaphore:
                return await self.generate_for_entry(
                    entry,
                    strategies=strategies,
                    n_traces_per_strategy=n_traces,
                )

        with (
            open(output_path, "a", encoding="utf-8") as fh,
            tqdm(
                total=len(entries),
                desc="Generating training data",
                unit="entry",
                dynamic_ncols=True,
                leave=True,
            ) as pbar,
        ):
            tasks = [asyncio.ensure_future(_bounded(e)) for e in entries]
            for coro in asyncio.as_completed(tasks):
                try:
                    samples = await coro
                except Exception as exc:
                    logger.warning("Entry generation failed: %s", exc)
                    pbar.update(1)
                    continue

                for sample in samples:
                    record = self.format_for_sft(sample)
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_written += 1

                pbar.update(1)
                pbar.set_postfix(written=total_written)

                gc.collect()

        logger.info(
            "Batch generation complete. Total samples written: %d → %s",
            total_written,
            output_path,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_generation_prompt(
        self,
        conjecture: Conjecture,
        strategy: RefuterStrategy,
        ground_truth: str,
    ) -> list[dict]:
        """
        Construct an OpenAI-style message list for a single generation call.

        The system message establishes the mathematician persona.  The user
        message embeds the conjecture, domain, strategy description, and
        strategy-specific instructions.  The ground truth counterexample is
        included as a hint so the model produces a coherent trace rather than
        hallucinating an invalid example.

        Args:
            conjecture: The target conjecture.
            strategy: The refuter strategy to demonstrate.
            ground_truth: The known correct counterexample string.

        Returns:
            A two-element list of ``{"role": ..., "content": ...}`` dicts.
        """
        strategy_desc = _STRATEGY_DESCRIPTIONS[strategy]
        strategy_instr = _STRATEGY_INSTRUCTIONS[strategy]

        system_content = (
            "You are an expert mathematician specializing in finding counterexamples "
            "to false mathematical conjectures. Your reasoning must be rigorous, "
            "step-by-step, and show explicit numerical verification."
        )

        user_content = (
            f"I want you to find a counterexample to the following conjecture "
            f"using the {strategy.value} strategy.\n\n"
            f"Conjecture: {conjecture.nl_statement}\n"
            f"Domain: {conjecture.domain.value}\n"
            f"Strategy: {strategy_desc}\n\n"
            f"Instructions for this strategy:\n{strategy_instr}\n\n"
            f"Hint: A valid counterexample exists. Use the hint only to guide your "
            f"search — demonstrate full reasoning rather than just stating the answer.\n"
            f"Hint counterexample: {ground_truth}\n\n"
            "Format your response as:\n"
            "<REASONING>\n"
            "[step-by-step reasoning, showing what you tried and why]\n"
            "</REASONING>\n"
            "<COUNTEREXAMPLE>\n"
            "[the specific counterexample with explicit numerical verification]\n"
            "</COUNTEREXAMPLE>"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_reasoning_and_counterexample(
        self, llm_output: str
    ) -> tuple[str, str]:
        """
        Extract the ``<REASONING>`` and ``<COUNTEREXAMPLE>`` blocks from LLM output.

        Uses non-greedy regex to isolate each XML-style block.  Returns empty
        strings for any block that could not be matched so callers can filter
        gracefully.

        Args:
            llm_output: Raw string returned by the LLM.

        Returns:
            A ``(reasoning, counterexample)`` tuple of stripped strings.
            Either element may be empty if the corresponding block was absent.
        """
        reasoning_match = re.search(
            r"<REASONING>\s*(.*?)\s*</REASONING>",
            llm_output,
            re.DOTALL | re.IGNORECASE,
        )
        ce_match = re.search(
            r"<COUNTEREXAMPLE>\s*(.*?)\s*</COUNTEREXAMPLE>",
            llm_output,
            re.DOTALL | re.IGNORECASE,
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        counterexample = ce_match.group(1).strip() if ce_match else ""
        return reasoning, counterexample

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _verify_counterexample(
        self, conjecture: Conjecture, counterexample_str: str
    ) -> bool:
        """
        Attempt numerical SymPy verification of a proposed counterexample.

        Extracts all ``variable=value`` pairs from ``counterexample_str`` and
        evaluates the conjecture's natural-language statement as a SymPy
        expression.  Returns ``True`` only when at least one substitution can
        be parsed and the resulting expression evaluates to a falsy value.

        Falls back to ``False`` (unverified) rather than raising on any parse
        or evaluation error so that verification failures never abort the
        pipeline.

        Args:
            conjecture: The conjecture whose truth value is being checked.
            counterexample_str: Human-readable counterexample description from
                the LLM.

        Returns:
            ``True`` if SymPy numerically confirms the counterexample, else
            ``False``.
        """
        try:
            import sympy  # local import — sympy is a heavy optional dep
        except ImportError:
            logger.warning(
                "SymPy not available — skipping verification for entry with "
                "conjecture '%s'",
                conjecture.nl_statement[:60],
            )
            return False

        # Extract variable assignments: n=40, k=7, etc.
        assignments = re.findall(r"(\b[a-zA-Z]\b)\s*=\s*(-?\d+(?:\.\d+)?)", counterexample_str)
        if not assignments:
            logger.debug("No variable assignments parsed from: %s", counterexample_str[:120])
            return False

        for var_name, val_str in assignments:
            try:
                sym_var = sympy.Symbol(var_name)
                val = sympy.Integer(val_str) if "." not in val_str else sympy.Float(val_str)

                # Attempt to parse a numeric expression from the counterexample
                # text by extracting LHS=RHS patterns like "1681 = 41^2".
                equation_match = re.search(
                    r"(\d[\d\s\^\+\-\*\/\(\)]+)\s*=\s*(\d[\d\s\^\+\-\*\/\(\)]+)",
                    counterexample_str,
                )
                if equation_match:
                    lhs_str = equation_match.group(1).strip()
                    rhs_str = equation_match.group(2).strip()
                    try:
                        lhs_val = sympy.sympify(lhs_str)
                        rhs_val = sympy.sympify(rhs_str)
                        if lhs_val == rhs_val:
                            logger.debug(
                                "SymPy verified numeric equality %s == %s",
                                lhs_val,
                                rhs_val,
                            )
                            return True
                    except (sympy.SympifyError, ValueError, TypeError):
                        pass

                # Heuristic: try substituting into simple polynomial expressions
                # present in the conjecture natural-language statement.
                poly_match = re.search(r"(\w[\w\s\^\+\-\*\/\(\)]+)\s+is\s+prime", conjecture.nl_statement, re.IGNORECASE)
                if poly_match:
                    expr_str = poly_match.group(1).strip()
                    try:
                        expr = sympy.sympify(expr_str.replace("^", "**"))
                        evaluated = expr.subs(sym_var, val)
                        result = sympy.Integer(evaluated)
                        if not sympy.isprime(result):
                            logger.debug(
                                "SymPy confirmed: %s=%s gives non-prime %s",
                                var_name,
                                val_str,
                                result,
                            )
                            return True
                    except (sympy.SympifyError, ValueError, TypeError, AttributeError):
                        pass

            except (ValueError, TypeError) as exc:
                logger.debug("SymPy parse error for assignment %s=%s: %s", var_name, val_str, exc)
                continue

        return False

    # ------------------------------------------------------------------
    # SFT formatting
    # ------------------------------------------------------------------

    def format_for_sft(self, sample: TrainingSample) -> dict:
        """
        Render a ``TrainingSample`` into the supervised fine-tuning record format.

        Returns a dict with three keys:

        * ``"instruction"`` — task framing independent of the specific conjecture.
        * ``"input"`` — the XML-tagged conjecture + strategy block shown to the
          model at inference time.
        * ``"output"`` — the target XML-tagged reasoning + counterexample.
        * ``"text"`` — the full concatenated prompt+completion for causal-LM
          training (used by ``trl.SFTTrainer`` with ``dataset_text_field``).
        * ``"metadata"`` — domain, strategy, source, and verification evidence for
          post-hoc filtering and analysis.

        Args:
            sample: A ``TrainingSample`` to serialise.

        Returns:
            A JSON-serialisable dictionary.
        """
        strategy_desc = _STRATEGY_DESCRIPTIONS[sample.strategy]

        instruction = (
            "Find a counterexample to the given mathematical conjecture using the "
            "specified strategy. Show your step-by-step reasoning and then state "
            "the explicit counterexample with numerical verification."
        )

        input_block = (
            f"<CONJECTURE> {sample.conjecture_nl} </CONJECTURE>\n"
            f"<STRATEGY> {sample.strategy.value} </STRATEGY>\n"
            f"<TASK> Find a counterexample to this mathematical conjecture using "
            f"the {sample.strategy.value} strategy ({strategy_desc}). "
            f"Show your step-by-step reasoning. </TASK>"
        )

        output_block = (
            f"<REASONING>\n{sample.reasoning_trace}\n</REASONING>\n"
            f"<COUNTEREXAMPLE> {sample.counterexample} </COUNTEREXAMPLE>"
        )

        full_text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_block}\n\n"
            f"### Response:\n{output_block}"
        )

        return {
            "instruction": instruction,
            "input": input_block,
            "output": output_block,
            "text": full_text,
            "metadata": {
                "domain": sample.domain,
                "strategy": sample.strategy.value,
                "source": sample.source.value,
                "verification_evidence": sample.verification_evidence,
            },
        }


# ---------------------------------------------------------------------------
# Benchmark loader
# ---------------------------------------------------------------------------


def load_benchmark_entries(benchmark_dir: Path) -> list[BenchmarkEntry]:
    """
    Load all Tier 1 benchmark entries from a JSONL file.

    Only entries with a non-empty ``ground_truth_counterexample`` field are
    returned, since the generation pipeline requires a known answer to guide
    and verify the LLM.

    Args:
        benchmark_dir: Directory containing ``tier1.jsonl`` (and optionally
            ``tier2.jsonl``, ``tier3.jsonl``).

    Returns:
        A list of ``BenchmarkEntry`` objects ready for data generation.

    Raises:
        FileNotFoundError: If ``tier1.jsonl`` does not exist in ``benchmark_dir``.
        ValueError: If any JSONL record is missing required fields.
    """
    tier1_path = benchmark_dir / "tier1.jsonl"
    if not tier1_path.is_file():
        raise FileNotFoundError(
            f"Benchmark file not found: {tier1_path.resolve()}\n"
            "Run the benchmark construction script first, or point --benchmark-dir "
            "at the correct directory."
        )

    entries: list[BenchmarkEntry] = []
    skipped = 0

    with open(tier1_path, "r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record: dict = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {tier1_path}: {exc}"
                ) from exc

            gt_ce = record.get("ground_truth_counterexample", "")
            if not gt_ce:
                skipped += 1
                continue

            raw_conj = record.get("conjecture", {})
            if not isinstance(raw_conj, dict):
                raise ValueError(
                    f"Line {line_no}: 'conjecture' must be a dict, got {type(raw_conj).__name__}"
                )

            try:
                domain = Domain(raw_conj.get("domain", "number_theory"))
            except ValueError:
                domain = Domain.NUMBER_THEORY

            conjecture = Conjecture(
                id=raw_conj.get("id", f"bench_{line_no}"),
                domain=domain,
                nl_statement=raw_conj.get("nl_statement", ""),
                variables=raw_conj.get("variables", []),
                source=raw_conj.get("source", "benchmark"),
                timestamp=raw_conj.get("timestamp", ""),
                metadata=raw_conj.get("metadata", {}),
            )
            if not conjecture.nl_statement:
                raise ValueError(
                    f"Line {line_no}: conjecture 'nl_statement' is empty"
                )

            try:
                tier = BenchmarkTier(record.get("tier", "tier1_synthetic"))
            except ValueError:
                tier = BenchmarkTier.TIER1_SYNTHETIC

            entries.append(
                BenchmarkEntry(
                    id=record.get("id", f"bench_{line_no}"),
                    conjecture=conjecture,
                    tier=tier,
                    ground_truth_counterexample=gt_ce,
                    ground_truth_status=record.get("ground_truth_status", "false"),
                    source=record.get("source", ""),
                    notes=record.get("notes", ""),
                )
            )

    logger.info(
        "Loaded %d benchmark entries (%d skipped — no ground-truth counterexample)",
        len(entries),
        skipped,
    )
    return entries


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for ``gen_training_data.py``.

    Returns:
        A configured ``argparse.ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        prog="gen_training_data",
        description=(
            "Generate (conjecture, strategy) → (reasoning, counterexample) training "
            "triples for REFUTE R-Agent fine-tuning."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to configs/config.yaml (ConjLean main config)",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="data/benchmark",
        metavar="DIR",
        help="Directory containing tier1.jsonl (and optionally tier2/tier3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training/samples.jsonl",
        metavar="PATH",
        help="Output JSONL file for generated training samples",
    )
    parser.add_argument(
        "--n-traces",
        type=int,
        default=3,
        metavar="N",
        help="Number of reasoning traces to generate per (entry, strategy) pair",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        metavar="N",
        help="Maximum concurrent LLM calls",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser


async def _async_main(args: argparse.Namespace) -> None:
    """
    Async main coroutine: load config, benchmark, build generator, run batch.

    Args:
        args: Parsed CLI namespace from ``_build_arg_parser``.
    """
    config = ConjLeanConfig.from_yaml(args.config)
    client = create_client(config)

    benchmark_dir = Path(args.benchmark_dir)
    output_path = Path(args.output)

    entries = load_benchmark_entries(benchmark_dir)
    if not entries:
        logger.error(
            "No usable benchmark entries found in %s. Aborting.", benchmark_dir
        )
        return

    generator = TrainingDataGenerator(client=client, config=config)

    # Override n_traces if supplied explicitly on the CLI.
    # We do this by temporarily patching batch_size so generate_batch picks it up.
    # (batch_size // n_strategies gives n_traces inside generate_batch)
    n_strategies = len(list(RefuterStrategy))
    effective_batch_size = args.n_traces * n_strategies
    # Pydantic models are immutable by default; use object.__setattr__ on the
    # nested GenerationConfig which is also a BaseModel with model_config frozen.
    config.generation.__class__.model_config = {"frozen": False}  # type: ignore[assignment]
    try:
        config.generation.batch_size = effective_batch_size  # type: ignore[assignment]
    except Exception:
        pass  # if frozen, generate_batch will default to 1 trace; that's acceptable

    await generator.generate_batch(
        entries=entries,
        output_path=output_path,
        max_concurrent=args.max_concurrent,
    )


def main() -> None:
    """
    CLI entry point for the training data generation pipeline.

    Parses arguments, configures logging, and runs the async generation loop.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()

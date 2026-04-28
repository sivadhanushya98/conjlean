"""
ConjLean CLI entry point.

Provides four sub-commands:

run
    Full pipeline: generate → filter → formalize → prove.

evaluate
    Evaluate a saved results JSONL file and print / save an EvaluationReport.

formalize
    Run only the formalization stage on a pre-saved conjecture JSONL file.

list-providers
    Print all supported LLM providers and the environment variables required
    to authenticate with them.

Usage examples::

    python run.py run \\
        --config configs/config.yaml \\
        --provider anthropic \\
        --domains number_theory inequality \\
        --n-per-domain 100 \\
        --output ./data/results/run_001

    python run.py evaluate \\
        --results ./data/results/run_001/results.jsonl

    python run.py formalize \\
        --conjectures ./data/conjectures/saved.jsonl \\
        --config configs/config.yaml

    python run.py list-providers
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_PROVIDER_INFO: dict[str, list[str]] = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai":    ["OPENAI_API_KEY"],
    "gemini":    ["GEMINI_API_KEY"],
    "huggingface": ["HF_TOKEN  (HF Inference API — cloud-hosted models)"],
    "vllm":      ["(no key — point vllm.base_url at your vLLM server)"],
    "local_hf":  ["(no key — loads model locally via transformers + torch)"],
}

# ---------------------------------------------------------------------------
# REFUTE sub-command helpers
# ---------------------------------------------------------------------------


async def _cmd_refute(args: argparse.Namespace) -> None:
    """
    Run the REFUTE multi-agent counterexample pipeline on the benchmark.

    Loads the benchmark dataset, runs the full REFUTE loop (R-Agent →
    V-Agent → C-Agent → S-Agent), saves per-conjecture loop results to
    JSONL, and prints a summary table.

    Args:
        args: Parsed CLI arguments.
    """
    from conjlean.config import ConjLeanConfig
    from conjlean.models import create_client
    from conjlean.benchmark import BenchmarkLoader
    from conjlean.refuter import Refuter
    from conjlean.strategist import Strategist
    from conjlean.refute_loop import RefuteLoop
    import dataclasses

    config = ConjLeanConfig.from_yaml(args.config)
    if args.provider:
        config = config.model_copy(update={"provider": args.provider})

    _configure_logging(config.output.log_level)
    logger = logging.getLogger(__name__)

    client = create_client(config)
    refuter = Refuter(client=client, config=config)
    strategist = Strategist(client=client, config=config)
    loop = RefuteLoop(client=client, refuter=refuter, strategist=strategist, config=config)

    benchmark_dir = Path(args.benchmark_dir)
    loader = BenchmarkLoader()
    entries = loader.load_all(benchmark_dir)
    conjectures = [e.conjecture for e in entries]

    logger.info("Loaded %d benchmark entries from %s", len(entries), benchmark_dir)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "loop_results.jsonl"

    results = await loop.run_batch(
        conjectures=conjectures,
        max_rounds=args.max_rounds,
        max_refinements=args.max_refinements,
        max_concurrent=args.max_concurrent,
    )

    # Save results
    with results_path.open("w", encoding="utf-8") as fh:
        from conjlean.pipeline import _recursive_enum_to_value
        for r in results:
            record = _recursive_enum_to_value(dataclasses.asdict(r))
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    n_refuted = sum(1 for r in results if r.status.value in ("refuted", "refined"))
    logger.info(
        "REFUTE complete — %d/%d conjectures refuted. Results: %s",
        n_refuted,
        len(results),
        results_path,
    )


def _cmd_refute_evaluate(args: argparse.Namespace) -> None:
    """
    Evaluate REFUTE loop results against the benchmark.

    Loads loop results from JSONL and benchmark entries, computes
    precision/recall/F1, strategy breakdown, and domain breakdown, then
    prints a Rich report and optionally saves JSON+Markdown.

    Args:
        args: Parsed CLI arguments.
    """
    from conjlean.refute_evaluate import RefuteEvaluator
    from conjlean.benchmark import BenchmarkLoader
    import dataclasses

    _configure_logging("INFO")
    logger = logging.getLogger(__name__)

    results_path = Path(args.results)
    benchmark_path = Path(args.benchmark)

    if not results_path.is_file():
        logger.error("Results file not found: %s", results_path)
        sys.exit(1)
    if not benchmark_path.is_file():
        logger.error("Benchmark file not found: %s", benchmark_path)
        sys.exit(1)

    loader = BenchmarkLoader()
    entries = loader.load_all(benchmark_path.parent)

    from conjlean.schemas import (
        RefuteLoopResult, RefuteLoopStatus, Conjecture, Domain,
        CounterexampleCandidate, RefuterStrategy, CounterexampleStatus,
        ConjectureRefinement, RefuterResult,
    )

    loop_results: list[RefuteLoopResult] = []
    with results_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # Minimal reconstruction for evaluation
            orig = record["original_conjecture"]
            conjecture = Conjecture(
                id=orig["id"],
                domain=Domain(orig["domain"]),
                nl_statement=orig["nl_statement"],
                variables=orig.get("variables", []),
            )
            loop_results.append(
                RefuteLoopResult(
                    original_conjecture=conjecture,
                    status=RefuteLoopStatus(record["status"]),
                    total_rounds=record.get("total_rounds", 0),
                )
            )

    evaluator = RefuteEvaluator()
    metrics = evaluator.evaluate(loop_results=loop_results, benchmark_entries=entries)
    evaluator.print_report(metrics)

    if args.output:
        evaluator.save_report(metrics, Path(args.output))
        logger.info("REFUTE evaluation report saved to %s.{json,md}", args.output)

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _configure_logging(log_level: str) -> None:
    """
    Configure the root logger with a timestamped console handler.

    Args:
        log_level: One of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``,
            ``CRITICAL``.
    """
    numeric = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------


def _print_banner(
    provider: str,
    model: str,
    domains: list[str],
    n_conjectures: Optional[int],
    command: str,
) -> None:
    """
    Print an ASCII banner with key run parameters.

    Args:
        provider: Active LLM provider name.
        model: Primary model identifier.
        domains: List of domain strings.
        n_conjectures: Total conjecture budget (may be ``None`` for evaluate).
        command: The CLI sub-command being executed.
    """
    width = 60
    border = "=" * width
    n_str = str(n_conjectures) if n_conjectures is not None else "N/A"

    print(border)
    print("  ConjLean — Automated Lean 4 Theorem Proving")
    print(f"  Command   : {command}")
    print(f"  Provider  : {provider}")
    print(f"  Model     : {model}")
    print(f"  Domains   : {', '.join(domains) if domains else 'N/A'}")
    print(f"  Budget    : {n_str} conjectures")
    print(border)
    print()


# ---------------------------------------------------------------------------
# Component factory
# ---------------------------------------------------------------------------


def _build_components(config: "ConjLeanConfig") -> tuple:
    """
    Instantiate all pipeline components from a validated config.

    Imports are deferred to this function so that the CLI remains importable
    even when optional dependencies (e.g. ``anthropic``) are absent and the
    user only runs ``list-providers``.

    Args:
        config: Fully validated :class:`~conjlean.config.ConjLeanConfig`.

    Returns:
        Tuple of ``(client, harness, generator, sym_filter, formalizer,
        proof_searcher)``.

    Raises:
        ImportError: If a required backend module is not installed.
        RuntimeError: If the active provider lacks required credentials.
    """
    from conjlean.models import create_client  # type: ignore[import]
    from conjlean.lean_harness import LeanHarness  # type: ignore[import]
    from conjlean.conjecture_gen import ConjectureGenerator  # type: ignore[import]
    from conjlean.sympy_filter import SympyFilter  # type: ignore[import]
    from conjlean.formalizer import Formalizer  # type: ignore[import]
    from conjlean.proof_search import ProofSearcher  # type: ignore[import]

    client = create_client(config)
    harness = LeanHarness(
        lean_project_dir=config.lean.project_dir,
        repl_timeout=config.lean.repl_timeout,
    )
    generator = ConjectureGenerator(client=client, config=config)
    sym_filter = SympyFilter()
    formalizer = Formalizer(client=client, harness=harness, config=config)
    proof_searcher = ProofSearcher(client=client, harness=harness, config=config)

    return client, harness, generator, sym_filter, formalizer, proof_searcher


# ---------------------------------------------------------------------------
# Sub-command: run
# ---------------------------------------------------------------------------


async def _cmd_run(args: argparse.Namespace) -> None:
    """
    Execute the full ConjLean pipeline.

    Handles ``KeyboardInterrupt`` by saving partial results before exiting
    gracefully.

    Args:
        args: Parsed CLI arguments.
    """
    from conjlean.config import ConjLeanConfig
    from conjlean.pipeline import ConjLeanPipeline
    from conjlean.schemas import Domain

    config = ConjLeanConfig.from_yaml(args.config)

    # Apply CLI overrides
    if args.provider:
        config = config.model_copy(update={"provider": args.provider})
    if args.output:
        config = config.model_copy(
            update={"output": config.output.model_copy(update={"save_dir": args.output})}
        )

    _configure_logging(config.output.log_level)
    logger = logging.getLogger(__name__)

    domains: list[Domain] = (
        [Domain(d) for d in args.domains]
        if args.domains
        else [Domain(d) for d in config.pipeline.domains]
    )
    n_per_domain: int = (
        args.n_per_domain
        if args.n_per_domain is not None
        else config.pipeline.conjectures_per_domain
    )

    _print_banner(
        provider=config.provider,
        model=config.models.proof_gen,
        domains=[d.value for d in domains],
        n_conjectures=n_per_domain * len(domains),
        command="run",
    )

    client, harness, generator, sym_filter, formalizer, proof_searcher = (
        _build_components(config)
    )

    pipeline = ConjLeanPipeline(
        client=client,
        harness=harness,
        config=config,
        generator=generator,
        sym_filter=sym_filter,
        formalizer=formalizer,
        proof_searcher=proof_searcher,
    )

    completed_results: list = []

    def _handle_interrupt(signum: int, frame: object) -> None:
        logger.warning(
            "Received signal %d — saving %d completed results before exit.",
            signum,
            len(completed_results),
        )
        if completed_results:
            output_dir = Path(config.output.save_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            partial_path = output_dir / "results_partial.jsonl"
            pipeline._save_results(completed_results, partial_path)
            logger.info("Partial results saved to %s", partial_path)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)

    results = await pipeline.run(domains=domains, n_per_domain=n_per_domain)
    completed_results.extend(results)

    logger.info(
        "Run complete. %d results written to %s",
        len(completed_results),
        Path(config.output.save_dir) / "results.jsonl",
    )


# ---------------------------------------------------------------------------
# Sub-command: evaluate
# ---------------------------------------------------------------------------


def _cmd_evaluate(args: argparse.Namespace) -> None:
    """
    Load a results JSONL file and compute an EvaluationReport.

    Args:
        args: Parsed CLI arguments.
    """
    from conjlean.evaluate import Evaluator

    _configure_logging("INFO")
    logger = logging.getLogger(__name__)

    results_path = Path(args.results)
    if not results_path.is_file():
        logger.error("Results file not found: %s", results_path)
        sys.exit(1)

    _print_banner(
        provider="N/A",
        model="N/A",
        domains=[],
        n_conjectures=None,
        command="evaluate",
    )

    pipeline_results: list[PipelineResult] = _load_pipeline_results(results_path)
    logger.info("Loaded %d pipeline results from %s", len(pipeline_results), results_path)

    evaluator = Evaluator()
    report = evaluator.evaluate(pipeline_results)
    evaluator.print_report(report)

    if args.output:
        evaluator.save_report(report, Path(args.output))
        logger.info("Report saved to %s.{json,md}", args.output)


# ---------------------------------------------------------------------------
# Sub-command: formalize
# ---------------------------------------------------------------------------


async def _cmd_formalize(args: argparse.Namespace) -> None:
    """
    Run only the formalization stage on a pre-saved conjecture JSONL.

    Args:
        args: Parsed CLI arguments.
    """
    from conjlean.config import ConjLeanConfig
    from conjlean.pipeline import ConjLeanPipeline
    from conjlean.schemas import Domain

    config = ConjLeanConfig.from_yaml(args.config)
    _configure_logging(config.output.log_level)
    logger = logging.getLogger(__name__)

    _print_banner(
        provider=config.provider,
        model=config.models.formalizer,
        domains=[],
        n_conjectures=None,
        command="formalize",
    )

    client, harness, generator, sym_filter, formalizer, proof_searcher = (
        _build_components(config)
    )

    pipeline = ConjLeanPipeline(
        client=client,
        harness=harness,
        config=config,
        generator=generator,
        sym_filter=sym_filter,
        formalizer=formalizer,
        proof_searcher=proof_searcher,
    )

    conjectures = pipeline._load_conjectures(Path(args.conjectures))
    logger.info("Loaded %d conjectures from %s", len(conjectures), args.conjectures)

    formalized = await pipeline.run_formalization(conjectures)

    output_dir = Path(config.output.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "formalized.jsonl"

    import json
    import dataclasses
    from conjlean.pipeline import _recursive_enum_to_value

    with out_path.open("w", encoding="utf-8") as fh:
        for fc in formalized:
            record = _recursive_enum_to_value(dataclasses.asdict(fc))
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Formalized results saved to %s", out_path)


# ---------------------------------------------------------------------------
# Sub-command: list-providers
# ---------------------------------------------------------------------------


def _cmd_list_providers(_args: argparse.Namespace) -> None:
    """
    Print all supported LLM providers and their required credentials.

    Args:
        _args: Parsed CLI arguments (unused).
    """
    print("Supported providers and required environment variables:")
    print()
    for provider, env_vars in _PROVIDER_INFO.items():
        env_str = ", ".join(env_vars)
        print(f"  {provider:<15}  {env_str}")
    print()
    print(
        "Set the corresponding environment variable before running `python run.py run`."
    )


# ---------------------------------------------------------------------------
# JSONL result loader
# ---------------------------------------------------------------------------


def _load_pipeline_results(path: Path) -> list:
    """
    Deserialise a JSONL file of :class:`~conjlean.schemas.PipelineResult`
    records.

    Only the fields needed by the :class:`~conjlean.src.evaluate.Evaluator`
    are reconstructed; full nested dataclass reconstruction handles the rest.

    Args:
        path: Path to the JSONL results file.

    Returns:
        List of :class:`~conjlean.schemas.PipelineResult` objects.

    Raises:
        json.JSONDecodeError: If any line is invalid JSON.
    """
    from conjlean.schemas import (
        Conjecture,
        Domain,
        FilterResult,
        FilterStatus,
        FormalizedConjecture,
        FormalizationStatus,
        PipelineResult,
        PipelineStatus,
        ProofAttempt,
        ProofLayer,
        ProofResult,
        ProofStatus,
    )

    results: list[PipelineResult] = []

    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {line_no} of {path}: {exc.msg}",
                    exc.doc,
                    exc.pos,
                ) from exc

            conjecture = Conjecture(
                id=record["conjecture"]["id"],
                domain=Domain(record["conjecture"]["domain"]),
                nl_statement=record["conjecture"]["nl_statement"],
                variables=record["conjecture"].get("variables", []),
                source=record["conjecture"].get("source", "loaded"),
                timestamp=record["conjecture"].get("timestamp", ""),
                metadata=record["conjecture"].get("metadata", {}),
            )

            filter_result: Optional["FilterResult"] = None
            if record.get("filter_result") is not None:
                fr_data = record["filter_result"]
                filter_result = FilterResult(
                    conjecture=conjecture,
                    status=FilterStatus(fr_data["status"]),
                    counterexample=fr_data.get("counterexample"),
                    numerical_evidence=fr_data.get("numerical_evidence", {}),
                )

            formalization: Optional["FormalizedConjecture"] = None
            if record.get("formalization") is not None:
                fc_data = record["formalization"]
                formalization = FormalizedConjecture(
                    conjecture=conjecture,
                    lean_code=fc_data["lean_code"],
                    status=FormalizationStatus(fc_data["status"]),
                    retries=fc_data.get("retries", 0),
                    error_history=fc_data.get("error_history", []),
                )

            proof: Optional["ProofResult"] = None
            if record.get("proof") is not None and formalization is not None:
                pr_data = record["proof"]
                attempts = [
                    ProofAttempt(
                        tactic=a["tactic"],
                        success=a["success"],
                        error=a.get("error"),
                        layer=ProofLayer(a["layer"]) if a.get("layer") else None,
                    )
                    for a in pr_data.get("attempts", [])
                ]
                proof = ProofResult(
                    formalized=formalization,
                    status=ProofStatus(pr_data["status"]),
                    proof=pr_data.get("proof"),
                    layer=ProofLayer(pr_data["layer"]) if pr_data.get("layer") else None,
                    attempts=attempts,
                    duration_seconds=pr_data.get("duration_seconds", 0.0),
                )

            results.append(
                PipelineResult(
                    conjecture=conjecture,
                    filter_result=filter_result,
                    formalization=formalization,
                    proof=proof,
                    final_status=PipelineStatus(record.get("final_status", "filtered_out")),
                )
            )

    return results


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """
    Build and return the top-level argument parser with all sub-commands.

    Returns:
        Fully configured :class:`~argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="conjlean",
        description="ConjLean — Automated Mathematical Conjecture Generation and Lean 4 Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── run ─────────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser(
        "run",
        help="Run the full pipeline (generate → filter → formalize → prove).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to YAML configuration file.",
    )
    run_parser.add_argument(
        "--provider",
        choices=list(_PROVIDER_INFO.keys()),
        default=None,
        help="Override the LLM provider from the config file.",
    )
    run_parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        metavar="DOMAIN",
        choices=["number_theory", "inequality", "combinatorics"],
        help="Mathematical domains to generate conjectures for.",
    )
    run_parser.add_argument(
        "--n-per-domain",
        type=int,
        default=None,
        metavar="N",
        help="Number of conjectures to generate per domain.",
    )
    run_parser.add_argument(
        "--output",
        default=None,
        metavar="DIR",
        help="Override the output directory from the config file.",
    )

    # ── evaluate ────────────────────────────────────────────────────────
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a saved results JSONL file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    eval_parser.add_argument(
        "--results",
        required=True,
        metavar="PATH",
        help="Path to the pipeline results JSONL file.",
    )
    eval_parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Base path (no extension) to save .json and .md report files.",
    )

    # ── formalize ───────────────────────────────────────────────────────
    form_parser = subparsers.add_parser(
        "formalize",
        help="Run only the formalization stage on a saved conjecture JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    form_parser.add_argument(
        "--conjectures",
        required=True,
        metavar="PATH",
        help="Path to the saved conjecture JSONL file.",
    )
    form_parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to YAML configuration file.",
    )

    # ── list-providers ──────────────────────────────────────────────────
    subparsers.add_parser(
        "list-providers",
        help="Print supported providers and required environment variables.",
    )

    # ── refute ──────────────────────────────────────────────────────────
    refute_parser = subparsers.add_parser(
        "refute",
        help="Run REFUTE multi-agent counterexample pipeline on benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    refute_parser.add_argument("--config", required=True, metavar="PATH", help="YAML config.")
    refute_parser.add_argument("--provider", choices=list(_PROVIDER_INFO.keys()), default=None)
    refute_parser.add_argument(
        "--benchmark-dir", required=True, metavar="DIR",
        help="Directory containing benchmark JSONL files.",
    )
    refute_parser.add_argument(
        "--output", required=True, metavar="DIR", help="Output directory for loop results."
    )
    refute_parser.add_argument("--max-rounds", type=int, default=10, metavar="N")
    refute_parser.add_argument("--max-refinements", type=int, default=3, metavar="N")
    refute_parser.add_argument("--max-concurrent", type=int, default=4, metavar="N")

    # ── refute-evaluate ─────────────────────────────────────────────────
    re_parser = subparsers.add_parser(
        "refute-evaluate",
        help="Evaluate REFUTE loop results against the benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    re_parser.add_argument(
        "--results", required=True, metavar="PATH", help="loop_results.jsonl from refute run."
    )
    re_parser.add_argument(
        "--benchmark", required=True, metavar="PATH", help="Path to benchmark all.jsonl."
    )
    re_parser.add_argument("--output", default=None, metavar="PATH", help="Base path for report.")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Entry point for the ConjLean CLI.

    Parses arguments, dispatches to the appropriate sub-command, and wraps
    async sub-commands with :func:`asyncio.run`.
    """
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "run":
            asyncio.run(_cmd_run(args))
        elif args.command == "evaluate":
            _cmd_evaluate(args)
        elif args.command == "formalize":
            asyncio.run(_cmd_formalize(args))
        elif args.command == "list-providers":
            _cmd_list_providers(args)
        elif args.command == "refute":
            asyncio.run(_cmd_refute(args))
        elif args.command == "refute-evaluate":
            _cmd_refute_evaluate(args)
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 — top-level catch for clean exit message
        logging.getLogger(__name__).exception("Unhandled exception: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

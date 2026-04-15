"""
End-to-end orchestration pipeline for ConjLean.

The :class:`ConjLeanPipeline` coordinates all four stages of the pipeline:

1. **Generation** — produce raw conjectures via the LLM
2. **Filtering** — symbolically / numerically filter conjectures with SymPy
3. **Formalization** — translate surviving conjectures to Lean 4 statements
4. **Proof Search** — attempt to close each formalized statement

Results are saved incrementally as JSONL so that a partial run is never lost.
Stage-level :mod:`tqdm` bars and structured :mod:`logging` provide real-time
visibility into pipeline progress.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Optional

from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

from conjlean.config import ConjLeanConfig
from conjlean.schemas import (
    Conjecture,
    Domain,
    FilterResult,
    FilterStatus,
    FormalizedConjecture,
    FormalizationStatus,
    PipelineResult,
    PipelineStatus,
    ProofResult,
    ProofStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forward-declared protocol stubs (runtime implementations injected at init)
# ---------------------------------------------------------------------------


class ConjectureGenerator:
    """Protocol: generates raw conjectures for a given domain."""

    async def generate(self, domain: Domain, n: int) -> list[Conjecture]:
        raise NotImplementedError


class SympyFilter:
    """Protocol: symbolically and numerically filters a conjecture."""

    def filter(self, conjecture: Conjecture) -> FilterResult:
        raise NotImplementedError


class Formalizer:
    """Protocol: translates a conjecture to a Lean 4 statement."""

    async def formalize(self, conjecture: Conjecture) -> FormalizedConjecture:
        raise NotImplementedError


class ProofSearcher:
    """Protocol: runs layered proof search on a formalized conjecture."""

    async def search(self, formalized: FormalizedConjecture) -> ProofResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ConjLeanPipeline:
    """
    End-to-end pipeline: generate → filter → formalize → prove.

    Orchestrates all four pipeline components, saves results incrementally
    to JSONL, and emits structured log summaries after each stage.

    Components are injected via constructor parameters to facilitate unit
    testing and alternative backend substitution without subclassing.

    Args:
        client: Async LLM client (passed through to generator and formalizer).
        harness: Lean 4 REPL harness (passed through to proof searcher).
        config: Validated pipeline configuration.
        generator: Conjecture generation component.
        sym_filter: Symbolic / numerical filter component.
        formalizer: Lean 4 formalization component.
        proof_searcher: Layered proof search component.
    """

    def __init__(
        self,
        client: object,
        harness: object,
        config: ConjLeanConfig,
        generator: ConjectureGenerator,
        sym_filter: SympyFilter,
        formalizer: Formalizer,
        proof_searcher: ProofSearcher,
    ) -> None:
        self._client = client
        self._harness = harness
        self._config = config
        self._generator = generator
        self._filter = sym_filter
        self._formalizer = formalizer
        self._proof_searcher = proof_searcher

        self._save_dir = Path(config.output.save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        domains: Optional[list[Domain]] = None,
        n_per_domain: Optional[int] = None,
        conjectures: Optional[list[Conjecture]] = None,
    ) -> list[PipelineResult]:
        """
        Execute the full pipeline and return per-conjecture results.

        If ``conjectures`` is provided, the generation stage is skipped and
        the supplied list is used directly (useful for re-running proof
        search on a previously saved conjecture set).

        Results are written incrementally to
        ``<save_dir>/results.jsonl`` so that a partial run is recoverable.

        Stage-level :mod:`tqdm` progress bars are displayed on stdout;
        compact summary tables are emitted via :mod:`logging` after each
        stage completes.

        Args:
            domains: Domains to generate conjectures for.  Defaults to
                ``config.pipeline.domains`` when omitted.
            n_per_domain: Conjectures to generate per domain.  Defaults to
                ``config.pipeline.conjectures_per_domain`` when omitted.
            conjectures: Pre-generated conjectures to skip the generation
                stage.  Mutually exclusive with ``domains`` / ``n_per_domain``.

        Returns:
            List of :class:`~conjlean.schemas.PipelineResult` objects, one
            per input conjecture.

        Raises:
            ValueError: If both ``conjectures`` and ``domains`` are supplied.
        """
        if conjectures is not None and domains is not None:
            raise ValueError(
                "Provide either 'conjectures' (skip generation) or 'domains', not both."
            )

        pipeline_start = time.monotonic()
        results_path = self._save_dir / "results.jsonl"

        # ── Stage 1: Generation ──────────────────────────────────────────
        if conjectures is not None:
            logger.info(
                "Skipping generation stage — using %d pre-loaded conjectures.",
                len(conjectures),
            )
            all_conjectures = conjectures
        else:
            resolved_domains = (
                [Domain(d) for d in self._config.pipeline.domains]
                if domains is None
                else domains
            )
            resolved_n = (
                self._config.pipeline.conjectures_per_domain
                if n_per_domain is None
                else n_per_domain
            )
            all_conjectures = await self.run_generation(resolved_domains, resolved_n)

        # ── Stage 2: Filtering ───────────────────────────────────────────
        filter_results = self.run_filtering(all_conjectures)
        surviving: list[Conjecture] = [
            fr.conjecture
            for fr in filter_results
            if fr.status is FilterStatus.SURVIVING
        ]

        # ── Stage 3: Formalization ───────────────────────────────────────
        formalized_list = await self.run_formalization(surviving)

        typechecked: list[FormalizedConjecture] = [
            fc for fc in formalized_list if fc.status is FormalizationStatus.TYPECHECKS
        ]

        # ── Stage 4: Proof Search ────────────────────────────────────────
        proof_results = await self.run_proof_search(typechecked)

        # ── Assemble PipelineResults ─────────────────────────────────────
        pipeline_results = _assemble_pipeline_results(
            all_conjectures=all_conjectures,
            filter_results=filter_results,
            formalized_list=formalized_list,
            proof_results=proof_results,
        )

        # ── Persist ──────────────────────────────────────────────────────
        self._save_results(pipeline_results, results_path)
        logger.info("Results saved to %s", results_path)

        elapsed = time.monotonic() - pipeline_start
        n_proved = sum(
            1 for r in pipeline_results if r.final_status is PipelineStatus.PROVED
        )
        logger.info(
            "Pipeline complete in %.1fs — %d/%d conjectures proved.",
            elapsed,
            n_proved,
            len(all_conjectures),
        )
        return pipeline_results

    # ------------------------------------------------------------------
    # Stage methods
    # ------------------------------------------------------------------

    async def run_generation(
        self, domains: list[Domain], n_per_domain: int
    ) -> list[Conjecture]:
        """
        Generate raw conjectures for each domain and persist them to disk.

        Each domain is generated sequentially with a :mod:`tqdm` bar tracking
        domain-level progress.  Results are saved to
        ``data/conjectures/<domain>.jsonl`` immediately after generation.

        Args:
            domains: Domains to generate for.
            n_per_domain: Target number of conjectures per domain.

        Returns:
            Flat list of all generated :class:`~conjlean.schemas.Conjecture`
            objects across all domains.
        """
        conjectures_dir = self._save_dir / "conjectures"
        conjectures_dir.mkdir(parents=True, exist_ok=True)

        all_conjectures: list[Conjecture] = []

        pbar = tqdm(domains, desc="Generating conjectures", unit="domain", dynamic_ncols=True)
        for domain in pbar:
            pbar.set_postfix_str(domain.value)
            domain_conjectures = await self._generator.generate(domain, n_per_domain)
            all_conjectures.extend(domain_conjectures)

            domain_path = conjectures_dir / f"{domain.value}.jsonl"
            _write_jsonl(domain_conjectures, domain_path)
            logger.info(
                "Generated %d conjectures for domain '%s' → %s",
                len(domain_conjectures),
                domain.value,
                domain_path,
            )

        self._log_stage_summary("generation", all_conjectures)
        return all_conjectures

    def run_filtering(self, conjectures: list[Conjecture]) -> list[FilterResult]:
        """
        Apply the symbolic / numerical filter to every conjecture.

        Runs synchronously with a :mod:`tqdm` bar.  Logs counts of surviving,
        disproved, and trivial conjectures after completion.

        Args:
            conjectures: Raw conjectures to filter.

        Returns:
            One :class:`~conjlean.schemas.FilterResult` per input conjecture.
        """
        results: list[FilterResult] = []

        pbar = tqdm(
            conjectures, desc="Filtering conjectures", unit="conj", dynamic_ncols=True
        )
        for conjecture in pbar:
            fr = self._filter.filter(conjecture)
            results.append(fr)
            pbar.set_postfix_str(fr.status.value)

        surviving = sum(1 for r in results if r.status is FilterStatus.SURVIVING)
        disproved = sum(1 for r in results if r.status is FilterStatus.DISPROVED)
        trivial = sum(1 for r in results if r.status is FilterStatus.TRIVIAL)
        logger.info(
            "Filtering complete — surviving: %d, disproved: %d, trivial: %d (total: %d)",
            surviving,
            disproved,
            trivial,
            len(results),
        )
        self._log_stage_summary("filtering", results)
        return results

    async def run_formalization(
        self, surviving: list[Conjecture]
    ) -> list[FormalizedConjecture]:
        """
        Formalize all surviving conjectures into Lean 4 statements.

        Launches all formalization coroutines concurrently and aggregates them
        with :func:`asyncio.gather`, tracking progress via an :mod:`tqdm` bar.

        Args:
            surviving: Conjectures that passed the filter stage.

        Returns:
            One :class:`~conjlean.schemas.FormalizedConjecture` per input
            conjecture.
        """
        logger.info("Starting formalization for %d conjectures.", len(surviving))

        tasks = [self._formalizer.formalize(c) for c in surviving]

        formalized: list[FormalizedConjecture] = []
        for coro in async_tqdm.as_completed(
            tasks,
            total=len(tasks),
            desc="Formalizing",
            unit="conj",
            dynamic_ncols=True,
        ):
            result = await coro
            formalized.append(result)

        typechecked = sum(
            1 for fc in formalized if fc.status is FormalizationStatus.TYPECHECKS
        )
        logger.info(
            "Formalization complete — typechecks: %d/%d (%.1f%%)",
            typechecked,
            len(formalized),
            100.0 * typechecked / max(len(formalized), 1),
        )
        self._log_stage_summary("formalization", formalized)
        return formalized

    async def run_proof_search(
        self, formalized: list[FormalizedConjecture]
    ) -> list[ProofResult]:
        """
        Run layered proof search on all type-checked formalized conjectures.

        Processes conjectures sequentially (each proof-search run can itself
        be highly parallel internally) and tracks progress via :mod:`tqdm`.
        Logs layer-by-layer closure counts after completion.

        Args:
            formalized: Formalized conjectures with ``status=TYPECHECKS``.

        Returns:
            One :class:`~conjlean.schemas.ProofResult` per input conjecture.
        """
        logger.info("Starting proof search for %d conjectures.", len(formalized))
        results: list[ProofResult] = []

        pbar = tqdm(
            formalized, desc="Proof search", unit="conj", dynamic_ncols=True
        )
        for fc in pbar:
            proof_result = await self._proof_searcher.search(fc)
            results.append(proof_result)
            status_tag = (
                f"✓ {proof_result.layer.value}"
                if proof_result.status is ProofStatus.PROVED
                else "open"
            )
            pbar.set_postfix_str(status_tag)

        proved = sum(1 for r in results if r.status is ProofStatus.PROVED)
        self._log_layer_breakdown(results)
        logger.info(
            "Proof search complete — proved: %d/%d (%.1f%%)",
            proved,
            len(results),
            100.0 * proved / max(len(results), 1),
        )
        self._log_stage_summary("proof_search", results)
        return results

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_results(self, results: list[PipelineResult], path: Path) -> None:
        """
        Serialise a list of :class:`~conjlean.schemas.PipelineResult` objects
        to JSONL (one JSON object per line).

        Uses :func:`dataclasses.asdict` for deep recursive serialization.
        Enum values are converted to their ``.value`` strings so the output is
        pure JSON.

        Args:
            results: Pipeline results to serialise.
            path: Destination file path (will be created or overwritten).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for result in results:
                record = _recursive_enum_to_value(dataclasses.asdict(result))
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_conjectures(self, path: Path) -> list[Conjecture]:
        """
        Load previously saved conjectures from a JSONL file.

        Each line is expected to be a JSON object whose fields match the
        :class:`~conjlean.schemas.Conjecture` dataclass.

        Args:
            path: Path to the JSONL file.

        Returns:
            Reconstructed list of :class:`~conjlean.schemas.Conjecture`
            objects.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            json.JSONDecodeError: If any line is invalid JSON.
        """
        if not path.is_file():
            raise FileNotFoundError(f"Conjecture file not found: {path}")

        conjectures: list[Conjecture] = []
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
                conjectures.append(
                    Conjecture(
                        id=record["id"],
                        domain=Domain(record["domain"]),
                        nl_statement=record["nl_statement"],
                        variables=record.get("variables", []),
                        source=record.get("source", "loaded"),
                        timestamp=record.get("timestamp", ""),
                        metadata=record.get("metadata", {}),
                    )
                )
        return conjectures

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_stage_summary(self, stage: str, results: list) -> None:
        """
        Emit a compact summary log line for a completed pipeline stage.

        Counts unique object types and statuses in ``results`` and formats
        them as a compact ``key=value`` table in a single ``INFO`` log entry.

        Args:
            stage: Human-readable stage name (e.g. ``"filtering"``).
            results: Objects produced by the completed stage.
        """
        counts: dict[str, int] = {}
        for item in results:
            # Try status attribute; fall back to type name
            status = getattr(item, "status", None)
            key = status.value if hasattr(status, "value") else type(item).__name__
            counts[key] = counts.get(key, 0) + 1

        summary_str = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        logger.info("[%s] total=%d  %s", stage.upper(), len(results), summary_str)

    def _log_layer_breakdown(self, proof_results: list[ProofResult]) -> None:
        """
        Log the count of proofs closed by each proof-search layer.

        Args:
            proof_results: Results from :meth:`run_proof_search`.
        """
        breakdown: dict[str, int] = {}
        for pr in proof_results:
            if pr.status is ProofStatus.PROVED and pr.layer is not None:
                key = pr.layer.value
                breakdown[key] = breakdown.get(key, 0) + 1

        if breakdown:
            layer_str = "  ".join(f"{k}={v}" for k, v in sorted(breakdown.items()))
            logger.info("[PROOF_SEARCH] layer_breakdown: %s", layer_str)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _assemble_pipeline_results(
    all_conjectures: list[Conjecture],
    filter_results: list[FilterResult],
    formalized_list: list[FormalizedConjecture],
    proof_results: list[ProofResult],
) -> list[PipelineResult]:
    """
    Merge stage outputs into a flat list of :class:`~conjlean.schemas.PipelineResult`.

    Builds lookup maps keyed by conjecture ID to join results across stages.

    Args:
        all_conjectures: Every conjecture that entered the pipeline.
        filter_results: One ``FilterResult`` per conjecture.
        formalized_list: Formalized conjectures for those that survived.
        proof_results: Proof results for those that were formalized.

    Returns:
        One :class:`~conjlean.schemas.PipelineResult` per input conjecture,
        with ``final_status`` set according to the furthest stage reached.
    """
    filter_map: dict[str, FilterResult] = {
        fr.conjecture.id: fr for fr in filter_results
    }
    formalized_map: dict[str, FormalizedConjecture] = {
        fc.conjecture.id: fc for fc in formalized_list
    }
    proof_map: dict[str, ProofResult] = {
        pr.formalized.conjecture.id: pr for pr in proof_results
    }

    pipeline_results: list[PipelineResult] = []
    for conjecture in all_conjectures:
        cid = conjecture.id
        fr = filter_map.get(cid)
        fc = formalized_map.get(cid)
        pr = proof_map.get(cid)

        final_status = _derive_final_status(fr, fc, pr)
        pipeline_results.append(
            PipelineResult(
                conjecture=conjecture,
                filter_result=fr,
                formalization=fc,
                proof=pr,
                final_status=final_status,
            )
        )

    return pipeline_results


def _derive_final_status(
    fr: Optional[FilterResult],
    fc: Optional[FormalizedConjecture],
    pr: Optional[ProofResult],
) -> PipelineStatus:
    """
    Derive the aggregate :class:`~conjlean.schemas.PipelineStatus` from stage outputs.

    Decision logic (in priority order):

    1. If no filter result → ``FILTERED_OUT``
    2. If filter status is ``DISPROVED`` → ``DISPROVED``
    3. If filter status is ``TRIVIAL`` → ``FILTERED_OUT``
    4. If no formalization → ``FILTERED_OUT``
    5. If formalization is ``UNFORMALIZABLE`` → ``UNFORMALIZABLE``
    6. If no proof result → ``OPEN``
    7. If proof status is ``PROVED`` → ``PROVED``
    8. Otherwise → ``OPEN``

    Args:
        fr: Filter result (may be ``None``).
        fc: Formalization result (may be ``None``).
        pr: Proof result (may be ``None``).

    Returns:
        The appropriate :class:`~conjlean.schemas.PipelineStatus`.
    """
    if fr is None:
        return PipelineStatus.FILTERED_OUT
    if fr.status is FilterStatus.DISPROVED:
        return PipelineStatus.DISPROVED
    if fr.status is FilterStatus.TRIVIAL:
        return PipelineStatus.FILTERED_OUT
    if fc is None:
        return PipelineStatus.FILTERED_OUT
    if fc.status is FormalizationStatus.UNFORMALIZABLE:
        return PipelineStatus.UNFORMALIZABLE
    if pr is None:
        return PipelineStatus.OPEN
    if pr.status is ProofStatus.PROVED:
        return PipelineStatus.PROVED
    return PipelineStatus.OPEN


def _write_jsonl(items: list, path: Path) -> None:
    """
    Write a list of dataclass instances to a JSONL file.

    Args:
        items: Dataclass instances to serialise.
        path: Output file path (parent directories created if absent).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            record = _recursive_enum_to_value(dataclasses.asdict(item))
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _recursive_enum_to_value(obj: object) -> object:
    """
    Recursively convert :class:`~enum.Enum` instances to their ``.value``
    within a nested dict / list structure produced by
    :func:`dataclasses.asdict`.

    Args:
        obj: Arbitrary nested structure (dict, list, Enum, or scalar).

    Returns:
        Same structure with all Enum instances replaced by their string values.
    """
    from enum import Enum

    if isinstance(obj, dict):
        return {k: _recursive_enum_to_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_recursive_enum_to_value(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    return obj

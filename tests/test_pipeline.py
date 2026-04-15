"""
Tests for conjlean.pipeline — ConjLeanPipeline end-to-end orchestration.

All four pipeline components (generator, filter, formalizer, proof searcher)
are mocked so tests run entirely in-memory with no external dependencies.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conjlean.config import ConjLeanConfig
from conjlean.pipeline import ConjLeanPipeline, _assemble_pipeline_results, _derive_final_status
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

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_LEAN_CODE = (
    "import Mathlib\n\n"
    "theorem test_thm (n : ℕ) : 2 ∣ n * (n + 1) := by\n"
    "  sorry"
)


def _conjectures(n: int, domain: Domain = Domain.NUMBER_THEORY) -> list[Conjecture]:
    """Create n Conjecture objects for testing."""
    return [
        Conjecture(
            id=f"conj_{i:03d}",
            domain=domain,
            nl_statement=f"Conjecture number {i}",
            variables=["n"],
        )
        for i in range(n)
    ]


def _make_config(save_dir: str = "/tmp/conjlean_test_results") -> ConjLeanConfig:
    """Build a minimal ConjLeanConfig pointing at a temp save directory."""
    cfg = ConjLeanConfig()
    cfg.output.save_dir = save_dir
    return cfg


def _make_pipeline(
    save_dir: str,
    generator: Any = None,
    sym_filter: Any = None,
    formalizer: Any = None,
    proof_searcher: Any = None,
    n_conjectures: int = 3,
) -> ConjLeanPipeline:
    """Build a ConjLeanPipeline with all components mocked."""
    cfg = _make_config(save_dir)

    conjs = _conjectures(n_conjectures)

    # Generator: returns n_conjectures per domain
    if generator is None:
        generator = MagicMock()
        generator.generate = AsyncMock(return_value=conjs)

    # Filter: all SURVIVING
    if sym_filter is None:
        sym_filter = MagicMock()
        sym_filter.filter = MagicMock(
            side_effect=lambda c: FilterResult(conjecture=c, status=FilterStatus.SURVIVING)
        )

    # Formalizer: all TYPECHECKS
    if formalizer is None:
        formalizer = MagicMock()
        formalizer.formalize = AsyncMock(
            side_effect=lambda c: FormalizedConjecture(
                conjecture=c,
                lean_code=_LEAN_CODE,
                status=FormalizationStatus.TYPECHECKS,
            )
        )

    # Proof searcher: all PROVED
    if proof_searcher is None:
        proof_searcher = MagicMock()

        async def _proved(fc: FormalizedConjecture) -> ProofResult:
            return ProofResult(
                formalized=fc,
                status=ProofStatus.PROVED,
                layer=ProofLayer.LAYER0_AUTO,
                proof=_LEAN_CODE.replace("sorry", "omega"),
                duration_seconds=0.1,
            )

        proof_searcher.search = AsyncMock(side_effect=_proved)

    client = MagicMock()
    harness = MagicMock()

    return ConjLeanPipeline(
        client=client,
        harness=harness,
        config=cfg,
        generator=generator,
        sym_filter=sym_filter,
        formalizer=formalizer,
        proof_searcher=proof_searcher,
    )


# ---------------------------------------------------------------------------
# TestPipelineEndToEnd
# ---------------------------------------------------------------------------


class TestPipelineEndToEnd:
    """Full pipeline run tests with all components mocked."""

    @pytest.mark.asyncio()
    async def test_full_pipeline_end_to_end(self) -> None:
        """3 conjectures, all succeed → 3 PipelineResult(PROVED)."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _make_pipeline(tmp, n_conjectures=3)
            results = await pipeline.run(
                conjectures=_conjectures(3)
            )

        assert len(results) == 3
        assert all(r.final_status is PipelineStatus.PROVED for r in results)

    @pytest.mark.asyncio()
    async def test_pipeline_handles_filtered_out(self) -> None:
        """Filter marks conjecture DISPROVED → PipelineResult(DISPROVED)."""
        with tempfile.TemporaryDirectory() as tmp:
            sym_filter = MagicMock()
            sym_filter.filter = MagicMock(
                side_effect=lambda c: FilterResult(
                    conjecture=c,
                    status=FilterStatus.DISPROVED,
                    counterexample="n=3 is a counterexample",
                )
            )

            pipeline = _make_pipeline(tmp, sym_filter=sym_filter, n_conjectures=2)
            results = await pipeline.run(conjectures=_conjectures(2))

        assert all(r.final_status is PipelineStatus.DISPROVED for r in results)

    @pytest.mark.asyncio()
    async def test_pipeline_handles_unformalizable(self) -> None:
        """Formalizer returns UNFORMALIZABLE → PipelineResult(UNFORMALIZABLE)."""
        with tempfile.TemporaryDirectory() as tmp:
            formalizer = MagicMock()
            formalizer.formalize = AsyncMock(
                side_effect=lambda c: FormalizedConjecture(
                    conjecture=c,
                    lean_code="",
                    status=FormalizationStatus.UNFORMALIZABLE,
                    error_history=["unknown identifier 'foo'"],
                )
            )

            pipeline = _make_pipeline(tmp, formalizer=formalizer, n_conjectures=2)
            results = await pipeline.run(conjectures=_conjectures(2))

        assert all(r.final_status is PipelineStatus.UNFORMALIZABLE for r in results)

    @pytest.mark.asyncio()
    async def test_pipeline_handles_open_proof(self) -> None:
        """Proof search gives OPEN → PipelineResult(OPEN)."""
        with tempfile.TemporaryDirectory() as tmp:
            proof_searcher = MagicMock()

            async def _open(fc: FormalizedConjecture) -> ProofResult:
                return ProofResult(
                    formalized=fc,
                    status=ProofStatus.OPEN,
                    duration_seconds=0.5,
                )

            proof_searcher.search = AsyncMock(side_effect=_open)
            pipeline = _make_pipeline(tmp, proof_searcher=proof_searcher, n_conjectures=2)
            results = await pipeline.run(conjectures=_conjectures(2))

        assert all(r.final_status is PipelineStatus.OPEN for r in results)

    @pytest.mark.asyncio()
    async def test_pipeline_skip_generation(self) -> None:
        """Passing conjectures= directly skips generation stage."""
        with tempfile.TemporaryDirectory() as tmp:
            generator = MagicMock()
            generator.generate = AsyncMock()

            pipeline = _make_pipeline(tmp, generator=generator, n_conjectures=3)
            await pipeline.run(conjectures=_conjectures(3))

        generator.generate.assert_not_called()

    @pytest.mark.asyncio()
    async def test_pipeline_raises_on_both_conjectures_and_domains(self) -> None:
        """Providing both conjectures= and domains= raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _make_pipeline(tmp, n_conjectures=2)
            with pytest.raises(ValueError):
                await pipeline.run(
                    conjectures=_conjectures(2),
                    domains=[Domain.NUMBER_THEORY],
                )


# ---------------------------------------------------------------------------
# TestPipelinePersistence
# ---------------------------------------------------------------------------


class TestPipelinePersistence:
    """Tests for pipeline result saving and loading."""

    @pytest.mark.asyncio()
    async def test_pipeline_saves_results_jsonl(self) -> None:
        """Results saved to JSONL can be read back as valid JSON objects."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _make_pipeline(tmp, n_conjectures=3)
            await pipeline.run(conjectures=_conjectures(3))

            results_path = Path(tmp) / "results.jsonl"
            assert results_path.is_file()

            lines = results_path.read_text().strip().split("\n")
            assert len(lines) == 3
            for line in lines:
                record = json.loads(line)
                assert "conjecture" in record
                assert "final_status" in record

    @pytest.mark.asyncio()
    async def test_pipeline_result_has_correct_final_status_string(self) -> None:
        """JSONL records store final_status as a plain string (not an Enum wrapper)."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _make_pipeline(tmp, n_conjectures=2)
            await pipeline.run(conjectures=_conjectures(2))

            results_path = Path(tmp) / "results.jsonl"
            lines = results_path.read_text().strip().split("\n")
            for line in lines:
                record = json.loads(line)
                assert isinstance(record["final_status"], str)


# ---------------------------------------------------------------------------
# TestAssemblePipelineResults
# ---------------------------------------------------------------------------


class TestAssemblePipelineResults:
    """Unit tests for _assemble_pipeline_results and _derive_final_status."""

    def _conj(self, cid: str) -> Conjecture:
        return Conjecture(id=cid, domain=Domain.NUMBER_THEORY, nl_statement="test", variables=[])

    def test_derive_status_proved(self) -> None:
        """_derive_final_status returns PROVED when proof is ProofStatus.PROVED."""
        c = self._conj("c1")
        fr = FilterResult(conjecture=c, status=FilterStatus.SURVIVING)
        fc = FormalizedConjecture(conjecture=c, lean_code="...", status=FormalizationStatus.TYPECHECKS)
        pr = ProofResult(formalized=fc, status=ProofStatus.PROVED)
        assert _derive_final_status(fr, fc, pr) is PipelineStatus.PROVED

    def test_derive_status_open(self) -> None:
        """_derive_final_status returns OPEN when proof is ProofStatus.OPEN."""
        c = self._conj("c2")
        fr = FilterResult(conjecture=c, status=FilterStatus.SURVIVING)
        fc = FormalizedConjecture(conjecture=c, lean_code="...", status=FormalizationStatus.TYPECHECKS)
        pr = ProofResult(formalized=fc, status=ProofStatus.OPEN)
        assert _derive_final_status(fr, fc, pr) is PipelineStatus.OPEN

    def test_derive_status_disproved(self) -> None:
        """_derive_final_status returns DISPROVED when filter status is DISPROVED."""
        c = self._conj("c3")
        fr = FilterResult(conjecture=c, status=FilterStatus.DISPROVED)
        assert _derive_final_status(fr, None, None) is PipelineStatus.DISPROVED

    def test_derive_status_unformalizable(self) -> None:
        """_derive_final_status returns UNFORMALIZABLE when formalization fails."""
        c = self._conj("c4")
        fr = FilterResult(conjecture=c, status=FilterStatus.SURVIVING)
        fc = FormalizedConjecture(conjecture=c, lean_code="", status=FormalizationStatus.UNFORMALIZABLE)
        assert _derive_final_status(fr, fc, None) is PipelineStatus.UNFORMALIZABLE

    def test_derive_status_filtered_out_no_filter(self) -> None:
        """_derive_final_status returns FILTERED_OUT when filter result is None."""
        assert _derive_final_status(None, None, None) is PipelineStatus.FILTERED_OUT

    def test_derive_status_filtered_out_trivial(self) -> None:
        """_derive_final_status returns FILTERED_OUT when filter status is TRIVIAL."""
        c = self._conj("c5")
        fr = FilterResult(conjecture=c, status=FilterStatus.TRIVIAL)
        assert _derive_final_status(fr, None, None) is PipelineStatus.FILTERED_OUT

    def test_assemble_all_proved(self) -> None:
        """_assemble_pipeline_results produces PROVED for each proved conjecture."""
        conjs = _conjectures(2)
        filter_results = [
            FilterResult(conjecture=c, status=FilterStatus.SURVIVING) for c in conjs
        ]
        formalized_list = [
            FormalizedConjecture(conjecture=c, lean_code="...", status=FormalizationStatus.TYPECHECKS)
            for c in conjs
        ]
        proof_results = [
            ProofResult(
                formalized=fc,
                status=ProofStatus.PROVED,
                layer=ProofLayer.LAYER0_AUTO,
            )
            for fc in formalized_list
        ]
        pipeline_results = _assemble_pipeline_results(
            all_conjectures=conjs,
            filter_results=filter_results,
            formalized_list=formalized_list,
            proof_results=proof_results,
        )
        assert len(pipeline_results) == 2
        assert all(pr.final_status is PipelineStatus.PROVED for pr in pipeline_results)

    def test_assemble_id_ordering(self) -> None:
        """_assemble_pipeline_results preserves input ordering by conjecture ID."""
        conjs = _conjectures(3)
        filter_results = [
            FilterResult(conjecture=c, status=FilterStatus.SURVIVING) for c in conjs
        ]
        formalized_list = [
            FormalizedConjecture(conjecture=c, lean_code="...", status=FormalizationStatus.TYPECHECKS)
            for c in conjs
        ]
        proof_results = [
            ProofResult(formalized=fc, status=ProofStatus.OPEN)
            for fc in formalized_list
        ]
        pipeline_results = _assemble_pipeline_results(
            all_conjectures=conjs,
            filter_results=filter_results,
            formalized_list=formalized_list,
            proof_results=proof_results,
        )
        for i, pr in enumerate(pipeline_results):
            assert pr.conjecture.id == conjs[i].id

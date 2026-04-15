"""
Tests for conjlean.schemas — enums, dataclasses, and field defaults.

Validates that every public type in the schemas module instantiates correctly,
carries the expected string values for enum members, and that default_factory
fields produce isolated containers per instance.
"""

from __future__ import annotations

import pytest

from conjlean.schemas import (
    Conjecture,
    Domain,
    FilterResult,
    FilterStatus,
    FormalizedConjecture,
    FormalizationStatus,
    LeanCheckResult,
    PipelineResult,
    PipelineStatus,
    ProofAttempt,
    ProofLayer,
    ProofResult,
    ProofStatus,
)


# ---------------------------------------------------------------------------
# Enum value tests
# ---------------------------------------------------------------------------


class TestDomainEnum:
    """Validate Domain enum string values and membership."""

    def test_number_theory_value(self) -> None:
        """Domain.NUMBER_THEORY must have the string value 'number_theory'."""
        assert Domain.NUMBER_THEORY.value == "number_theory"

    def test_inequality_value(self) -> None:
        """Domain.INEQUALITY must have the string value 'inequality'."""
        assert Domain.INEQUALITY.value == "inequality"

    def test_combinatorics_value(self) -> None:
        """Domain.COMBINATORICS must have the string value 'combinatorics'."""
        assert Domain.COMBINATORICS.value == "combinatorics"

    def test_domain_is_str(self) -> None:
        """Domain subclasses str so it can be used directly as a string key."""
        assert isinstance(Domain.NUMBER_THEORY, str)
        assert Domain.NUMBER_THEORY == "number_theory"

    def test_domain_from_string(self) -> None:
        """Domain can be constructed from its string value."""
        assert Domain("number_theory") is Domain.NUMBER_THEORY
        assert Domain("inequality") is Domain.INEQUALITY
        assert Domain("combinatorics") is Domain.COMBINATORICS

    def test_all_three_members_exist(self) -> None:
        """Exactly three domain members are defined."""
        assert len(Domain) == 3


class TestFilterStatusEnum:
    """Validate FilterStatus enum values."""

    def test_surviving_value(self) -> None:
        """FilterStatus.SURVIVING must have value 'surviving'."""
        assert FilterStatus.SURVIVING.value == "surviving"

    def test_disproved_value(self) -> None:
        """FilterStatus.DISPROVED must have value 'disproved'."""
        assert FilterStatus.DISPROVED.value == "disproved"

    def test_trivial_value(self) -> None:
        """FilterStatus.TRIVIAL must have value 'trivial'."""
        assert FilterStatus.TRIVIAL.value == "trivial"


class TestFormalizationStatusEnum:
    """Validate FormalizationStatus enum values."""

    def test_typechecks_value(self) -> None:
        """FormalizationStatus.TYPECHECKS must have value 'typechecks'."""
        assert FormalizationStatus.TYPECHECKS.value == "typechecks"

    def test_unformalizable_value(self) -> None:
        """FormalizationStatus.UNFORMALIZABLE must have value 'unformalizable'."""
        assert FormalizationStatus.UNFORMALIZABLE.value == "unformalizable"


class TestProofStatusEnum:
    """Validate ProofStatus enum values."""

    def test_proved_value(self) -> None:
        """ProofStatus.PROVED must have value 'proved'."""
        assert ProofStatus.PROVED.value == "proved"

    def test_open_value(self) -> None:
        """ProofStatus.OPEN must have value 'open'."""
        assert ProofStatus.OPEN.value == "open"


class TestProofLayerEnum:
    """Validate ProofLayer enum — all four values must be present."""

    def test_layer0_value(self) -> None:
        """ProofLayer.LAYER0_AUTO must have value 'layer0_auto'."""
        assert ProofLayer.LAYER0_AUTO.value == "layer0_auto"

    def test_layer1_value(self) -> None:
        """ProofLayer.LAYER1_COMBO must have value 'layer1_combo'."""
        assert ProofLayer.LAYER1_COMBO.value == "layer1_combo"

    def test_layer2_value(self) -> None:
        """ProofLayer.LAYER2_SEARCH must have value 'layer2_search'."""
        assert ProofLayer.LAYER2_SEARCH.value == "layer2_search"

    def test_layer3_value(self) -> None:
        """ProofLayer.LAYER3_LLM must have value 'layer3_llm'."""
        assert ProofLayer.LAYER3_LLM.value == "layer3_llm"

    def test_exactly_four_layers(self) -> None:
        """Exactly four ProofLayer members are defined."""
        assert len(ProofLayer) == 4


class TestPipelineStatusEnum:
    """Validate PipelineStatus — all five values must be present."""

    def test_proved_value(self) -> None:
        """PipelineStatus.PROVED must have value 'proved'."""
        assert PipelineStatus.PROVED.value == "proved"

    def test_open_value(self) -> None:
        """PipelineStatus.OPEN must have value 'open'."""
        assert PipelineStatus.OPEN.value == "open"

    def test_disproved_value(self) -> None:
        """PipelineStatus.DISPROVED must have value 'disproved'."""
        assert PipelineStatus.DISPROVED.value == "disproved"

    def test_unformalizable_value(self) -> None:
        """PipelineStatus.UNFORMALIZABLE must have value 'unformalizable'."""
        assert PipelineStatus.UNFORMALIZABLE.value == "unformalizable"

    def test_filtered_out_value(self) -> None:
        """PipelineStatus.FILTERED_OUT must have value 'filtered_out'."""
        assert PipelineStatus.FILTERED_OUT.value == "filtered_out"

    def test_exactly_five_values(self) -> None:
        """Exactly five PipelineStatus members are defined."""
        assert len(PipelineStatus) == 5


# ---------------------------------------------------------------------------
# Dataclass instantiation tests
# ---------------------------------------------------------------------------


class TestConjectureDataclass:
    """Validate Conjecture dataclass construction and field defaults."""

    def test_required_fields_only(self) -> None:
        """Conjecture instantiates correctly with only required fields."""
        c = Conjecture(
            id="c1",
            domain=Domain.NUMBER_THEORY,
            nl_statement="n*(n+1) is even",
            variables=["n"],
        )
        assert c.id == "c1"
        assert c.domain is Domain.NUMBER_THEORY
        assert c.nl_statement == "n*(n+1) is even"
        assert c.variables == ["n"]

    def test_default_source(self) -> None:
        """Conjecture.source defaults to 'generated'."""
        c = Conjecture(
            id="c2",
            domain=Domain.INEQUALITY,
            nl_statement="a >= 0",
            variables=["a"],
        )
        assert c.source == "generated"

    def test_default_timestamp(self) -> None:
        """Conjecture.timestamp defaults to empty string."""
        c = Conjecture(
            id="c3",
            domain=Domain.COMBINATORICS,
            nl_statement="C(n,2) >= 0",
            variables=["n"],
        )
        assert c.timestamp == ""

    def test_metadata_default_is_empty_dict(self) -> None:
        """Conjecture.metadata defaults to an empty dict."""
        c = Conjecture(
            id="c4",
            domain=Domain.NUMBER_THEORY,
            nl_statement="foo",
            variables=[],
        )
        assert c.metadata == {}

    def test_metadata_isolation(self) -> None:
        """Two Conjecture instances must not share the same metadata dict."""
        c1 = Conjecture(id="c5", domain=Domain.NUMBER_THEORY, nl_statement="foo", variables=[])
        c2 = Conjecture(id="c6", domain=Domain.NUMBER_THEORY, nl_statement="bar", variables=[])
        c1.metadata["key"] = "value"
        assert "key" not in c2.metadata

    def test_optional_fields_set(self) -> None:
        """Conjecture accepts all optional fields."""
        c = Conjecture(
            id="c7",
            domain=Domain.NUMBER_THEORY,
            nl_statement="test",
            variables=["n", "m"],
            source="curated",
            timestamp="2026-04-15T00:00:00Z",
            metadata={"model": "claude-3"},
        )
        assert c.source == "curated"
        assert c.timestamp == "2026-04-15T00:00:00Z"
        assert c.metadata["model"] == "claude-3"


class TestFilterResultDataclass:
    """Validate FilterResult dataclass."""

    def test_required_fields(self) -> None:
        """FilterResult instantiates with conjecture and status."""
        c = Conjecture(id="f1", domain=Domain.NUMBER_THEORY, nl_statement="test", variables=[])
        fr = FilterResult(conjecture=c, status=FilterStatus.SURVIVING)
        assert fr.conjecture is c
        assert fr.status is FilterStatus.SURVIVING
        assert fr.counterexample is None

    def test_numerical_evidence_isolation(self) -> None:
        """Two FilterResult instances must not share the same evidence dict."""
        c = Conjecture(id="f2", domain=Domain.NUMBER_THEORY, nl_statement="test", variables=[])
        fr1 = FilterResult(conjecture=c, status=FilterStatus.SURVIVING)
        fr2 = FilterResult(conjecture=c, status=FilterStatus.SURVIVING)
        fr1.numerical_evidence["n=1"] = True
        assert "n=1" not in fr2.numerical_evidence

    def test_counterexample_set(self) -> None:
        """FilterResult counterexample field is stored correctly."""
        c = Conjecture(id="f3", domain=Domain.NUMBER_THEORY, nl_statement="test", variables=[])
        fr = FilterResult(
            conjecture=c,
            status=FilterStatus.DISPROVED,
            counterexample="n=5: 5^2+1=26, 7 does not divide 26",
        )
        assert fr.counterexample == "n=5: 5^2+1=26, 7 does not divide 26"


class TestLeanCheckResultDataclass:
    """Validate LeanCheckResult dataclass."""

    def test_default_env_id(self) -> None:
        """LeanCheckResult.env_id defaults to 0."""
        r = LeanCheckResult(success=True, messages=[])
        assert r.env_id == 0

    def test_messages_list_isolation(self) -> None:
        """Two LeanCheckResult instances do not share the same messages list."""
        r1 = LeanCheckResult(success=True, messages=[{"severity": "info", "data": "ok"}])
        r2 = LeanCheckResult(success=False, messages=[])
        assert r1.messages != r2.messages

    def test_explicit_env_id(self) -> None:
        """LeanCheckResult accepts an explicit env_id."""
        r = LeanCheckResult(success=True, messages=[], env_id=42)
        assert r.env_id == 42


class TestFormalizedConjectureDataclass:
    """Validate FormalizedConjecture dataclass."""

    def test_required_fields(self) -> None:
        """FormalizedConjecture instantiates with required fields."""
        c = Conjecture(id="fc1", domain=Domain.NUMBER_THEORY, nl_statement="test", variables=[])
        fc = FormalizedConjecture(
            conjecture=c,
            lean_code="theorem foo : True := by sorry",
            status=FormalizationStatus.TYPECHECKS,
        )
        assert fc.retries == 0
        assert fc.error_history == []

    def test_error_history_isolation(self) -> None:
        """Two FormalizedConjecture instances must not share error_history lists."""
        c = Conjecture(id="fc2", domain=Domain.NUMBER_THEORY, nl_statement="test", variables=[])
        fc1 = FormalizedConjecture(conjecture=c, lean_code="...", status=FormalizationStatus.TYPECHECKS)
        fc2 = FormalizedConjecture(conjecture=c, lean_code="...", status=FormalizationStatus.TYPECHECKS)
        fc1.error_history.append("some error")
        assert len(fc2.error_history) == 0


class TestProofAttemptDataclass:
    """Validate ProofAttempt dataclass."""

    def test_required_fields(self) -> None:
        """ProofAttempt instantiates with tactic and success fields."""
        pa = ProofAttempt(tactic="omega", success=True)
        assert pa.tactic == "omega"
        assert pa.success is True
        assert pa.error is None
        assert pa.layer is None

    def test_optional_fields(self) -> None:
        """ProofAttempt accepts optional layer and error fields."""
        pa = ProofAttempt(
            tactic="simp",
            success=False,
            error="unknown identifier",
            layer=ProofLayer.LAYER0_AUTO,
        )
        assert pa.layer is ProofLayer.LAYER0_AUTO
        assert pa.error == "unknown identifier"


class TestProofResultDataclass:
    """Validate ProofResult dataclass."""

    def test_attempts_isolation(self) -> None:
        """Two ProofResult instances must not share the same attempts list."""
        c = Conjecture(id="pr1", domain=Domain.NUMBER_THEORY, nl_statement="test", variables=[])
        fc = FormalizedConjecture(conjecture=c, lean_code="...", status=FormalizationStatus.TYPECHECKS)
        pr1 = ProofResult(formalized=fc, status=ProofStatus.OPEN)
        pr2 = ProofResult(formalized=fc, status=ProofStatus.OPEN)
        pr1.attempts.append(ProofAttempt(tactic="omega", success=False))
        assert len(pr2.attempts) == 0

    def test_default_duration(self) -> None:
        """ProofResult.duration_seconds defaults to 0.0."""
        c = Conjecture(id="pr2", domain=Domain.NUMBER_THEORY, nl_statement="test", variables=[])
        fc = FormalizedConjecture(conjecture=c, lean_code="...", status=FormalizationStatus.TYPECHECKS)
        pr = ProofResult(formalized=fc, status=ProofStatus.OPEN)
        assert pr.duration_seconds == 0.0


class TestPipelineResultDataclass:
    """Validate PipelineResult dataclass."""

    def test_default_status(self) -> None:
        """PipelineResult.final_status defaults to FILTERED_OUT."""
        c = Conjecture(id="plr1", domain=Domain.NUMBER_THEORY, nl_statement="test", variables=[])
        plr = PipelineResult(conjecture=c)
        assert plr.final_status is PipelineStatus.FILTERED_OUT

    def test_all_optional_fields_none(self) -> None:
        """PipelineResult optional stage fields default to None."""
        c = Conjecture(id="plr2", domain=Domain.NUMBER_THEORY, nl_statement="test", variables=[])
        plr = PipelineResult(conjecture=c)
        assert plr.filter_result is None
        assert plr.formalization is None
        assert plr.proof is None

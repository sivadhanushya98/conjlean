"""
Tests for conjlean.lean_harness — LeanHarness subprocess management and
response parsing helpers.

No real Lean subprocess is spawned. All subprocess.Popen calls are mocked.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from conjlean.lean_harness import (
    LeanHarness,
    LeanREPLNotFoundError,
    _parse_response,
    _result_is_success,
    _strip_ansi,
    _has_sorry_warning,
)
from conjlean.schemas import LeanCheckResult


# ---------------------------------------------------------------------------
# Helper: build a mock subprocess.Popen for the harness
# ---------------------------------------------------------------------------


def _make_mock_process(response_json: str | None = None) -> MagicMock:
    """
    Build a mock subprocess.Popen instance.

    Args:
        response_json: The JSON line the fake REPL stdout will return.
                       Defaults to a successful empty-message response.
    """
    if response_json is None:
        response_json = json.dumps({"env": 1, "messages": []})

    proc = MagicMock()
    proc.poll = MagicMock(return_value=None)  # process is running
    proc.pid = 12345
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.flush = MagicMock()
    proc.stdin.close = MagicMock()
    proc.stdout = MagicMock()
    proc.stdout.readline = MagicMock(return_value=response_json + "\n")
    proc.stderr = MagicMock()
    proc.terminate = MagicMock()
    proc.wait = MagicMock()
    proc.kill = MagicMock()
    return proc


def _make_harness_with_project_dir() -> tuple[LeanHarness, str]:
    """Create a LeanHarness pointed at a real temp directory."""
    tmpdir = tempfile.mkdtemp()
    harness = LeanHarness(lean_project_dir=tmpdir, repl_timeout=5)
    return harness, tmpdir


# ---------------------------------------------------------------------------
# Module-level helper function tests
# ---------------------------------------------------------------------------


class TestParseResponse:
    """Tests for the _parse_response helper."""

    def test_parses_valid_json(self) -> None:
        """Valid JSON is parsed into a dict."""
        raw = '{"env": 2, "messages": []}\n'
        result = _parse_response(raw)
        assert result["env"] == 2
        assert result["messages"] == []

    def test_raises_on_invalid_json(self) -> None:
        """Non-JSON input raises ValueError."""
        with pytest.raises(ValueError, match="non-JSON"):
            _parse_response("not json")

    def test_strips_trailing_whitespace(self) -> None:
        """Trailing whitespace/newlines are stripped before parsing."""
        raw = '{"env": 0, "messages": []}   \n  '
        result = _parse_response(raw)
        assert result["env"] == 0


class TestResultIsSuccess:
    """Tests for _result_is_success."""

    def test_empty_messages_is_success(self) -> None:
        """No messages means success."""
        assert _result_is_success([]) is True

    def test_info_only_is_success(self) -> None:
        """Info-only messages are still considered success."""
        messages = [{"severity": "info", "data": "loaded Mathlib"}]
        assert _result_is_success(messages) is True

    def test_error_message_is_not_success(self) -> None:
        """A single error-severity message means failure."""
        messages = [{"severity": "error", "data": "unknown identifier 'foo'"}]
        assert _result_is_success(messages) is False

    def test_mixed_messages_with_error_is_not_success(self) -> None:
        """Even one error in a mixed list means failure."""
        messages = [
            {"severity": "info", "data": "ok"},
            {"severity": "error", "data": "type mismatch"},
        ]
        assert _result_is_success(messages) is False


class TestStripAnsi:
    """Tests for _strip_ansi."""

    def test_strips_colour_codes(self) -> None:
        """Standard ANSI colour codes are removed."""
        text = "\x1B[31mError\x1B[0m: something failed"
        result = _strip_ansi(text)
        assert result == "Error: something failed"

    def test_no_codes_unchanged(self) -> None:
        """Plain text without ANSI codes is returned unchanged."""
        text = "plain text message"
        result = _strip_ansi(text)
        assert result == text

    def test_multiple_codes_stripped(self) -> None:
        """Multiple embedded codes are all removed."""
        text = "\x1B[1m\x1B[31mBold Red\x1B[0m Normal"
        result = _strip_ansi(text)
        assert "\x1B" not in result
        assert "Bold Red" in result


class TestHasSorryWarning:
    """Tests for _has_sorry_warning."""

    def test_detects_sorry_in_message(self) -> None:
        """Returns True when a message contains 'sorry'."""
        messages = [{"severity": "warning", "data": "declaration uses sorry"}]
        assert _has_sorry_warning(messages) is True

    def test_no_sorry_returns_false(self) -> None:
        """Returns False when no message mentions sorry."""
        messages = [{"severity": "info", "data": "all goals closed"}]
        assert _has_sorry_warning(messages) is False

    def test_case_insensitive(self) -> None:
        """Detection is case-insensitive."""
        messages = [{"severity": "warning", "data": "Declaration uses Sorry"}]
        assert _has_sorry_warning(messages) is True


# ---------------------------------------------------------------------------
# LeanHarness lifecycle tests
# ---------------------------------------------------------------------------


class TestHarnessLifecycle:
    """Tests for LeanHarness.start, stop, and is_running."""

    def test_not_started_is_not_running(self) -> None:
        """A fresh harness is not running before start() is called."""
        harness, _ = _make_harness_with_project_dir()
        assert harness.is_running is False

    def test_invalid_project_dir_raises(self) -> None:
        """FileNotFoundError is raised for a non-existent project directory."""
        with pytest.raises(FileNotFoundError):
            LeanHarness(lean_project_dir="/no/such/directory")

    def test_check_statement_before_start_raises(self) -> None:
        """check_statement before start() raises RuntimeError."""
        harness, _ = _make_harness_with_project_dir()
        with pytest.raises(RuntimeError, match="not running"):
            harness.check_statement("theorem foo : True := by sorry")

    def test_try_proof_before_start_raises(self) -> None:
        """try_proof before start() raises RuntimeError."""
        harness, _ = _make_harness_with_project_dir()
        with pytest.raises(RuntimeError, match="not running"):
            harness.try_proof("theorem foo : True := by sorry", "trivial")

    def test_verify_full_proof_before_start_raises(self) -> None:
        """verify_full_proof before start() raises RuntimeError."""
        harness, _ = _make_harness_with_project_dir()
        with pytest.raises(RuntimeError, match="not running"):
            harness.verify_full_proof("theorem foo : True := by trivial")

    def test_repl_not_found_raises_lean_error(self) -> None:
        """When 'lake exe repl' is not found, LeanREPLNotFoundError is raised."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = LeanHarness(lean_project_dir=tmpdir, repl_timeout=2)

            with patch("subprocess.Popen", side_effect=FileNotFoundError("lake: not found")):
                with pytest.raises(LeanREPLNotFoundError):
                    harness.start()

    def test_lean_repl_not_found_error_has_helpful_message(self) -> None:
        """LeanREPLNotFoundError message mentions lake and the REPL."""
        err = LeanREPLNotFoundError("test detail")
        msg = str(err)
        assert "lake" in msg.lower() or "repl" in msg.lower()


# ---------------------------------------------------------------------------
# LeanHarness send_command tests (with mocked subprocess)
# ---------------------------------------------------------------------------


class TestSendCommand:
    """Tests for LeanHarness.send_command with mocked subprocess."""

    def _start_harness_with_mock(
        self,
        response_json: str | None = None,
    ) -> tuple[LeanHarness, MagicMock]:
        """Start a harness whose subprocess is fully mocked."""
        # Mathlib import response (env_id=1, no errors)
        mathlib_response = json.dumps({"env": 1, "messages": []}) + "\n"
        custom_response = (response_json + "\n") if response_json else (json.dumps({"env": 2, "messages": []}) + "\n")

        call_count = [0]

        def _readline() -> str:
            call_count[0] += 1
            # First call is the Mathlib import warmup
            if call_count[0] == 1:
                return mathlib_response
            return custom_response

        proc = _make_mock_process()
        proc.stdout.readline = MagicMock(side_effect=_readline)

        tmpdir = tempfile.mkdtemp()
        harness = LeanHarness(lean_project_dir=tmpdir, repl_timeout=5)

        with patch("subprocess.Popen", return_value=proc):
            harness.start()

        harness._process = proc
        harness._mathlib_env = 1
        return harness, proc

    def test_parse_success_response(self) -> None:
        """JSON with no error messages → LeanCheckResult(success=True)."""
        harness, proc = self._start_harness_with_mock(
            json.dumps({"env": 2, "messages": []})
        )
        result = harness.send_command("theorem foo : True := by sorry", env=1)
        assert result.success is True
        assert result.env_id == 2
        assert result.messages == []

    def test_parse_error_response(self) -> None:
        """JSON with severity='error' → LeanCheckResult(success=False)."""
        error_resp = json.dumps({
            "env": 0,
            "messages": [{"severity": "error", "pos": {"line": 1, "column": 0}, "data": "unknown identifier 'Foo'"}],
        })
        harness, _ = self._start_harness_with_mock(error_resp)
        result = harness.send_command("theorem bad : Foo := by sorry", env=1)
        assert result.success is False
        assert len(result.messages) == 1
        assert result.messages[0]["severity"] == "error"

    def test_parse_strips_ansi(self) -> None:
        """ANSI codes in message data are stripped before returning."""
        ansi_resp = json.dumps({
            "env": 0,
            "messages": [{"severity": "error", "data": "\x1B[31mError\x1B[0m: bad identifier"}],
        })
        harness, _ = self._start_harness_with_mock(ansi_resp)
        result = harness.send_command("...", env=1)
        for msg in result.messages:
            assert "\x1B" not in msg.get("data", ""), "ANSI codes were not stripped"


# ---------------------------------------------------------------------------
# LeanHarness high-level method tests
# ---------------------------------------------------------------------------


class TestCheckStatement:
    """Tests for LeanHarness.check_statement."""

    def _running_harness(self, response_json: str | None = None) -> LeanHarness:
        """Build a started harness with a mocked process."""
        tmpdir = tempfile.mkdtemp()
        harness = LeanHarness(lean_project_dir=tmpdir, repl_timeout=5)
        harness._process = _make_mock_process(response_json)
        harness._mathlib_env = 1
        return harness

    def test_check_statement_adds_sorry(self) -> None:
        """check_statement appends ':= by sorry' when not present."""
        resp = json.dumps({"env": 2, "messages": []})
        harness = self._running_harness(resp)

        calls: list[str] = []
        original_send = harness.send_command

        def _spy(cmd: str, **kwargs: Any) -> LeanCheckResult:
            calls.append(cmd)
            return original_send(cmd, **kwargs)

        harness.send_command = _spy  # type: ignore[method-assign]
        harness.check_statement("theorem foo (n : ℕ) : n = n")

        assert len(calls) == 1
        assert "sorry" in calls[0]

    def test_check_statement_does_not_double_sorry(self) -> None:
        """check_statement does NOT append sorry when code already has 'by sorry'."""
        resp = json.dumps({"env": 2, "messages": []})
        harness = self._running_harness(resp)

        calls: list[str] = []
        original_send = harness.send_command

        def _spy(cmd: str, **kwargs: Any) -> LeanCheckResult:
            calls.append(cmd)
            return original_send(cmd, **kwargs)

        harness.send_command = _spy  # type: ignore[method-assign]
        code = "theorem foo : True := by sorry"
        harness.check_statement(code)

        assert calls[0].count("sorry") == 1


class TestTryProof:
    """Tests for LeanHarness.try_proof."""

    def _running_harness(self, response_json: str | None = None) -> LeanHarness:
        tmpdir = tempfile.mkdtemp()
        harness = LeanHarness(lean_project_dir=tmpdir, repl_timeout=5)
        harness._process = _make_mock_process(response_json)
        harness._mathlib_env = 1
        return harness

    def test_try_proof_replaces_sorry(self) -> None:
        """try_proof replaces sorry in the statement code with the tactic body."""
        resp = json.dumps({"env": 2, "messages": []})
        harness = self._running_harness(resp)

        calls: list[str] = []
        original_send = harness.send_command

        def _spy(cmd: str, **kwargs: Any) -> LeanCheckResult:
            calls.append(cmd)
            return original_send(cmd, **kwargs)

        harness.send_command = _spy  # type: ignore[method-assign]

        statement = "theorem foo : True := by\n  sorry"
        harness.try_proof(statement, "trivial")

        assert len(calls) == 1
        assert "sorry" not in calls[0] or "trivial" in calls[0]


class TestVerifyFullProof:
    """Tests for LeanHarness.verify_full_proof."""

    def _running_harness(self, response_json: str | None = None) -> LeanHarness:
        tmpdir = tempfile.mkdtemp()
        harness = LeanHarness(lean_project_dir=tmpdir, repl_timeout=5)
        harness._process = _make_mock_process(response_json)
        harness._mathlib_env = 1
        return harness

    def test_verify_success_when_no_sorry_warning(self) -> None:
        """verify_full_proof returns success when no sorry warning and no error."""
        resp = json.dumps({"env": 2, "messages": [{"severity": "info", "data": "done"}]})
        harness = self._running_harness(resp)
        result = harness.verify_full_proof("theorem foo : True := by trivial")
        assert result.success is True

    def test_verify_fails_when_sorry_warning_present(self) -> None:
        """verify_full_proof returns success=False when messages contain 'sorry'."""
        resp = json.dumps({
            "env": 2,
            "messages": [{"severity": "warning", "data": "declaration uses sorry"}],
        })
        harness = self._running_harness(resp)
        result = harness.verify_full_proof("theorem foo : True := by sorry")
        assert result.success is False


# ---------------------------------------------------------------------------
# Timeout test
# ---------------------------------------------------------------------------


class TestTimeout:
    """Tests for REPL command timeout behaviour."""

    def test_timeout_raises_timeout_error(self) -> None:
        """A hanging REPL triggers TimeoutError after the configured timeout."""
        import threading
        import time

        tmpdir = tempfile.mkdtemp()
        harness = LeanHarness(lean_project_dir=tmpdir, repl_timeout=1)

        proc = _make_mock_process()

        # readline blocks forever to simulate a hanging REPL
        def _hang() -> str:
            time.sleep(10)
            return ""

        proc.stdout.readline = MagicMock(side_effect=_hang)
        harness._process = proc
        harness._mathlib_env = 1

        with pytest.raises(TimeoutError):
            harness.send_command("import Mathlib", env=0, timeout=1)

"""
Python wrapper around the LeanDojo REPL subprocess for Lean 4 proof checking.

The REPL communicates over stdin/stdout using a newline-delimited JSON protocol.
Each command returns a new environment ID that can be chained to build on prior
imports.  ``import Mathlib`` is performed once at startup and its environment ID
is cached so all subsequent theorem checks and proof attempts share the full
Mathlib namespace without re-importing.

Protocol reference:
    Send:    {"cmd": "<lean_code>", "env": <env_id>}\\n
    Receive: {"env": <new_env_id>, "messages": [{"severity": "...", "data": "..."}]}\\n

Environment IDs:
    env=0  Empty environment (no imports)
    env=N  State after the N-th successful command (chained imports / defs)
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
from pathlib import Path
from typing import Optional

from conjlean.schemas import LeanCheckResult

logger = logging.getLogger(__name__)

# Regex that matches ANSI CSI escape sequences (colour, cursor, etc.)
_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]|\x1B[@-_][0-9;]*")


class LeanREPLNotFoundError(RuntimeError):
    """
    Raised when ``lake exe repl`` cannot be found or fails to start.

    Provides actionable installation guidance so the user can resolve the
    issue without reading source code.
    """

    def __init__(self, detail: str = "") -> None:
        msg = (
            "The LeanDojo REPL executable was not found via 'lake exe repl'.\n"
            "To install it, add the following dependency to your lakefile.toml:\n\n"
            "  [[require]]\n"
            "  name = \"repl\"\n"
            "  from = \"git\"\n"
            "  url = \"https://github.com/leanprover-community/repl\"\n"
            "  rev = \"main\"\n\n"
            "Then run: lake build repl\n\n"
            f"Original error: {detail}" if detail else
            "The LeanDojo REPL executable was not found via 'lake exe repl'.\n"
            "To install it, add the REPL dependency to lakefile.toml and run: lake build repl"
        )
        super().__init__(msg)


def _strip_ansi(text: str) -> str:
    """
    Remove all ANSI escape sequences from a string.

    Args:
        text: Raw string potentially containing ANSI colour/cursor codes.

    Returns:
        Clean string with all escape sequences removed.
    """
    return _ANSI_ESCAPE_RE.sub("", text)


def _parse_response(raw: str) -> dict:
    """
    Parse a single JSON line from the REPL stdout.

    Args:
        raw: Raw newline-terminated JSON string from the REPL.

    Returns:
        Parsed response dict with at minimum ``"env"`` and ``"messages"`` keys.

    Raises:
        ValueError: If ``raw`` is not valid JSON.
    """
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"REPL returned non-JSON output: {raw!r}") from exc


def _result_is_success(messages: list[dict]) -> bool:
    """
    Determine whether a REPL response represents a successful command.

    A command is considered successful when none of the returned messages
    carry severity ``"error"``.

    Args:
        messages: The ``"messages"`` list from a parsed REPL response.

    Returns:
        True if no error-severity messages are present.
    """
    return all(m.get("severity") != "error" for m in messages)


def _has_sorry_warning(messages: list[dict]) -> bool:
    """
    Check whether the REPL flagged an unsolved ``sorry`` placeholder.

    Args:
        messages: The ``"messages"`` list from a parsed REPL response.

    Returns:
        True if any message contains the word ``"sorry"`` in its data.
    """
    return any("sorry" in m.get("data", "").lower() for m in messages)


class LeanHarness:
    """
    Manages a persistent LeanDojo REPL subprocess for interactive proof checking.

    The REPL is started via ``lake exe repl`` inside the Lean project directory
    and communicates over stdin/stdout using newline-delimited JSON.

    ``import Mathlib`` is issued once at construction time (inside ``start``),
    and the resulting environment ID is cached as ``_mathlib_env``.  All
    subsequent operations build on that environment so the full Mathlib library
    is always in scope without re-importing.

    The harness supports the context-manager protocol for safe resource cleanup::

        with LeanHarness("/path/to/lean/project") as harness:
            result = harness.check_statement("theorem foo : 1 + 1 = 2 := by sorry")

    Args:
        lean_project_dir: Path to the Lean project root containing
            ``lakefile.toml``.
        repl_timeout: Default per-command timeout in seconds.  Individual
            calls may override this via the ``timeout`` parameter.
    """

    def __init__(
        self,
        lean_project_dir: str | Path,
        repl_timeout: int = 30,
    ) -> None:
        self._project_dir = Path(lean_project_dir).resolve()
        if not self._project_dir.is_dir():
            raise FileNotFoundError(
                f"Lean project directory does not exist: {self._project_dir}"
            )

        self._repl_timeout = repl_timeout
        self._process: Optional[subprocess.Popen] = None
        self._mathlib_env: int = 0
        self._lock = threading.Lock()
        # RLock allows start() -> send_command() re-entrancy from within _restart_lock.
        self._restart_lock = threading.RLock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Launch the REPL subprocess and pre-load Mathlib.

        Executes ``lake exe repl`` in the Lean project directory.  The command
        must succeed within ``repl_timeout * 5`` seconds (Mathlib imports can
        be slow on cold cache).

        Raises:
            LeanREPLNotFoundError: If the REPL binary cannot be found or the
                subprocess fails to start.
            RuntimeError: If the Mathlib import itself returns an error.
        """
        if self.is_running:
            logger.debug("LeanHarness.start() called on already-running REPL; no-op")
            return

        logger.info("Starting Lean REPL in %s", self._project_dir)
        try:
            self._process = subprocess.Popen(
                ["lake", "exe", "repl"],
                cwd=self._project_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise LeanREPLNotFoundError(str(exc)) from exc
        except OSError as exc:
            raise LeanREPLNotFoundError(str(exc)) from exc

        # Warmup: import Mathlib once and cache the resulting env ID.
        # Use a generous timeout because Mathlib oleans may need building.
        import_timeout = max(self._repl_timeout * 5, 120)
        logger.info("Importing Mathlib (timeout=%ds) — this may take a while on first run", import_timeout)
        result = self.send_command("import Mathlib", env=0, timeout=import_timeout)

        if not result.success:
            errors = [_strip_ansi(m.get("data", "")) for m in result.messages if m.get("severity") == "error"]
            self.stop()
            raise RuntimeError(
                f"Failed to import Mathlib in the REPL. Errors:\n" + "\n".join(errors)
            )

        self._mathlib_env = result.env_id
        logger.info("Mathlib imported successfully | env_id=%d", self._mathlib_env)

    def stop(self) -> None:
        """
        Gracefully terminate the REPL subprocess.

        Sends SIGTERM and waits up to 5 seconds; escalates to SIGKILL if the
        process does not exit in time.
        """
        if self._process is None:
            return

        logger.info("Stopping Lean REPL (pid=%d)", self._process.pid)
        try:
            self._process.stdin.close()  # type: ignore[union-attr]
        except OSError:
            pass

        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("REPL did not exit after SIGTERM; sending SIGKILL")
            self._process.kill()
            self._process.wait()
        except OSError:
            pass
        finally:
            self._process = None
            self._mathlib_env = 0

    def __enter__(self) -> "LeanHarness":
        """Start the REPL and return self for use as a context manager."""
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Stop the REPL on context-manager exit regardless of exceptions."""
        self.stop()

    @property
    def is_running(self) -> bool:
        """
        True if the REPL subprocess is alive and has not exited.

        Returns:
            Boolean indicating subprocess liveness.
        """
        return self._process is not None and self._process.poll() is None

    # ------------------------------------------------------------------
    # Core command dispatch
    # ------------------------------------------------------------------

    def _restart_if_dead(self) -> None:
        """
        Restart the REPL subprocess if it has exited unexpectedly.

        Called defensively before every ``send_command`` invocation.  If the
        process is dead, ``start()`` is invoked to re-establish the Mathlib
        environment.
        """
        if not self.is_running:
            logger.warning("REPL subprocess is dead; attempting automatic restart")
            self._process = None
            self._mathlib_env = 0
            self.start()

    def send_command(
        self,
        cmd: str,
        env: int = 0,
        timeout: int | None = None,
    ) -> LeanCheckResult:
        """
        Send a single command to the REPL and return the parsed result.

        Thread-safe: acquires ``self._lock`` for the duration of the
        write-read cycle.

        Args:
            cmd: Lean 4 source code to evaluate.
            env: Environment ID to evaluate the command in.  Use ``0`` for
                the empty environment or ``self._mathlib_env`` to build on
                the Mathlib import.
            timeout: Per-command timeout in seconds.  Falls back to
                ``self._repl_timeout`` when ``None``.

        Returns:
            A ``LeanCheckResult`` with the success flag, messages, and
            the new environment ID.

        Raises:
            TimeoutError: If the REPL does not respond within the timeout.
            RuntimeError: If the REPL process dies and cannot be restarted.
        """
        effective_timeout = timeout if timeout is not None else self._repl_timeout

        # Serialise restart attempts so concurrent threads don't each spawn a
        # new REPL process when the existing one is found dead.
        with self._restart_lock:
            self._restart_if_dead()

        payload = json.dumps({"cmd": cmd, "env": env}) + "\n"
        raw_response: list[str] = []
        timed_out = threading.Event()

        def _kill_on_timeout() -> None:
            timed_out.set()
            logger.error(
                "REPL command timed out after %ds; killing subprocess", effective_timeout
            )
            if self._process and self._process.poll() is None:
                self._process.kill()

        timer = threading.Timer(effective_timeout, _kill_on_timeout)

        with self._lock:
            assert self._process is not None
            assert self._process.stdin is not None
            assert self._process.stdout is not None

            try:
                logger.debug("REPL send | env=%d | cmd=%s", env, cmd[:120])
                timer.start()
                self._process.stdin.write(payload)
                self._process.stdin.flush()
                line = self._process.stdout.readline()
                raw_response.append(line)
            except OSError as exc:
                raise RuntimeError(f"REPL I/O error during command dispatch: {exc}") from exc
            finally:
                timer.cancel()

        if timed_out.is_set():
            raise TimeoutError(
                f"REPL command timed out after {effective_timeout}s: {cmd[:80]!r}"
            )

        if not raw_response or not raw_response[0].strip():
            raise RuntimeError("REPL returned an empty response")

        parsed = _parse_response(raw_response[0])

        # Sanitise message data by stripping ANSI codes
        messages: list[dict] = []
        for msg in parsed.get("messages", []):
            clean_msg = dict(msg)
            if "data" in clean_msg:
                clean_msg["data"] = _strip_ansi(clean_msg["data"])
            messages.append(clean_msg)

        new_env_id: int = parsed.get("env", env)
        success = _result_is_success(messages)

        logger.debug(
            "REPL recv | env=%d -> %d | success=%s | messages=%d",
            env,
            new_env_id,
            success,
            len(messages),
        )
        return LeanCheckResult(success=success, messages=messages, env_id=new_env_id)

    # ------------------------------------------------------------------
    # High-level checking methods
    # ------------------------------------------------------------------

    def check_statement(self, theorem_code: str) -> LeanCheckResult:
        """
        Verify that a Lean 4 theorem statement typechecks with ``sorry``.

        If ``theorem_code`` does not already contain ``by sorry``, it is
        appended automatically.  The command is evaluated on top of the cached
        Mathlib environment.

        Args:
            theorem_code: Lean 4 source code for the theorem statement (with
                or without a ``by sorry`` body).

        Returns:
            A ``LeanCheckResult`` with ``success=True`` if the statement
            typechecks without error-severity messages.
        """
        if not self.is_running:
            raise RuntimeError("LeanHarness is not running. Call start() or use as a context manager.")

        code = theorem_code.strip()
        if "by sorry" not in code and not code.endswith("sorry"):
            code = code + " := by sorry"

        logger.debug("check_statement | %s", code[:120])
        return self.send_command(code, env=self._mathlib_env)

    def try_proof(self, statement_code: str, tactic_body: str) -> LeanCheckResult:
        """
        Attempt to close a theorem using a provided tactic body.

        Replaces the ``sorry`` placeholder in ``statement_code`` with
        ``tactic_body``.  If no ``sorry`` is present the tactic body is
        substituted for any existing ``by\\n  <content>`` block.

        Args:
            statement_code: Lean 4 theorem statement (usually ending with
                ``by\\n  sorry``).
            tactic_body: Tactic expression to substitute for ``sorry``.

        Returns:
            A ``LeanCheckResult`` with ``success=True`` only if the resulting
            theorem has no error-severity messages.
        """
        if not self.is_running:
            raise RuntimeError("LeanHarness is not running. Call start() or use as a context manager.")

        code = statement_code.strip()

        if "sorry" in code:
            proof_code = code.replace("sorry", tactic_body, 1)
        else:
            # Append the tactic as the proof body
            proof_code = code + f"\n  := by\n  {tactic_body}"

        logger.debug("try_proof | tactic=%s | code=%s", tactic_body[:60], proof_code[:120])
        return self.send_command(proof_code, env=self._mathlib_env)

    def verify_full_proof(self, full_lean_code: str) -> LeanCheckResult:
        """
        Verify a complete Lean 4 proof that contains no ``sorry`` placeholders.

        A proof is considered verified only when the REPL returns no
        error-severity messages *and* no ``sorry``-related warnings.

        Args:
            full_lean_code: Complete Lean 4 source (theorem + proof, no sorry).

        Returns:
            A ``LeanCheckResult`` where ``success=True`` indicates a
            fully verified, sorry-free proof.
        """
        if not self.is_running:
            raise RuntimeError("LeanHarness is not running. Call start() or use as a context manager.")

        if "sorry" in full_lean_code:
            logger.warning(
                "verify_full_proof received code containing 'sorry'; "
                "this will not be considered a valid proof"
            )

        logger.debug("verify_full_proof | code=%s", full_lean_code[:120])
        result = self.send_command(full_lean_code, env=self._mathlib_env)

        if result.success and _has_sorry_warning(result.messages):
            return LeanCheckResult(
                success=False,
                messages=result.messages,
                env_id=result.env_id,
            )

        return result

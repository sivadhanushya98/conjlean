#!/usr/bin/env bash
# install_lean.sh — Install Lean 4 + Mathlib via elan and build the ConjLean
# Lean project.
#
# Usage:
#   bash scripts/install_lean.sh
#
# This script:
#   1. Checks whether elan is already installed; installs it if not.
#   2. Adds ~/.elan/bin to PATH for the current session.
#   3. Verifies lake is reachable.
#   4. cd-s into the Lean project and runs lake update + lake build.
#
# Expected total runtime: 20-40 minutes on first run (Mathlib compilation).
# Subsequent runs are fast because oleans are cached.

set -euo pipefail

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
_bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
_info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
_ok()    { printf '\033[1;32m[ OK ]\033[0m  %s\n' "$*"; }
_warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
_step()  { printf '\n\033[1;36m━━━  %s  ━━━\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
# Resolve the directory of this script so we can find the Lean project
# regardless of where the script is invoked from.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LEAN_PROJECT_DIR="${REPO_ROOT}/lean/ConjLean"
ELAN_BIN="${HOME}/.elan/bin"

_bold "ConjLean — Lean 4 + Mathlib Installation"
_info "Repo root:         ${REPO_ROOT}"
_info "Lean project:      ${LEAN_PROJECT_DIR}"
_info "elan bin dir:      ${ELAN_BIN}"

# ---------------------------------------------------------------------------
# Step 1 — Install elan if absent
# ---------------------------------------------------------------------------
_step "Step 1: Check / install elan (Lean version manager)"

if [[ -x "${ELAN_BIN}/elan" ]]; then
    _ok "elan already installed at ${ELAN_BIN}/elan"
    "${ELAN_BIN}/elan" --version
else
    _info "elan not found — downloading installer …"
    curl https://elan.lean-lang.org/elan-init.sh -sSf | sh -s -- -y --no-modify-path
    _ok "elan installer finished"
fi

# ---------------------------------------------------------------------------
# Step 2 — Extend PATH for this session
# ---------------------------------------------------------------------------
_step "Step 2: Add ~/.elan/bin to PATH"

export PATH="${ELAN_BIN}:${PATH}"
_ok "PATH updated: ${ELAN_BIN} prepended"

# Sanity-check that elan and lake are now on PATH
if ! command -v lake &>/dev/null; then
    # lake may not be on PATH yet if the toolchain hasn't been activated.
    # Resolve explicitly from elan's managed bin directory.
    if [[ -x "${ELAN_BIN}/lake" ]]; then
        _ok "lake found at ${ELAN_BIN}/lake (via elan)"
    else
        _warn "lake not yet on PATH — it will be activated when elan resolves the toolchain"
    fi
else
    _ok "lake on PATH: $(command -v lake)"
fi

# ---------------------------------------------------------------------------
# Step 3 — Verify lake version
# ---------------------------------------------------------------------------
_step "Step 3: Verify lake is available"

if command -v lake &>/dev/null; then
    lake --version
    _ok "lake version verified"
elif [[ -x "${ELAN_BIN}/lake" ]]; then
    "${ELAN_BIN}/lake" --version
    _ok "lake version verified (direct path)"
else
    _warn "lake binary not found after elan install — elan may still be setting up the toolchain"
    _info "Attempting to activate toolchain via: elan toolchain install leanprover/lean4:stable"
    "${ELAN_BIN}/elan" toolchain install leanprover/lean4:stable || true
    _info "Re-checking for lake …"
    if [[ -x "${ELAN_BIN}/lake" ]]; then
        "${ELAN_BIN}/lake" --version
        _ok "lake version verified after toolchain install"
    else
        printf '\033[1;31m[FAIL]\033[0m  lake still not found — check elan installation.\n'
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Step 4 — Verify the Lean project directory exists
# ---------------------------------------------------------------------------
_step "Step 4: Verify Lean project directory"

if [[ ! -d "${LEAN_PROJECT_DIR}" ]]; then
    printf '\033[1;31m[FAIL]\033[0m  Lean project directory not found: %s\n' "${LEAN_PROJECT_DIR}"
    _info "Expected layout:  lean/ConjLean/lakefile.toml"
    exit 1
fi

if [[ ! -f "${LEAN_PROJECT_DIR}/lakefile.toml" ]]; then
    printf '\033[1;31m[FAIL]\033[0m  lakefile.toml not found in %s\n' "${LEAN_PROJECT_DIR}"
    exit 1
fi

_ok "Lean project found: ${LEAN_PROJECT_DIR}"

# ---------------------------------------------------------------------------
# Step 5 — lake update (resolve + download Mathlib and other dependencies)
# ---------------------------------------------------------------------------
_step "Step 5: lake update — resolving Mathlib dependencies"
_info "This downloads Mathlib source and cached oleans (~1-2 GB). May take several minutes."

cd "${LEAN_PROJECT_DIR}"
lake update
_ok "lake update complete"

# ---------------------------------------------------------------------------
# Step 6 — lake build (compile the project against Mathlib)
# ---------------------------------------------------------------------------
_step "Step 6: lake build — compiling ConjLean project"
_warn "This step can take 20-40 minutes on a cold cache while Lean compiles Mathlib."
_info "Subsequent builds are fast because oleans are cached in .lake/build/."

lake build
_ok "lake build complete"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
_step "Installation complete"
_bold "Lean 4 + Mathlib + ConjLean project are ready."
_info "Run the setup check:   python3 scripts/check_setup.py"
_info "Run the smoke test:    python3 scripts/run_smoke_test.py"

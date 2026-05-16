"""
Tool: execute_python
Sandboxed Python execution for the Artifact Agent.
Captures stdout / stderr, enforces a timeout, and tracks any files written.
"""

from __future__ import annotations
import os
import subprocess
import sys
import textwrap
import tempfile
import glob
from pathlib import Path
from langchain_core.tools import tool
from app.models.schemas import CodeExecutionInput, CodeExecutionOutput

WORKSPACE_ARTIFACTS = Path("workspace/artifacts")
WORKSPACE_ARTIFACTS.mkdir(parents=True, exist_ok=True)

# Preamble injected into every executed script so matplotlib saves to disk
_PREAMBLE = """\
import os, sys, warnings
warnings.filterwarnings("ignore")
os.makedirs("workspace/artifacts", exist_ok=True)

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt

# Patch plt.show() so any show() call saves the figure instead
_fig_counter = [0]
_original_show = plt.show
def _patched_show(*args, **kwargs):
    _fig_counter[0] += 1
    path = f"workspace/artifacts/figure_{_fig_counter[0]}.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    print(f"[artifact_saved] {path}")
    plt.close()
plt.show = _patched_show
"""


@tool("execute_python", args_schema=CodeExecutionInput)
def execute_python(code: str, timeout_seconds: int = 30) -> dict:
    """
    Execute Python code in a subprocess sandbox.
    - Captures stdout and stderr separately.
    - Enforces a timeout (default 30 s).
    - Automatically saves matplotlib figures to workspace/artifacts/.
    - Returns success flag and paths of any artifacts produced.

    Use this whenever you need to compute something, build a chart,
    or produce a structured table from census data.
    """
    full_code = textwrap.dedent(_PREAMBLE) + "\n" + code

    # Write to a temp file so tracebacks show real line numbers
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="agent_code_"
    ) as f:
        f.write(full_code)
        tmp_path = f.name

    # Snapshot artifacts dir before execution
    before = set(glob.glob("workspace/artifacts/*"))

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=os.getcwd(),
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        success = result.returncode == 0

    except subprocess.TimeoutExpired:
        stdout = ""
        stderr = f"TimeoutError: execution exceeded {timeout_seconds}s"
        success = False

    except Exception as e:
        stdout = ""
        stderr = f"ExecutionError: {e}"
        success = False

    finally:
        os.unlink(tmp_path)

    # Detect newly created artifacts
    after = set(glob.glob("workspace/artifacts/*"))
    new_artifacts = sorted(after - before)

    return CodeExecutionOutput(
        stdout=stdout,
        stderr=stderr,
        success=success,
        artifact_paths=new_artifacts,
    ).model_dump()
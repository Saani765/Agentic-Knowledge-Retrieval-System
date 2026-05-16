"""
Tool: execute_python
Sandboxed Python execution for the Artifact Agent.
Each run gets a unique timestamp-based filename — no overwrites.
"""

from __future__ import annotations
import os
import subprocess
import sys
import textwrap
import tempfile
import glob
from datetime import datetime
from pathlib import Path
from langchain_core.tools import tool
from app.models.schemas import CodeExecutionInput, CodeExecutionOutput

WORKSPACE_ARTIFACTS = Path("workspace/artifacts")
WORKSPACE_ARTIFACTS.mkdir(parents=True, exist_ok=True)

# Preamble injected into every script:
# - matplotlib non-interactive backend
# - plt.show() patched to save with unique timestamped filename
_PREAMBLE = """\
import os, sys, warnings
warnings.filterwarnings("ignore")
os.makedirs("workspace/artifacts", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

# Each figure gets a unique timestamp so nothing is overwritten
_fig_counter = [0]
def _patched_show(*args, **kwargs):
    _fig_counter[0] += 1
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"workspace/artifacts/figure_{ts}_{_fig_counter[0]}.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    print(f"[artifact_saved] {path}")
    plt.close()
plt.show = _patched_show
"""


@tool("execute_python", args_schema=CodeExecutionInput)
def execute_python(code: str, timeout_seconds: int = 30) -> dict:
    """
    Execute Python code in a subprocess sandbox.
    - Unique timestamped filenames — figures never overwrite each other.
    - Captures stdout and stderr separately.
    - Enforces a timeout (default 30s).
    - Returns success flag and paths of any new artifacts produced.
    """
    full_code = textwrap.dedent(_PREAMBLE) + "\n" + code

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="agent_code_"
    ) as f:
        f.write(full_code)
        tmp_path = f.name

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

    after = set(glob.glob("workspace/artifacts/*"))
    new_artifacts = sorted(after - before)

    return CodeExecutionOutput(
        stdout=stdout,
        stderr=stderr,
        success=success,
        artifact_paths=new_artifacts,
    ).model_dump()
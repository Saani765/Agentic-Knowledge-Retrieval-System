"""
Tools: write_file / read_file
Agent filesystem working memory — artifacts, notes, conversation snapshots.
"""

from __future__ import annotations
from pathlib import Path
from langchain_core.tools import tool
from app.models.schemas import WriteFileInput, WriteFileOutput

WORKSPACE_ROOT = Path("workspace")


@tool("write_file", args_schema=WriteFileInput)
def write_file(filename: str, content: str, subdir: str = "artifacts") -> dict:
    """
    Write a text file to the agent's working memory directory.
    Use subdir='artifacts' for produced outputs (CSV, HTML tables, etc.),
    subdir='notes' for intermediate reasoning notes.
    Returns the full path written.
    """
    target_dir = WORKSPACE_ROOT / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
    path.write_text(content, encoding="utf-8")
    return WriteFileOutput(path=str(path), success=True).model_dump()


@tool("read_file")
def read_file(path: str) -> dict:
    """
    Read a file from the agent's working memory directory.
    Use this to load previously produced artifacts or notes.
    """
    p = Path(path)
    if not p.exists():
        return {"success": False, "content": "", "error": f"File not found: {path}"}
    return {"success": True, "content": p.read_text(encoding="utf-8"), "error": ""}
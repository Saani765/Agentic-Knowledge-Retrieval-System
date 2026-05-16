"""
LLM factory.
Tested working models on Groq (May 2026):
- llama-3.3-70b-versatile     → tool calling agents (RAG, Summarizer, Artifact)
- llama-3.1-8b-instant         → Supervisor only (fast, no tools needed)
"""

from __future__ import annotations
import os
from groq import Groq


def get_groq_client() -> Groq:
    """Return a Groq native client for tool-calling agents."""
    return Groq(api_key=os.environ["GROQ_API_KEY"])


TOOL_MODEL = "llama-3.3-70b-versatile"
SUPERVISOR_MODEL = "llama-3.1-8b-instant"
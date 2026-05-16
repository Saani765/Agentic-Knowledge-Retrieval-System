"""
Run this to find which Groq model works for tool calling on your machine.
Usage: python test_models.py
"""
import os
import json
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Search census documents for relevant information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "description": "Number of results"}
            },
            "required": ["query"]
        }
    }
}]

MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-groq-70b-8192-tool-use-preview",
    "llama3-groq-8b-8192-tool-use-preview",
]

print("Testing tool calling on all Groq models...\n")

for model in MODELS:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Always call search_documents before answering."},
                {"role": "user",   "content": "What is the literacy rate in Odisha in 2011?"}
            ],
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
            max_tokens=256,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            args = tc.function.arguments
            print(f"✅ WORKS  | {model}")
            print(f"          tool={tc.function.name}, args={args[:80]}\n")
        else:
            print(f"⚠️  NO TOOL | {model}")
            print(f"          answered directly: {(msg.content or '')[:80]}\n")
    except Exception as e:
        err = str(e)[:120]
        if "decommissioned" in err:
            print(f"💀 DEAD   | {model} — decommissioned\n")
        elif "tool_use_failed" in err or "failed_generation" in err:
            print(f"❌ BROKEN | {model} — tool_use_failed (XML format bug)\n")
        elif "not found" in err.lower() or "does not exist" in err.lower():
            print(f"🚫 N/A    | {model} — model not available\n")
        else:
            print(f"❌ ERROR  | {model}\n          {err}\n")

print("\nUse the first ✅ model in llm.py")
"""
Graph Node: Supervisor
Classifies intent and routes. Uses fast 8B model — no tools needed.
"""

from __future__ import annotations
from app.models.schemas import AgentState
from app.graph.llm import get_groq_client, SUPERVISOR_MODEL

SYSTEM_PROMPT = """You are a routing supervisor for a census document Q&A system.
The system ONLY has data about Indian Census 2011.

Classify the user's question into EXACTLY ONE intent word:

- query      : A specific factual question answerable from Indian Census 2011 data.
               Examples: "literacy rate in UP", "population of Odisha", "sex ratio in Bihar"

- summarize  : A request for a high-level overview or key findings IN TEXT FORM.
               Examples: "summarize Odisha findings", "what are key demographics of MP"
               NOTE: If the user asks to "build", "create", "make", "generate", "show"
               a summary — that is still summarize ONLY if they do NOT mention
               chart, graph, table, plot, visualization.

- artifact   : A request to produce ANY visual output — chart, graph, table, plot,
               bar chart, pie chart, line chart, comparison table, data table, CSV.
               Examples: "bar chart of population", "build a table of literacy rates",
               "create a comparison table", "show me a graph", "plot urban vs rural",
               "make a table summarizing population" — ANY mention of table/chart/graph
               routes HERE even if the word "summarize" also appears.

- refuse     : The question is NOT about Indian Census 2011 data.
               Examples: "population of France", "GDP of USA", "capital of Japan",
               "who is the president", "weather today", "write a poem",
               ANY question about a country/region outside India,
               ANY question not related to census or demographic data of India.

CRITICAL RULES:
- If the question mentions any non-Indian geography (France, USA, China, UK, etc.) → refuse
- If the question mentions "table", "chart", "graph", "plot", "visualization" → artifact
- "build a table" = artifact, "create a chart" = artifact, "show me a graph" = artifact
- Only route to summarize for pure text summary requests with no visual output requested

Respond with ONLY the single intent word. No explanation. No punctuation.
"""


def supervisor_node(state: AgentState) -> AgentState:
    client = get_groq_client()

    history = state.get("conversation_history", [])
    user_msg = state["user_query"]

    if history:
        recent = history[-3:]
        snippet = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)
        user_msg = f"Conversation so far:\n{snippet}\n\nNew question: {user_msg}"

    response = client.chat.completions.create(
        model=SUPERVISOR_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0,
        max_tokens=10,
    )

    intent = response.choices[0].message.content.strip().lower().strip(".")

    # Hard-coded keyword overrides — catches cases the LLM misses
    query_lower = state["user_query"].lower()

    # If any visual keyword present → always artifact
    visual_keywords = ["table", "chart", "graph", "plot", "bar", "pie",
                       "visualization", "visualise", "visualize", "csv", "diagram"]
    if any(kw in query_lower for kw in visual_keywords):
        intent = "artifact"

    # Sanitise
    if intent not in ("query", "summarize", "artifact", "refuse"):
        intent = "query"

    return {**state, "intent": intent}


def route_by_intent(state: AgentState) -> str:
    return state.get("intent", "query")
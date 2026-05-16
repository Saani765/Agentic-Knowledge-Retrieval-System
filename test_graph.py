import asyncio
from app.graph.builder import graph

def test_query(query: str):
    initial_state = {
        "user_query": query,
        "conversation_history": [],
        "intent": "",
        "retrieved_chunks": [],
        "citations": [],
        "generated_code": "",
        "code_output": "",
        "code_error": "",
        "artifact_path": "",
        "retry_count": 0,
        "final_response": "",
        "refused": False,
        "refusal_reason": "",
    }
    config = {"configurable": {"thread_id": "test_session_123"}}
    try:
        result = graph.invoke(initial_state, config=config)
        print(f"Success for '{query}': intent={result.get('intent')}")
    except Exception as e:
        import traceback
        print(f"\n--- Error for query '{query}' ---")
        traceback.print_exc()

if __name__ == "__main__":
    queries = [
        "hello",
        "plot the closing price of AAPL for the last 30 days",
        "what is the revenue of microsoft?",
        "compare AAPL and MSFT"
    ]
    for q in queries:
        test_query(q)

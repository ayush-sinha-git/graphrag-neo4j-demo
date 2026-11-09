"""LangGraph orchestration for retrieval and grounded answer generation."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from retrievers import entity_to_document, hybrid_search, text2cypher


class GraphRAGState(TypedDict, total=False):
    user_query: str
    selected_strategy: str
    retrieved_context: str
    cypher_query: str
    final_answer: str
    error: str
    trace: list[dict[str, Any]]
    retrieval_metadata: dict[str, Any]


def create_workflow(
    *,
    driver: Any,
    database: str,
    graph: Any,
    embeddings: Any,
    llm: Any,
) -> Any:
    """Compile the stateful graph with in-memory checkpoint persistence."""
    builder = StateGraph(GraphRAGState)

    def route(state: GraphRAGState) -> dict[str, Any]:
        return {"trace": [{"node": "route", "status": state.get("selected_strategy", "hybrid")}]}

    def retrieve(strategy: str):
        def node(state: GraphRAGState) -> dict[str, Any]:
            started = perf_counter()
            try:
                if strategy == "hybrid":
                    result = hybrid_search(
                        driver=driver,
                        database=database,
                        embeddings=embeddings,
                        query=state["user_query"],
                    )
                elif strategy == "text2cypher":
                    result = text2cypher(
                        driver=driver,
                        database=database,
                        graph=graph,
                        llm=llm,
                        query=state["user_query"],
                    )
                else:
                    result = entity_to_document(
                        driver=driver,
                        database=database,
                        llm=llm,
                        query=state["user_query"],
                    )
                metadata = result["metadata"]
                return {
                    "retrieved_context": result["context"],
                    "retrieval_metadata": metadata,
                    "cypher_query": metadata.get("cypher", ""),
                    "trace": state.get("trace", [])
                    + [{"node": f"retrieve_{strategy}", "elapsed_ms": round((perf_counter() - started) * 1000, 2)}],
                }
            except Exception as error:
                return {
                    "error": f"{type(error).__name__}: {error}",
                    "retrieved_context": "",
                    "trace": state.get("trace", [])
                    + [{"node": f"retrieve_{strategy}", "error": str(error)}],
                }

        return node

    def generate(state: GraphRAGState) -> dict[str, Any]:
        started = perf_counter()
        if state.get("error"):
            return {
                "final_answer": f"Retrieval failed: {state['error']}",
                "trace": state.get("trace", [])
                + [{"node": "generate_answer", "skipped": "retrieval error"}],
            }
        try:
            response = llm.invoke(
                [
                    (
                        "system",
                        "Answer the user using only the supplied retrieved context. "
                        "If it does not contain the answer, say so plainly. Distinguish "
                        "reported allegations or inquiries from established findings.",
                    ),
                    (
                        "human",
                        f"Question:\n{state['user_query']}\n\nRetrieved context:\n"
                        f"{state.get('retrieved_context', '')}",
                    ),
                ]
            )
            return {
                "final_answer": str(response.content),
                "trace": state.get("trace", [])
                + [{"node": "generate_answer", "elapsed_ms": round((perf_counter() - started) * 1000, 2)}],
            }
        except Exception as error:
            return {
                "error": f"{type(error).__name__}: {error}",
                "final_answer": f"Answer generation failed: {error}",
                "trace": state.get("trace", [])
                + [{"node": "generate_answer", "error": str(error)}],
            }

    builder.add_node("route", route)
    builder.add_node("retrieve_hybrid", retrieve("hybrid"))
    builder.add_node("retrieve_text2cypher", retrieve("text2cypher"))
    builder.add_node("retrieve_entity_to_document", retrieve("entity_to_document"))
    builder.add_node("generate_answer", generate)
    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        lambda state: _strategy_key(state.get("selected_strategy", "hybrid")),
        {
            "hybrid": "retrieve_hybrid",
            "text2cypher": "retrieve_text2cypher",
            "entity_to_document": "retrieve_entity_to_document",
        },
    )
    builder.add_edge("retrieve_hybrid", "generate_answer")
    builder.add_edge("retrieve_text2cypher", "generate_answer")
    builder.add_edge("retrieve_entity_to_document", "generate_answer")
    builder.add_edge("generate_answer", END)
    return builder.compile(checkpointer=MemorySaver())


def _strategy_key(strategy: str) -> Literal["hybrid", "text2cypher", "entity_to_document"]:
    if strategy == "text2cypher":
        return "text2cypher"
    if strategy == "entity_to_document":
        return "entity_to_document"
    return "hybrid"

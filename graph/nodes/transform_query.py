from typing import Any, Dict

from graph.state import GraphState
from graph.chains.rewriter import question_rewriter


def transform_query(state: GraphState) -> Dict[str, Any]:
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("Re-writing user query...")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}
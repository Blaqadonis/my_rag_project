from typing import Any, Dict

from graph.state import GraphState
from graph.chains.substitute_llm import rag_chain2


def llm_fallback(state: GraphState) -> Dict[str, Any]:
    """
    Generate answer using the LLM w/o vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("Substitute LLM.")
    question = state["question"]
    generation2 = rag_chain2.invoke({"question": question})
    return {"question": question, "generation": generation2}
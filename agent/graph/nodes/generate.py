from typing import Any, Dict

from graph.chains.generation import rag_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("Generating response...")
    question = state["question"]
    documents = state["documents"]

    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

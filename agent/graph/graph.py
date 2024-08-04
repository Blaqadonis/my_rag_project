from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH, SUBSTITUTE_LLM, TRANSFORM_QUERY
from graph.nodes import generate, grade_documents, retrieve, web_search, llm_fallback, transform_query
from graph.state import GraphState
from pprint import pprint

load_dotenv()


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("Assessing Graded documents...")
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # It will re-generate a new query
        print("DECISION: Retrieved documents are NOT relevant to query, restructuring query...")
        return TRANSFORM_QUERY
    else:
        # relevant documents, so generate answer
        print("DECISION: Respond.")
        return GENERATE

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("Checking for hallucination...")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    hallucination_grade = score.binary_score

    # Check hallucination
    if hallucination_grade := score.binary_score:
        print("DECISION: Response is grounded. No hallucination.")
        # Check question-answering
        print("Grading relevancy to user query...")
        score = answer_grader.invoke({"question": question,"generation": generation})
        answer_grade = score.binary_score
        if answer_grade := score.binary_score:
            print("DECISION: Response is relevant to user query.")
            return "useful"
        else:
            print("DECISION: Response DOES NOT address user query.")
            return "not useful"
    else:
        pprint("DECISION: Response is NOT grounded with knowledge base. Retrying...")
        return "not supported"

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("Routing query...")
    question = state["question"]
    source = question_router.invoke({"question": question})

    # Fallback to LLM or raise error if no decision
    #if "tool_calls" not in source.datasource or len(source.datasource["tool_calls"]) == 0:
        #print("Routing query to substitute generator...")
        #return "llm_fallback"

    # Choose datasource
    datasource = source.datasource        #["tool_calls"][0]["function"]["name"]
    if datasource == 'websearch':
        print("Routing query to the web search tool...")
        return "web_search"
    elif datasource == 'vectorstore':
        print("Routing query to vector database...")
        return "retrieve"
    else:
        print("Routing query to substitute generator...")
        return "llm_fallback"
   


workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)
workflow.add_node(SUBSTITUTE_LLM, llm_fallback) 
workflow.add_node(TRANSFORM_QUERY, transform_query)


# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
        SUBSTITUTE_LLM: SUBSTITUTE_LLM,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        TRANSFORM_QUERY: TRANSFORM_QUERY,
        GENERATE: GENERATE,
    },
)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_v_documents_and_question,
    {
        "not supported": END, # Hallucinations: well, that's the internet fault not my app's. lol 
        "not useful": TRANSFORM_QUERY, # restructure the query 
        "useful": END,
    },
)
workflow.add_edge(TRANSFORM_QUERY, RETRIEVE)
workflow.add_edge(SUBSTITUTE_LLM, END)

# Compile
app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")

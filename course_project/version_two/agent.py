# Import necessary libraries
from dotenv import load_dotenv
import os
from uuid import uuid4
import requests
from getpass import getpass
from ragatouille import RAGPretrainedModel
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import MessagesState, StateGraph, END, START
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables from .env file
load_dotenv()

# Generate a unique identifier
unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"PDFChatbot LangGraph - {unique_id}"
os.environ['LANGCHAIN_API_KEY'] = getpass("Enter your LANGCHAIN API Key: ")

# Get PDF document
pdf_path = "human-nutrition-text.pdf"

# Download PDF if it doesn't already exist
if not os.path.exists(pdf_path):
    print("File doesn't exist, downloading...")
    url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
    response = requests.get(url)
    if response.status_code == 200:
        with open(pdf_path, "wb") as file:
            file.write(response.content)
        print(f"The file has been downloaded and saved as {pdf_path}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
else:
    print(f"File {pdf_path} exists.")

# Load the document
loader = PyMuPDFLoader(pdf_path)
text_data = loader.load()

# Join text data into a single string
def join_text_data(text_data):
    combined_text = "\n".join([data.page_content for data in text_data])
    return combined_text

doc = join_text_data(text_data)

# Initialize the RAG model and index the document
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
RAG.index(
    collection=[doc],
    index_name="human-nutrition-text-colbert",
    max_document_length=200,
    split_documents=True,
)

retriever = RAG.as_langchain_retriever(k=1)

# Create the retrieval tool
retrieval_tool = create_retriever_tool(
    retriever,
    "search_database",
    """Searches and retrieves relevant excerpts from the 'Human Nutrition - 2020 Edition' PDF.""",
)

# Web search tool
@tool
def web_search_tool(messages: str) -> str:
    search = TavilySearchResults()
    search_results = search.invoke(messages)
    results = "\n".join([f"URL: {res['url']}\nContent: {res['content']}\n" for res in search_results])
    return {"messages": [results]}

# Guard tool for sensitive topics
@tool
def guard_tool(question: str) -> str:
    guard_prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks.
                    You are not allowed to discuss sensitive topics such as flirting, politics, religion, or other controversial subjects.
                    Answer the following question only with this reply:
                    "Sorry, this is sensitive. I have been programmed by 🅱🅻🅰🆀 not to engage in discussions on this topic."
                    Question: {question}.
                    Your reply: "Sorry, this is sensitive. I have been programmed by 🅱🅻🅰🆀 not to engage in discussions on this topic."
                    """,
        input_variables=["question"]
    )
    guard = ChatGroq(temperature=0.1, model_name="llama-3.1-70b-versatile")
    guardrail = guard_prompt | guard | StrOutputParser()
    response = guardrail.invoke({"question": question})
    return {"response": response}

# Initialize the LLM
llm = ChatGroq(temperature=0.1, model_name="llama3-groq-8b-8192-tool-use-preview")
tools = [retrieval_tool, web_search_tool, guard_tool]
llm_with_tools = llm.bind_tools(tools)

# Assistant node
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Summarize conversation
def summarize_conversation(state: MessagesState):
    summary_message = "Create a summary of the conversation above:"
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

# Define graph and nodes
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_node("summarize_conversation", summarize_conversation)
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition, should_continue)
builder.add_edge("tools", "assistant")
builder.add_edge("summarize_conversation", END)

# Memory and graph compilation
memory = MemorySaver()
react_graph = builder.compile(interrupt_before=["assistant"], checkpointer=memory)
display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))

# User interaction
user_input = {"messages": input("Enter your query here: ")}
for event in react_graph.stream(user_input, {"configurable": {"thread_id": "0"}}, stream_mode="values"):
    event["messages"][-1].pretty_print()

feedback = input("Which tool should I use - internet, or database: ")
react_graph.update_state({"configurable": {"thread_id": "0"}}, {"messages": feedback}, as_node="assistant")
for event in react_graph.stream(None, {"configurable": {"thread_id": "0"}}, stream_mode="values"):
    event["messages"][-1].pretty_print()
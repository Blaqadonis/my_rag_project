from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0)


class LLMFallback(BaseModel):
    """Reply to unrelated questions."""

    reply: str = Field(
        description="A reply to the input question that indicates that the assistant will only chat about the uploaded documents."
    )

structured_llm_fallback = llm.with_structured_output(LLMFallback)

system = """You are an assistant for question-answering tasks. Answer the following question only with this reply - I'm sorry, but Blaq will like us to only chat about your documents. 
    Please no preamble or extra sentences just this same reply - I'm sorry, but Blaq will like us to only chat about your documents."""
prompt2 = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Input question: {question}"),
    ]
)

rag_chain2 = prompt2 | structured_llm_fallback

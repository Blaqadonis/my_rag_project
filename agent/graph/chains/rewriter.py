from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0)


class RewriteQuestion(BaseModel):
    """Rewritten question optimized for vectorstore retrieval."""

    rewritten_question: str = Field(
        description="A rewritten version of the input question that is optimized for vectorstore retrieval."
    )

structured_llm_rewriter = llm.with_structured_output(RewriteQuestion)

system = """You are a question rewriter that converts an input question to a better version that is optimized for vectorstore retrieval.
Based on the input question, please formulate an improved question that is optimized for vectorstore retrieval.
Please only return the improved question, with no preamble or extra sentences."""
rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Input question: {question}"),
    ]
)

question_rewriter = rewrite_prompt | structured_llm_rewriter

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch", "classified"] = Field(
        ...,
        description="Given a user question choose to route it to web search, a vectorstore, or classify it as unrelated.",
    )


llm = ChatOpenAI(temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing user questions to the appropriate data source. Based on the provided question, please classify it into one of the following categories:

1. vectorstore: Questions that are related to human nutrition, such as questions about macronutrients, micronutrients, dietary guidelines, food labels, or the health benefits of specific foods or ingredients.
2. classified: Questions directed at the chatbot, such as asking about the chatbot's feelings, emotions, or personal experiences. Questions about intimacy with the chatbot, or personal inquiries about the chatbot's creator or developers, should also be classified as classified.
3. websearch: Any other question not classified as vectorstore nor classified. This could include questions about current events, trivia, or general knowledge.

Reply with a JSON object containing a single key 'datasource' that indicates the appropriate category for the question.
"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

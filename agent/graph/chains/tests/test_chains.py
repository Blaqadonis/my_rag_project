import pytest
from pprint import pprint
from dotenv import load_dotenv
from graph.chains.generation import rag_chain
from graph.chains.hallucination_grader import GradeHallucinations, hallucination_grader
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever

load_dotenv()

def test_generation_chain():
    question = "What are macronutrients?"
    docs = retriever.invoke(question)
    generation = rag_chain.invoke({"context": docs, "question": question})
    pprint(generation)

def test_retrieval_grader_answer_yes():
    question = "What are electrolytes required in the human body?"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    res: GradeDocuments = retrieval_grader.invoke({"question": question, "document": doc_txt})
    assert res.binary_score == "yes"

def test_retrieval_grader_answer_no():
    question = "What are macronutrients?"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    res: GradeDocuments = retrieval_grader.invoke({"question": "What is the website of NSCDC?", "document": doc_txt})
    assert res.binary_score == "no"

def test_hallucination_grader_answer_yes():
    question = "What are macronutrients?"
    docs = retriever.invoke(question)
    generation = rag_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke({"documents": docs, "generation": generation})
    assert res.binary_score

def test_hallucination_grader_answer_no():
    question = "What are macronutrients?"
    docs = retriever.invoke(question)
    res: GradeHallucinations = hallucination_grader.invoke({"documents": docs, "generation": "The website of NSCDC. I cannot categorically tell you one now."})
    assert not res.binary_score
import os
from dotenv import load_dotenv
from uuid import uuid4
import requests
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
import pandas as pd
from langchain_community.retrievers import SVMRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Generate a unique project ID
unique_id = uuid4().hex[0:8]

# Set environment variables for LangChain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Blaq's PDF Chatbot Vector Store Evaluation - {unique_id}"

# Download PDF if it doesn't already exist
pdf_path = "human-nutrition-text.pdf"
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

# Check if there is a FAISS index locally
index_path = "faiss_index"
hf = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
if os.path.exists(index_path):
    vectorstore = FAISS.load_local(index_path, hf, allow_dangerous_deserialization=True)
else:
    # Load the document and create embeddings
    loader = PyMuPDFLoader(pdf_path)
    text_data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False
    )
    chunks = text_splitter.split_documents(text_data)
    vectorstore = FAISS.from_documents(chunks, hf)
    # Save the FAISS index locally
    vectorstore.save_local(index_path)

# Initialize the gpt-4o-mini model for evaluation
gpt4_mini = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Define different retrieval methods
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 3}
)

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10}
)

svm_retriever = SVMRetriever.from_documents(chunks, hf)

# Create an ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[similarity_retriever, mmr_retriever, svm_retriever],
    weights=[0.4, 0.3, 0.3]
)

# Define a function to create the evaluation prompt
def create_evaluation_prompt(query, correct_answer, retrieved_content):
    prompt = f"""
    Question: {query}
    Correct Answer: {correct_answer}
    Retrieved Content: {retrieved_content}

    Evaluate the retrieved content based on the following criteria:
    1. Relevance: Does the content contain information relevant to answering the question?
    2. Completeness: Does the content provide all the necessary information to answer the question?
    3. Accuracy: Is the information in the content correct and aligned with the correct answer?

    Provide a score from 0 to 10, where 0 is completely irrelevant or incorrect, and 10 is perfectly relevant and accurate.
    Only respond with the numeric score.
    """
    return prompt

# Load the qna.csv file
qna_df = pd.read_csv("qna.csv")

# Function to evaluate a retriever
def evaluate_retriever(retriever, name):
    scores = []
    for _, row in qna_df.iterrows():
        query = row['question']
        correct_answer = row['answer']
        
        docs = retriever.get_relevant_documents(query)
        retrieved_content = " ".join([doc.page_content for doc in docs])
        
        evaluation_prompt = create_evaluation_prompt(query, correct_answer, retrieved_content)
        score = int(gpt4_mini.invoke(evaluation_prompt).content.strip())
        scores.append(score)
    
    average_score = sum(scores) / len(scores)
    print(f"{name} Average Score: {average_score:.2f}")
    return average_score

# Evaluate each retriever
similarity_score = evaluate_retriever(similarity_retriever, "Similarity")
mmr_score = evaluate_retriever(mmr_retriever, "MMR")
svm_score = evaluate_retriever(svm_retriever, "SVM")
ensemble_score = evaluate_retriever(ensemble_retriever, "Ensemble")

# Choose the best performing retriever
best_retriever = max(
    [("Similarity", similarity_score), ("MMR", mmr_score), 
     ("SVM", svm_score), ("Ensemble", ensemble_score)],
    key=lambda x: x[1]
)

print(f"Best performing retriever: {best_retriever[0]} with score {best_retriever[1]:.2f}")

# Use the best performing retriever for the final evaluation
best_retriever_instance = globals()[f"{best_retriever[0].lower()}_retriever"]

# Lists to store the results
questions = []
correct_answers = []
retrieved_contents = []
scores = []

# Iterate through each question in the qna.csv file
for _, row in qna_df.iterrows():
    query = row['question']
    correct_answer = row['answer']
    
    docs = best_retriever_instance.get_relevant_documents(query)
    retrieved_content = " ".join([doc.page_content for doc in docs])
    
    evaluation_prompt = create_evaluation_prompt(query, correct_answer, retrieved_content)
    score = int(gpt4_mini.invoke(evaluation_prompt).content.strip())
    
    # Store the results
    questions.append(query)
    correct_answers.append(correct_answer)
    retrieved_contents.append(retrieved_content)
    scores.append(score)

# Calculate the average score
average_score = sum(scores) / len(scores)

print(f"Final Average Score: {average_score:.2f}")

# Create a DataFrame for the output CSV
output_df = pd.DataFrame({
    "question": questions,
    "correct_answer": correct_answers,
    "retrieved_content": retrieved_contents,
    "score": scores
})

# Save the output to a CSV file
output_file = f"{best_retriever[0].lower()}-retriever-report.csv"
output_df.to_csv(output_file, index=False)
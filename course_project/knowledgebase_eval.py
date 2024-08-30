import os
from dotenv import load_dotenv
from uuid import uuid4
import requests
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
import pandas as pd
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
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'k': 1, 'score_threshold': 0.5}
    )
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
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'k': 1, 'score_threshold': 0.5}
    )
    # Save the FAISS index locally
    vectorstore.save_local(index_path)

# Initialize the gpt4o-mini model
chat_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Define a function to create the relevance checking prompt
def create_relevance_prompt(query, document_content):
    prompt = f"""
    I have a query: "{query}".
    The following is a part of a document:
    \"\"\"{document_content}\"\"\"
    The goal is to grade the performance of a vector database. Is this document content relevant to answering the query? Respond with 'YES' or 'NO'.
    """
    return prompt

# Load the qna.csv file
qna_df = pd.read_csv("qna.csv")

# Lists to store the results
questions = []
answers = []
llm_answers = []
grades = []

# Iterate through each question in the qna.csv file
for index, row in qna_df.iterrows():
    query = row['question']
    correct_answer = row['answer']
    
    # Invoke the retriever to get relevant documents
    docs = retriever.invoke(query)
    
    # Check relevance for each document returned
    for doc in docs:
        relevance_prompt = create_relevance_prompt(query, doc.page_content)
        relevance_response = chat_model([{"role": "user", "content": relevance_prompt}])
        
        # Only interested in "YES" or "NO" responses
        grade = relevance_response.content.strip()
        if grade == "YES":
            llm_answer = doc.page_content
            break
        else:
            llm_answer = doc.page_content  # Use the last document if no "YES" found
    
    # Store the results
    questions.append(query)
    answers.append(correct_answer)
    llm_answers.append(llm_answer)
    grades.append(grade)

# Calculate the relevancy metric
num_yes = grades.count("YES")
total_questions = len(questions)
relevancy_percentage = (num_yes / total_questions) * 100

print(f"Relevancy Metric: {relevancy_percentage:.2f}%")

# Create a DataFrame for the output CSV
output_df = pd.DataFrame({
    "question": questions,
    "answer": answers,
    "llm_answer": llm_answers,
    "grade": grades
})

# Save the output to a CSV file
output_df.to_csv("all-mpnet-base-v2-report.csv", index=False)
print("Output saved to all-mpnet-base-v2-report.csv")

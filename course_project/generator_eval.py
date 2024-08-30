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

# Define a function to create the hallucination checking prompt
def create_hallucination_prompt(query, llm_answer, documents):
    prompt = f"""
    I have a query: "{query}".
    The AI generated the following answer:
    \"\"\"{llm_answer}\"\"\"
    The following are the retrieved documents:
    \"\"\"{documents}\"\"\"
    Based on the retrieved documents, does the AI's answer come from the provided context, or is it hallucinated? Respond with 'FROM CONTEXT' if the answer is derived from the documents or 'HALLUCINATED' if it appears to be made up.
    """
    return prompt

# Load the qna.csv file
qna_df = pd.read_csv("qna.csv")

# Lists to store the results
questions = []
answers = []
llm_answers = []
hallucination_flags = []

# Iterate through each question in the qna.csv file
for index, row in qna_df.iterrows():
    query = row['question']
    correct_answer = row['answer']
    
    # Invoke the retriever to get relevant documents
    docs = retriever.invoke(query)
    
    # Format the retrieved documents as context
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Generate the answer using the context
    llm_answer_prompt = f"Answer the following question based only on the provided context.\n\nQuestion: {query}\nContext: {context}"
    llm_response = chat_model([{"role": "user", "content": llm_answer_prompt}])
    llm_answer = llm_response.content.strip()
    
    # Check for hallucination using gpt4o-mini
    hallucination_prompt = create_hallucination_prompt(query, llm_answer, context)
    hallucination_response = chat_model([{"role": "user", "content": hallucination_prompt}])
    
    # Only interested in "FROM CONTEXT" or "HALLUCINATED" responses
    hallucination_flag = hallucination_response.content.strip()
    
    # Store the results
    questions.append(query)
    answers.append(correct_answer)
    llm_answers.append(llm_answer)
    hallucination_flags.append(hallucination_flag)

# Calculate the hallucination rate
num_hallucinations = hallucination_flags.count("HALLUCINATED")
total_questions = len(questions)
hallucination_rate = (num_hallucinations / total_questions) * 100

print(f"Hallucination Rate: {hallucination_rate:.2f}%")

# Create a DataFrame for the output CSV
output_df = pd.DataFrame({
    "question": questions,
    "answer": answers,
    "llm_answer": llm_answers,
    "hallucination_flag": hallucination_flags
})

# Save the output to a CSV file
output_df.to_csv("llama-3.1-70b--hallucination-report.csv", index=False)
print("Output saved to llama-3.1-70b--hallucination-report.csv")

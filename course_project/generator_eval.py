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
from langchain_groq import ChatGroq

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

# Initialize the models
gpt4_mini = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
llama_70b = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile")
llama_70b_8192 = ChatGroq(temperature=0, model_name="llama3-70b-8192")

models = {
    "llama-3.1-70b-versatile": llama_70b,
    "llama3-70b-8192": llama_70b_8192
}

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

# Define a function to create the answer correctness prompt
def create_correctness_prompt(query, correct_answer, llm_answer):
    prompt = f"""
    Question: {query}
    Correct Answer: {correct_answer}
    AI's Answer: {llm_answer}

    Evaluate the AI's answer based on the following criteria:
    1. Accuracy: Is the information in the answer correct?
    2. Completeness: Does the answer provide all necessary information?
    3. Relevance: Does the answer address the question?

    Provide a score from 0 to 10, where 0 is completely incorrect or irrelevant, and 10 is perfect.
    Only respond with the numeric score.
    """
    return prompt

# Load the qna.csv file
qna_df = pd.read_csv("qna.csv")

# Function to evaluate a model
def evaluate_model(model_name, model):
    questions = []
    answers = []
    llm_answers = []
    hallucination_flags = []
    correctness_scores = []

    for _, row in qna_df.iterrows():
        query = row['question']
        correct_answer = row['answer']
        
        # Retrieve relevant documents
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        # Generate answer
        llm_answer_prompt = f"Answer the following question based only on the provided context.\n\nQuestion: {query}\nContext: {context}"
        llm_response = model.invoke(llm_answer_prompt)
        llm_answer = llm_response.content.strip()
        
        # Check for hallucination
        hallucination_prompt = create_hallucination_prompt(query, llm_answer, context)
        hallucination_response = gpt4_mini.invoke(hallucination_prompt)
        hallucination_flag = hallucination_response.content.strip()
        
        # Check answer correctness
        correctness_prompt = create_correctness_prompt(query, correct_answer, llm_answer)
        correctness_response = gpt4_mini.invoke(correctness_prompt)
        correctness_score = int(correctness_response.content.strip())
        
        # Store results
        questions.append(query)
        answers.append(correct_answer)
        llm_answers.append(llm_answer)
        hallucination_flags.append(hallucination_flag)
        correctness_scores.append(correctness_score)

    # Calculate metrics
    hallucination_rate = (hallucination_flags.count("HALLUCINATED") / len(questions)) * 100
    average_correctness = sum(correctness_scores) / len(correctness_scores)

    print(f"\nResults for {model_name}:")
    print(f"Hallucination Rate: {hallucination_rate:.2f}%")
    print(f"Average Correctness Score: {average_correctness:.2f}")

    # Create and save output DataFrame
    output_df = pd.DataFrame({
        "question": questions,
        "correct_answer": answers,
        "llm_answer": llm_answers,
        "hallucination_flag": hallucination_flags,
        "correctness_score": correctness_scores
    })
    output_file = f"{model_name}-evaluation-report.csv"
    output_df.to_csv(output_file, index=False)
    print(f"Output saved to {output_file}")

    return hallucination_rate, average_correctness

# Evaluate each model
results = {}
for model_name, model in models.items():
    hallucination_rate, average_correctness = evaluate_model(model_name, model)
    results[model_name] = {
        "hallucination_rate": hallucination_rate,
        "average_correctness": average_correctness
    }

# Compare models
print("\nModel Comparison:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"  Hallucination Rate: {metrics['hallucination_rate']:.2f}%")
    print(f"  Average Correctness Score: {metrics['average_correctness']:.2f}")

# Identify the best performing model
best_model = min(results, key=lambda x: (results[x]['hallucination_rate'], -results[x]['average_correctness']))
print(f"\nBest performing model: {best_model}")
print(f"  Hallucination Rate: {results[best_model]['hallucination_rate']:.2f}%")
print(f"  Average Correctness Score: {results[best_model]['average_correctness']:.2f}")

# Save overall comparison results
comparison_df = pd.DataFrame(results).T
comparison_df.to_csv("model_comparison_results.csv")
print("\nOverall comparison results saved to model_comparison_results.csv")

# Additional analysis
print("\nAdditional Analysis:")

# Calculate overall statistics
total_questions = len(qna_df)
total_hallucinations = sum(results[model]['hallucination_rate'] * total_questions / 100 for model in results)
overall_hallucination_rate = total_hallucinations / (total_questions * len(models)) * 100
overall_correctness = sum(results[model]['average_correctness'] for model in results) / len(models)

print(f"Overall Hallucination Rate: {overall_hallucination_rate:.2f}%")
print(f"Overall Average Correctness Score: {overall_correctness:.2f}")

# Identify questions that both models struggled with
difficult_questions = []
for _, row in qna_df.iterrows():
    query = row['question']
    if all(pd.read_csv(f"{model}-evaluation-report.csv")
           .query(f"question == '{query}' and correctness_score < 5")
           .shape[0] > 0 for model in models):
        difficult_questions.append(query)

print("\nQuestions both models struggled with:")
for question in difficult_questions:
    print(f"- {question}")

# Suggestions for improvement
print("\nSuggestions for Improvement:")
if overall_hallucination_rate > 10:
    print("- Focus on reducing hallucinations across both models.")
if overall_correctness < 7:
    print("- Work on improving the overall accuracy and completeness of answers.")
if difficult_questions:
    print("- Investigate why certain questions are difficult for both models and improve context retrieval or model understanding for these types of questions.")

# Detailed comparison
print("\nDetailed Comparison:")
for _, row in qna_df.iterrows():
    query = row['question']
    print(f"\nQuestion: {query}")
    for model_name in models:
        model_df = pd.read_csv(f"{model_name}-evaluation-report.csv")
        model_row = model_df[model_df['question'] == query].iloc[0]
        print(f"{model_name}:")
        print(f"  Answer: {model_row['llm_answer']}")
        print(f"  Hallucination: {model_row['hallucination_flag']}")
        print(f"  Correctness Score: {model_row['correctness_score']}")

# Performance analysis by question type
if 'question_type' in qna_df.columns:
    print("\nPerformance Analysis by Question Type:")
    for model_name in models:
        model_df = pd.read_csv(f"{model_name}-evaluation-report.csv")
        merged_df = pd.merge(model_df, qna_df[['question', 'question_type']], on='question')
        performance_by_type = merged_df.groupby('question_type').agg({
            'correctness_score': 'mean',
            'hallucination_flag': lambda x: (x == 'HALLUCINATED').mean() * 100
        }).rename(columns={'hallucination_flag': 'hallucination_rate'})
        
        print(f"\n{model_name}:")
        print(performance_by_type)

# Identify areas of improvement for each model
print("\nAreas of Improvement:")
for model_name in models:
    model_df = pd.read_csv(f"{model_name}-evaluation-report.csv")
    low_scoring_questions = model_df[model_df['correctness_score'] < 5]
    hallucinated_questions = model_df[model_df['hallucination_flag'] == 'HALLUCINATED']
    
    print(f"\n{model_name}:")
    print(f"  Questions with low correctness scores (<5):")
    for _, row in low_scoring_questions.iterrows():
        print(f"    - {row['question']} (Score: {row['correctness_score']})")
    
    print(f"  Questions with hallucinations:")
    for _, row in hallucinated_questions.iterrows():
        print(f"    - {row['question']}")

# Calculate confidence intervals for hallucination rates and correctness scores
from scipy import stats

print("\nConfidence Intervals (95%):")
for model_name in models:
    model_df = pd.read_csv(f"{model_name}-evaluation-report.csv")
    hallucination_rate = (model_df['hallucination_flag'] == 'HALLUCINATED').mean() * 100
    correctness_score = model_df['correctness_score'].mean()
    
    hallucination_ci = stats.norm.interval(0.95, loc=hallucination_rate, scale=stats.sem(model_df['hallucination_flag'] == 'HALLUCINATED') * 100)
    correctness_ci = stats.norm.interval(0.95, loc=correctness_score, scale=stats.sem(model_df['correctness_score']))
    
    print(f"\n{model_name}:")
    print(f"  Hallucination Rate: {hallucination_rate:.2f}% (95% CI: {hallucination_ci[0]:.2f}% - {hallucination_ci[1]:.2f}%)")
    print(f"  Correctness Score: {correctness_score:.2f} (95% CI: {correctness_ci[0]:.2f} - {correctness_ci[1]:.2f})")

print("\nEvaluation complete. Check the CSV files for detailed results.")

if __name__ == "__main__":
    print("Generator evaluation script executed successfully")
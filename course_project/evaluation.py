import os
import sys
import codecs

sys.stdout = codecs.getwriter('utf8')(sys.stdout.buffer)
import yaml
from typing import List
from operator import itemgetter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
#from langchain_cohere import CohereRerank
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from PIL import Image
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from IPython.display import display
from pdf2image import convert_from_path

from giskard import Dataset, Model, scan, GiskardClient
import pandas as pd

load_dotenv()

class PDFChatbot:
    def __init__(self, config_path: str):
        """
        Initialize the PDFChatbot with the configuration from the given path.

        Args:
            config_path (str): The path to the configuration file.
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.vector_store_model_name = config['vector_store']['model_name']
        self.vector_store_k = config['vector_store']['k']
        self.llm_model_name = config['llm']['model_name']
        self.llm_temperature = config['llm']['temperature']

        self.store = {}
        self.faiss_index = None
        self.hf = SentenceTransformerEmbeddings(model_name=self.vector_store_model_name)
        self.llm = ChatGroq(temperature=self.llm_temperature, model_name=self.llm_model_name)  

    def get_by_session_id(self, session_id: str) -> BaseChatMessageHistory:
        """
        Get the chat message history for the given session ID.

        Args:
            session_id (str): The session ID.

        Returns:
            BaseChatMessageHistory: The chat message history for the session.
        """
        if session_id not in self.store:
            self.store[session_id] = self.InMemoryHistory()
        return self.store[session_id]

    class InMemoryHistory(BaseChatMessageHistory, BaseModel):
        """
        A chat message history that stores messages in memory.
        """
        messages: List[BaseMessage] = Field(default_factory=list)

        def add_messages(self, messages: List[BaseMessage]) -> None:
            """
            Add messages to the chat message history.

            Args:
                messages (List[BaseMessage]): The messages to add.
            """
            self.messages.extend(messages)

        def clear(self) -> None:
            """
            Clear the chat message history.
            """
            self.messages = []

    def process_pdfs(self, pdf_files: List[str], query: str, history: List[List[str]] = []) -> List[List[str]]:
        """
        Process the uploaded PDF files and answer the given query using the chatbot.

        Args:
            pdf_files (List[str]): The uploaded PDF files.
            query (str): The query to answer.
            history (List[List[str]], optional): The chat history. Defaults to an empty list.

        Returns:
            List[List[str]]: The updated chat history.
        """
        # Display a message indicating that the document(s) are being uploaded and processed
        print("Uploading and processing document(s)...")

        # If the FAISS index has not been created yet, create it and save it to disk
        if self.faiss_index is None:
            chunks = []
            for pdf_file in pdf_files:
                pdf_path = pdf_file.name
                chunks.extend(self.load_and_chunk_data(pdf_path))
            self.faiss_index = FAISS.from_documents(chunks, self.hf)
            self.faiss_index.save_local("faiss_index")

        # Define the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You're a Chatbot."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Given this history: {history} and this context: {context}, answer this question below: {query}. \
                by strictly following these instructions: Answer the question based only on the history and context and nothing else. If asked about unrelated topics, simply provide the relevant information from the context. \
                If you don't know the answer based on history and context provided, ONLY say - I'm sorry, Blaq wants us to only chat about your documents. \
                Reply ONLY in one sentence and keep the answer concise. Do not get chatty"),
        ])

        # Define the retriever and retrieval chain
        output_parser = StrOutputParser()
        retriever = self.faiss_index.as_retriever(search_kwargs={"k": self.vector_store_k})
        #compressor = LLMChainExtractor.from_llm(self.llm_model_name)           #CohereRerank()
        compressor = EmbeddingsFilter(embeddings=self.hf, similarity_threshold=0.76)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        retrieval_chain = (
            {"context": itemgetter("query") | compression_retriever, "query": itemgetter("query"), "history": itemgetter("history")}
            | RunnableParallel({"output": prompt | self.llm | output_parser, "context": itemgetter("context")})
        )

        # Define the retrieval_chain_with_history
        retrieval_chain_with_history = RunnableWithMessageHistory(
            retrieval_chain,
            self.get_by_session_id,
            input_messages_key="query",
            history_messages_key="history",
        )
        result = retrieval_chain_with_history.invoke({"query": query, "history": history}, config={"configurable": {"session_id": "foo"}})
        history.append([query, result["output"]])
        return history, result["output"]

    def load_and_chunk_data(self, pdf_path: str) -> List[str]:
        """
        Load and chunk the data from the given PDF file.

        Args:
            pdf_path (str): The path to the PDF file.

        Returns:
            List[str]: The chunks of text from the PDF file.
        """
        loader = PyMuPDFLoader(pdf_path)
        text_data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False
        )
        chunks = text_splitter.split_documents(text_data)
        return chunks

    def render_file(self, pdf_files: List[str]) -> Image.Image:
        """
        Render the front page of the first uploaded PDF file.

        Args:
            pdf_files (List[str]): The uploaded PDF files.

        Returns:
            Image.Image: The image of the front page.
        """
        # Choose the first PDF file
        pdf_file = pdf_files[0]

        # Convert the front page of the PDF file to an image
        images = convert_from_path(pdf_file.name, first_page=1, last_page=1)
        image = images[0]

        return image

def main():
    chatbot = PDFChatbot('config.yaml')

    pdf_files = []
    while True:
        pdf_path = input("Enter the path to a PDF file (or 'q' to quit): ")
        if pdf_path.lower() == 'q':
            break
        pdf_files.append(open(pdf_path, 'rb'))

    if pdf_files:
        image = chatbot.render_file(pdf_files)
        image.show()

        history = []
        dataset = []
        while True:
            query = input("Type here to ask your PDF (or 'q' to quit): ")
            if query.lower() == 'q':
                break
            history, response = chatbot.process_pdfs(pdf_files, query, history)
            for q, a in history:
                print(f"Q: {q}")
                print(f"A: {a}")

            # Add the query and response to the dataset
            dataset.append({"query": query, "response": response})

        # Convert the dataset to a pandas DataFrame
        df = pd.DataFrame(dataset)

        # Define a custom Giskard model wrapper for the serialization.
        class PDFChatbotModel(Model):
            def model_predict(self, df: pd.DataFrame) -> pd.DataFrame:
                history = []
                responses = []
                for query in df["query"]:
                    _, response = self.model.process_pdfs(pdf_files, query, history)
                    responses.append(response)
                return pd.DataFrame({"response": responses})

        # Wrap the chatbot
        giskard_model = PDFChatbotModel(
            model=chatbot,  # A prediction function that encapsulates all the data pre-processing steps and that could be executed with the dataset used by the scan.
            model_type="text_generation",  # Either regression, classification or text_generation.
            name="PDF Chatbot",  # Optional.
            description="This model answers questions based on uploaded PDF documents",  # Is used to generate prompts during the scan.
            feature_names=["query"]  # Default: all columns of your dataset.
        )

        # Wrap the dataset
        giskard_dataset = Dataset(df)

        # Validate the wrapped model and dataset.
        print(giskard_model.predict(giskard_dataset).prediction)

        # Run the Giskard scan
        results = scan(giskard_model, giskard_dataset)

        # Display the results
        display(results)

        # Generate a test suite
        test_suite = results.generate_test_suite("Test suite generated by scan")

        # Run the test suite
        test_suite.run()

if __name__ == '__main__':
    main()

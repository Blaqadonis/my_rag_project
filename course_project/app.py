import os
import yaml
import logging
from typing import List, Tuple
from uuid import uuid4

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv
from pdf2image import convert_from_path
import gradio as gr

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFChatbot:
    def __init__(self, config_path: str):
        logger.info("Initializing PDFChatbot")
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            self.vector_store_model_name = config['vector_store']['model_name']
            self.vector_store_k = config['vector_store']['k']
            self.llm_model_name = config['llm']['model_name']
            self.llm_temperature = config['llm']['temperature']
            
            self.store = {}
            self.faiss_index = None
            self.hf = SentenceTransformerEmbeddings(model_name=self.vector_store_model_name, model_kwargs={"trust_remote_code": True})
            self.llm = ChatGroq(temperature=self.llm_temperature, model_name=self.llm_model_name)

            # Initialize LangChain tracing
            unique_id = uuid4().hex[:8]
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = f"Blaq's Project PDF Chatbot: all-mpnet-base-v2 and LLama3.1 70b - {unique_id}"

            logger.info("PDFChatbot initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing PDFChatbot: {e}")
            raise

    def get_by_session_id(self, session_id: str) -> BaseChatMessageHistory:
        logger.info(f"Getting chat message history for session {session_id}")
        if session_id not in self.store:
            self.store[session_id] = self.InMemoryHistory()
        return self.store[session_id]

    class InMemoryHistory(BaseChatMessageHistory, BaseModel):
        messages: List[BaseMessage] = Field(default_factory=list)

        def add_messages(self, messages: List[BaseMessage]) -> None:
            self.messages.extend(messages)

        def clear(self) -> None:
            self.messages = []

    def process_pdfs(self, pdf_files: List[str], query: str, history: List[List[str]] = []) -> List[List[str]]:
        logger.info("Processing PDF files")
        try:
            if self.faiss_index is None:
                chunks = []
                for pdf_file in pdf_files:
                    pdf_path = pdf_file.name
                    chunks.extend(self.load_and_chunk_data(pdf_path))
                self.faiss_index = FAISS.from_documents(chunks, self.hf)
                self.faiss_index.save_local("faiss_index")

            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a Chatbot designed to assist with questions about uploaded documents.
                            You are to answer questions based only on the provided history and context. 
                            If the question is unrelated to the documents, reply with: 'I'm sorry, but 🅱🅻🅰🆀 will like us to only chat about your documents.' 
                            If you don't know the answer based on the provided information, reply with: 'I don't know.' 
                            If asked about your emotions or feelings, reply with: 'I'm sorry, but 🅱🅻🅰🆀 will like us to only chat about your documents.'"""),
                MessagesPlaceholder(variable_name="history"),
                ("human", """Given this history: {history} and this context from the documents: {context}, answer the following question: {query}""")
            ])

            retriever = self.faiss_index.as_retriever(search_kwargs={"k": self.vector_store_k})
            compressor = ContextualCompressionRetriever(base_retriever=retriever)
            retrieval_chain = {
                "context": compressor,
                "query": query,
                "history": history
            } | StrOutputParser()

            result = retrieval_chain.invoke({"query": query, "history": history})
            history.append([query, result["output"]])
            return history
        except Exception as e:
            logger.error(f"Error processing PDFs: {e}")
            raise

    def load_and_chunk_data(self, pdf_path: str) -> List[str]:
        logger.info(f"Loading and chunking data from PDF: {pdf_path}")
        try:
            loader = PyMuPDFLoader(pdf_path)
            text_data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=50,
                length_function=len
            )
            chunks = text_splitter.split_documents(text_data)
            return chunks
        except Exception as e:
            logger.error(f"Error loading and chunking data from PDF: {e}")
            raise

    def render_file(self, pdf_files: List[str]) -> gr.Image:
        logger.info("Rendering front page of the first uploaded PDF file")
        try:
            if not pdf_files:
                return None
            pdf_file = pdf_files[0]
            images = convert_from_path(pdf_file.name, first_page=1, last_page=1)
            return images[0]
        except Exception as e:
            logger.error(f"Error rendering front page of the PDF file: {e}")
            raise

    def create_demo(self) -> Tuple[gr.Blocks, gr.Chatbot, gr.Textbox, gr.Button, gr.Files, gr.Image]:
        logger.info("Creating Gradio demo")
        try:
            with gr.Blocks(title="🅱🅻🅰🆀's Chatbot") as demo:
                gr.HTML("<h1>🅱🅻🅰🆀's Chatbot</h1>")
                chat_history = gr.Chatbot(value=[], elem_id='chatbot', height=300)
                query = gr.Textbox(show_label=False, placeholder="Type here to ask your PDF", container=False)
                submit_btn = gr.Button('Send')
                pdf_files = gr.Files(label="📁 Upload PDF", file_types=[".pdf"])
                image_box = gr.Image(label='Your Document:')

                submit_btn.click(self.process_pdfs, inputs=[pdf_files, query, chat_history], outputs=[chat_history])
                pdf_files.change(self.render_file, inputs=[pdf_files], outputs=[image_box])

            return demo, chat_history, query, submit_btn, pdf_files, image_box
        except Exception as e:
            logger.error(f"Error creating Gradio demo: {e}")
            raise

if __name__ == '__main__':
    chatbot = PDFChatbot('config.yaml')
    demo, chat_history, query, submit_btn, pdf_files, image_box = chatbot.create_demo()
    demo.launch()
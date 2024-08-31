import os
import yaml
from typing import List, Tuple
from operator import itemgetter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from PIL import Image
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from pdf2image import convert_from_path
from langfuse.callback import CallbackHandler
import logging
import gradio as gr

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

        # Initialize Langfuse handler
        self.langfuse_handler = CallbackHandler(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST"),
        )

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
    ("system", """You are a Chatbot designed to assist with questions about uploaded documents.
                You are to answer questions based only on the provided history and context. 
                If the question is unrelated to the documents, reply with: 'I'm sorry, but Blaq will like us to only chat about your documents.' 
                If you don't know the answer based on the provided information, reply with: 'I don't know.' 
                If asked about your emotions or feelings, reply with: 'I'm sorry, but Blaq will like us to only chat about your documents.'"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", """Given this history: {history} and this context from the documents: {context}, answer the following question: {query}""")
])


        # Define the retriever and retrieval chain
        output_parser = StrOutputParser()
        retriever = self.faiss_index.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": self.vector_store_k, "score_threshold": 0.5})
        
        retrieval_chain = (
            {"context": itemgetter("query") | retriever, "query": itemgetter("query"), "history": itemgetter("history")}
            | RunnableParallel({"output": prompt | self.llm | output_parser, "context": itemgetter("context")})
        )
        
        logging.getLogger().setLevel(logging.ERROR) # hide warning log

        # Define the retrieval_chain_with_history
        retrieval_chain_with_history = RunnableWithMessageHistory(
            retrieval_chain,
            self.get_by_session_id,
            input_messages_key="query",
            history_messages_key="history",
        )
        result = retrieval_chain_with_history.invoke({"query": query, "history": history}, config={"configurable": {"session_id": "foo"}, "callbacks": [self.langfuse_handler]})
        history.append([query, result["output"]])
        return history

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
        # Check if pdf_files is None
        if pdf_files is None:
            return None

        # Choose the first PDF file
        pdf_file = pdf_files[0]

        # Convert the front page of the PDF file to an image
        images = convert_from_path(pdf_file.name, first_page=1, last_page=1)
        image = images[0]

        return image

    def create_demo(self) -> Tuple[gr.Blocks, gr.Chatbot, gr.Textbox, gr.Button, gr.Files, gr.Image]:
        """
        Create the Gradio demo for the PDF chatbot.

        Returns:
            Tuple[gr.Blocks, gr.Chatbot, gr.Textbox, gr.Button, gr.Files, gr.Image]: The Gradio demo components.
        """
        with gr.Blocks(title="🅱🅻🅰🆀's Chatbot", theme=gr.themes.Glass(primary_hue=gr.themes.colors.red, secondary_hue=gr.themes.colors.pink)) as demo:
            with gr.Column():
                gr.HTML("<h1>🅱🅻🅰🆀's Chatbot</h1>")
                with gr.Row():
                    chat_history = gr.Chatbot(value=[], elem_id='chatbot', height=300)

                with gr.Row():
                    with gr.Column(scale=8):
                        query = gr.Textbox(
                            show_label=False,
                            placeholder="Type here to ask your PDF",
                            container=False
                        )

                    with gr.Column(scale=1):
                        submit_btn = gr.Button('Send')

                    with gr.Column(scale=1):
                        pdf_files = gr.Files(label="📁 Upload PDF", file_types=[".pdf"])

                with gr.Row():
                    image_box = gr.Image(label='Your Document:')

                submit_btn.click(self.process_pdfs, inputs=[pdf_files, query, chat_history], outputs=[chat_history])
                pdf_files.change(self.render_file, inputs=[pdf_files], outputs=[image_box])

        return demo, chat_history, query, submit_btn, pdf_files, image_box

if __name__ == '__main__':
    chatbot = PDFChatbot('config.yaml')
    demo, chat_history, query, submit_btn, pdf_files, image_box = chatbot.create_demo()
    demo.launch()
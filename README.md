# **The RAG Bible**

### Powered by 🅱🅻🅰🆀
-----------------------------------------------------------
![image](https://github.com/Blaqadonis/my_rag_project/assets/100685852/e6f00e40-9f39-4857-b3da-9916aad1d54b)

This project is an AI application that allows you to chat with any PDF document by simply uploading the document to its knowledge base. The chatbot is powered by the LangChain framework and open-source models.
The focus of this project is on the Retrieval Augmented Generation (RAG) technique to create a chatbot capable of learning from an external knowledge base.

## **Features**
In this project, I've implemented more advanced versions of RAG in addition to the basic version. The advanced version includes the following techniques:
* **Contextual Compression**: This technique reduces the number of retrieved documents by compressing them based on their relevance to the query.
* **AI assistants/Agentic workflows**: These techniques use agents and other runnable components in addition to chains. For this purpose, I have used LangGraph.
* **Langfuse**: Langfuse is a tool for debugging and monitoring language models. I've integrated Langfuse into the project to track and analyze the chatbot's performance.
* **Giskard**: Giskard is a tool for testing and evaluating machine learning models. I've used Giskard to evaluate the chatbot's performance on a test suite generated from the uploaded PDF documents.

## **Requirements**
* Python 3.10 or higher
* LangChain library
* Open-source and closed-source models (e.g., HuggingFace, Groq, Openai, Cohere)
* PDF document(s) for the knowledge base
## **Installation**
* Clone the repository:   ```git clone https://github.com/your-username/pdf-chatbot.git```
* ``` cd pdf-chatbot ```
* Install the required dependencies:  ```pip install -r requirements.txt```
* Set up the environment variables by creating a ```.env``` file in the project root directory and adding the following variables:

```
LANGFUSE_SECRET_KEY=<your_langfuse_secret_key>
LANGFUSE_PUBLIC_KEY=<your_langfuse_public_key>
LANGFUSE_HOST=<your_langfuse_host>
```
Get started with Langfuse [here](https://cloud.langfuse.com/?getStarted=1)
## **Usage**
### **Command Line Interface**
**Agentic RAG**: (NB: For this approach, it is hard-coded that you must query [this pdf](https://pressbooks.oer.hawaii.edu/humannutrition2/))
* Run the script: ```python main.py```
* Start chatting. Use ```q``` to quit.
  
**RAG with just chains**:
* Run the script: ```python script.py```
* Upload your PDF document(s) by providing the filepath when prompted.
* Start chatting. Use ```q``` to quit.

### **Web User Interface**
**Gradio**:
* ```cd web_app```
* Run the script: ```python app.py```
* Follow this link for UI: [gradio](http://127.0.0.1:7860/)

## **Configuration**
```config.yaml``` contains the configuration for the vector store and language model used in the chatbot. You can modify the following parameters as you wish:
```
vector_store.model_name: The name of the embedding model used for vectorizing the text data.
vector_store.k: The number of nearest neighbors to retrieve during the RAG process.
llm.model_name: The name of the language model used for generating responses.
llm.temperature: The temperature parameter for the language model.
```
## **Evaluation**
To evaluate the performance of the chatbot, you can use ```evaluation.py```. This script generates a test suite based on the uploaded PDF documents and evaluates the chatbot's responses. 
Get started with Giskard. The results are displayed in the [console](https://docs.giskard.ai/en/latest/getting_started/quickstart/quickstart_llm.html)

## **Contributing**
Contributions are welcome! If you have any suggestions, improvements, or bug fixes, please submit a pull request. For major changes, please open an issue first to discuss the proposed changes.

## **Acknowledgements**
* LangChain
* HuggingFace
* Cohere
* Openai
* Groq
* Ollama
* Giskard
  
## **Contact**
For any questions, inquiries, or further collaboration, please contact [🅱🅻🅰🆀](https://www.linkedin.com/in/chinonsoodiaka/)

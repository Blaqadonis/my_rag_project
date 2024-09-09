# PDFChatbot

![project_architecture](https://github.com/user-attachments/assets/76c3e4e2-4151-4438-b25b-621f97b131bf)


This project is part of the **LLM Zoomcamp 2024** course hosted by DataTalksClub. The aim of this project is to create a chatbot that can assist with questions about uploaded PDF documents using various LangChain components and a Gradio interface.

## Project Structure

course_project/
├── app.py
├── config.yaml
├── Dockerfile
├── knowledgebase_eval.py
├── generator_eval.py
├── requirements.txt
└── README.md




## Overview

PDFChatbot is a powerful tool designed to assist users with extracting and querying information from PDF documents. It leverages state-of-the-art natural language processing models and embeddings to provide accurate and contextually relevant answers.

## Features

- **Document Processing**: Upload and process multiple PDF files.
- **Contextual Question Answering**: Ask questions related to the content of the uploaded documents.
- **History Management**: Maintains the history of the chat sessions.
- **Interactive Interface**: User-friendly interface built with Gradio.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Virtual Environment (optional but recommended)

### Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/Blaqadonis/my_rag_project.git
    cd course_project
    ```

2. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Set up environment variables**:
    - Create a `.env` file and update it with your configuration.
    ```env
    HUGGINGFACE_HUB_API_KEY = '...'
    LANGCHAIN_API_KEY = '...'
    GROQ_API_KEY = '...'
    OPENAI_API_KEY = '...'
    LANGFUSE_SECRET_KEY= 'sk-...'
    LANGFUSE_PUBLIC_KEY= 'pk-...'
    LANGFUSE_HOST= 'https://cloud.langfuse.com'
    ```
    Langfuse and Langsmith basically do the same thing - monitoring. For the project, I used [Langfuse](https://cloud.langfuse.com/project/clwuk3f8o0000bzc5mov0mjsa/traces/350841d0-16a1-47fd-8750-db04d36780f1). But to run evaluations, I used [Langsmith](https://smith.langchain.com/o/f2adffe6-d93b-5c6f-9047-1174f7260035/projects/p/f4ce8c05-5aa0-4bc8-8cc4-bfac3afe2736?timeModel=%7B%22duration%22%3A%227d%22%7D). And I used Openai's ```GPT-4o-mini``` for evaluation. 
    
   **It costs 0 USD to use this chatbot**. However, running the entire project, including the evaluation scripts, costs < 1 USD.

### Configuration

The `config.yaml` file contains all the necessary configuration settings for the chatbot. Update the file with the appropriate values for your models and settings.

```yaml
vector_store:
  model_name: "sentence-transformers/all-mpnet-base-v2"  

llm:
  model_name: "llama-3.1-70b-versatile"    
  temperature: 0
```



Running the Application
=======
## Running the Application

To start the chatbot application, 

**1. Simply run**:

```sh
python app.py
```
This will launch a Gradio interface where you can upload PDF files and interact with the chatbot. Follow this [link](http://localhost:7860/)

**2. Using the Deployed Chatbot**:

You can also interact with the chatbot that has been deployed on Hugging Face Spaces. To start using the deployed chatbot:

Visit the Deployed Chatbot: 
Click on [this link](https://huggingface.co/spaces/Blaqadonis/Blaqs-PDF-Chatbot) to open the chatbot application.

OR

**3. Build the Docker image**:
```sh
docker build -t pdfchatbot .
```

Run the Docker container:

```sh
docker run -it -p 7860:7860 --env-file .env pdfchatbot
```

## Usage
* Upload PDF: Click on the upload button to select one or more PDF files.
* Ask Questions: Type your question in the text box and click send. The chatbot will provide answers based on the content of the uploaded PDFs.
* View Document: The front page of the uploaded PDF will be displayed in the image box.

![is_today_thursday](https://github.com/user-attachments/assets/f399931c-3435-4829-a7b7-bf99ff073093)
![spaces_error](https://github.com/user-attachments/assets/9a9b8e2f-7f9a-47bf-a6c7-c4eb383f7ad7)


## Monitoring
You need an account with Langchain to access [Langsmith](https://smith.langchain.com/o/f2adffe6-d93b-5c6f-9047-1174f7260035/projects/p/3eb74abf-1641-4802-a971-d5d244e6ac86?timeModel=%7B%22duration%22%3A%227d%22%7D)

![monitoring_evaluation](https://github.com/user-attachments/assets/1c1df7c2-45c7-40db-a75c-2e152658288c)
![image](https://github.com/user-attachments/assets/83019e02-ee80-4f9c-bc04-b4db7e7cb07f)
![image](https://github.com/user-attachments/assets/fa738cfe-f325-4fb2-9d02-d67efc8661ee)





## Evaluation
To ensure the chatbot's performance is optimal, an evaluation process has been set up to check the accuracy of the responses and detect any hallucinations. This evaluation generates the following reports:

```ensemble-retriever-report.csv```

```llama-3.1-70b-versatile-evaluation-report.csv```

```model_comparison_results.csv```

```llama3-70b-8192-evaluation-report.csv```

### Running the Evaluation
Run the evaluation scripts:

After processing your PDFs and interacting with the chatbot, you can evaluate the performance of your choice vector database, or the performance of your LLM
by running the following scripts:

``` python knowledgebase_eval.py ```

``` python generator_eval.py ```

These scripts will generate the following CSV reports:

* llama-3.1-70b-versatile-evaluation-report.csv
* llama3-70b-8192-evaluation-report.csv
* model_comparison_results.csv
* ensemble-retriever-report.csv

### Understanding the Reports
* ```llama-3.1-70b-versatile-evaluation-report.csv```: Provides a detailed analysis of the performance of the llama-3.1-70b-versatile model.
* ```llama3-70b-8192-evaluation-report.csv```: Provides a detailed analysis of the performance of the llama3-70b-8192 model.
* ```model_comparison_results.csv```: Highlights the comparison between different models to identify the best-performing one.
* ```ensemble-retriever-report.csv```: Provides a detailed evaluation of the ensemble retriever's performance, comparing its effectiveness with   other retrievers used in the project.

![retriever_evaluation](https://github.com/user-attachments/assets/ff47e677-4b20-4ed4-ba93-39b08ec87c8f)
![generator_evaluation](https://github.com/user-attachments/assets/94244ca7-ff1a-4961-a13c-444bd293004e)
![image](https://github.com/user-attachments/assets/a44c4cea-3bba-4116-bb58-52219e0dc984)





### Additional Notes
- Waiting for the First Query: The Time Taken to First Token (TTFT) for this application is quite high, depending on your computer's specifications. Therefore, you may have to wait a while for the first query to get answered. However, it is smooth-sailing after that.
- Due to a Poppler dependency issue peculiar to Hugging Face Spaces, the first page of the PDF may not show on Spaces. However, it will display correctly locally if you have Poppler installed. Please ignore the warning and proceed with writing your query.

## Contributing
Contributions to enhance the PDFChatbot project are welcome. Please follow these steps to contribute:

* Fork the repository
* Create a new branch (git checkout -b feature-branch)
* Make your changes
* Commit your changes (git commit -m 'Add some feature')
* Push to the branch (git push origin feature-branch)
* Open a pull request

## License
This project is licensed under the MIT License

## Acknowledgments
I want to acknowledge [Alexey Grigorev](https://www.linkedin.com/in/agrigorev/), founder [DataTalksClub](https://datatalks.club/) - for providing the platform to learn and grow.

Feel free to reach out if you have any questions or need further assistance - [🅱🅻🅰🆀](https://www.linkedin.com/in/chinonsoodiaka/)

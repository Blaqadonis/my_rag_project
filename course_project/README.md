# PDFChatbot

![project_architecture](https://github.com/user-attachments/assets/76c3e4e2-4151-4438-b25b-621f97b131bf)


This project is part of the **LLM Zoomcamp 2024** course hosted by DataTalksClub. The aim of this project is to create a chatbot that can assist with questions about uploaded PDF documents using various LangChain components and a Gradio interface.

## Project Structure

course_project/
├── app.py
├── config.yaml
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
    git clone https://github.com/yourusername/PDFChatbot.git
    cd PDFChatbot
    ```

2. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Set up environment variables**:
    - Create a `.env` file and update it with your configuration.
    ```
    HUGGINGFACE_HUB_API_KEY = '...'
    LANGCHAIN_API_KEY = '...'
    GROQ_API_KEY = '...'
    LANGFUSE_SECRET_KEY= 'sk-...'
    LANGFUSE_PUBLIC_KEY= 'pk-...'
    LANGFUSE_HOST= 'https://cloud.langfuse.com'

    ```

### Configuration

The `config.yaml` file contains all the necessary configuration settings for the chatbot. Update the file with the appropriate values for your models and settings.

```yaml
vector_store:
  model_name: "all-mpnet-base-v2"
  k: 5

llm:
  model_name: "LLama3.1"
  temperature: 0.7
```

## Running the Application
To start the chatbot application, simply run:

```sh
python app.py
```
This will launch a Gradio interface where you can upload PDF files and interact with the chatbot.

## Usage
* Upload PDF: Click on the upload button to select one or more PDF files.
* Ask Questions: Type your question in the text box and click send. The chatbot will provide answers based on the content of the uploaded PDFs.
* View Document: The front page of the uploaded PDF will be displayed in the image box.


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

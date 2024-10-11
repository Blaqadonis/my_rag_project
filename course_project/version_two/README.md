# **PDFChatbot v2.0**

![talkblaq](https://github.com/user-attachments/assets/ea385113-a5e2-4dac-9e41-ecce3dc87a08)



## Overview


* This project is an upgraded version of PDFChatbot, now incorporating advanced techniques and stateful architecture to significantly improve its performance and 
  flexibility.

* Key Changes
  ColBERT Indexing Technique:

  Implemented ColBERT for the indexing stage, which improves the chatbot's ability to handle a broader range of PDF formats, including those with illustrations.
  This is a major improvement over the previous approach using cosine similarity and chunk-level embeddings, which could lose context and reduce accuracy.

* Redesigned with LangGraph:

  The entire app has been restructured using LangGraph, offering a more flexible and stateful architecture.
  The previous rigid chain system has been replaced, enabling the use of loops and states to dynamically define the app's behavior, making it more customizable and 
  adaptable.
  
## What's Next

UI deployment via LangGraph Studio for seamless user interaction.

## Future Work

Multimodal integration (ColPali).

Stay tuned for more updates as the project progresses!

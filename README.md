---
title: Chat with any Data App
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: 3.39.0
app_file: app.py
pinned: false
license: MIT
---

# Chat with any data app
Chatbot app to chat with any source of data (doc, url, ...) leveraging LLMs, LangChain, and Gradio. Current version has the following feautures:
- LLM: Llama2
- Data source: "folder", "csv", "doc", "docx", "epub", "html", "md", "pdf", "ppt", "pptx", "txt", "ipynb", "py", and "url".

This app has been inspired by a few DeepLearning AI courses about LangChain.

## Installation and execution
You should have already installed Ollama in your computer. Then, install this app in a python environment:
```
    conda create --name chat-with-any-data-app python=3.11 --yes
    conda activate chat-with-any-data-app
    cd chat_with_any_data_app-main
    pip install -r requirements.txt
    python app.py
```
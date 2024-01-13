# Description: Run QA on your data.
#
# Read and process data:
#  - Read data from a file or a folder 
#  - Split data into chunks
#  - Create a vector database from data
# 
# Find relevant documents:
#  - Prompt the user to instruct a question
#  - Find relevant documents
#  - Reduce documents
#
# QA Chain:
#  - Prompt the user to instruct a question
#  - Run QA chain

import os

from utils.in_out_helper import load_config
from utils.app_helper import chat_to_your_data_ui

# Working directory
print(f"Working directory: {os.getcwd()}")

# Import YAML parameters from config/config.yaml
config_file = "config/config.yaml"

# Load config file 
param = load_config(config_file)

def main(param):
    # PARAMETERS
    
    # LLM
    temperature = param['llm']["temperature"]
    llm_name = param['llm']["llm_name"]
    # Document
    doc_type = param['doc']["doc_type"]
    doc_path = param['doc']["doc_path"]
    folder_path = param['doc']["folder_path"]
    chunk_overlap = param['doc']["chunk_overlap"]
    chunk_size = param['doc']["chunk_size"]
    # Database
    persist_path = param['db']["persist_path"]
    mode_input = param['db']["mode_input"]
    similarit_on = param['db']["similarit_on"]
    question_k = param['db']["question_k"]
    # Chat
    qa_on = param['chat']["qa_on"]
    qa_chain_type = param['chat']["qa_chain_type"]
    chat_examples = param['chat']["chat_examples"]
    chat_description = param['chat']["chat_description"]
    share_gradio = param['chat']["share_gradio"]

    # Interface to chat with your data
    chat_to_your_data_ui(doc_type, doc_path, folder_path, chunk_size, chunk_overlap,
                         llm_name, temperature, share_gradio)

if __name__ == "__main__":
    main(param)
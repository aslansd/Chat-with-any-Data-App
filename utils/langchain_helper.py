import copy
import glob
from tqdm import tqdm

from langchain_community.document_loaders import (
    CSVLoader,
    NotebookLoader,
    PyPDFium2Loader,
    PythonLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
)

from langchain_community.document_loaders.merge import MergedDataLoader

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch

# Load document with langchain.document_loaders
def read_doc(doc_type, doc_path, mode_print=False):
    if doc_type == "csv":
        loader = CSVLoader(doc_path)
    elif doc_type == "doc":
        loader = UnstructuredWordDocumentLoader(doc_path)
    elif doc_type == "docx":
        loader = UnstructuredWordDocumentLoader(doc_path)
    elif doc_type == "epub":
        loader = UnstructuredEPubLoader(doc_path)
    elif doc_type == "html":
        loader = UnstructuredHTMLLoader(doc_path)
    elif doc_type == "md":
        loader = UnstructuredMarkdownLoader(doc_path)
    elif doc_type == "pdf":
        loader = PyPDFium2Loader(doc_path)
    elif doc_type == "ppt":
        loader = UnstructuredPowerPointLoader(doc_path)
    elif doc_type == "pptx":
        loader = UnstructuredPowerPointLoader(doc_path)
    elif doc_type == "txt":
        loader = TextLoader(doc_path)
    elif doc_type == "ipynb":
        loader = NotebookLoader(doc_path)
    elif doc_type == "py":
        loader = PythonLoader(doc_path)
    elif doc_type == "url":
        loader = WebBaseLoader(doc_path)

    docs = loader.load()
    
    if mode_print is True:
        print(f"Loaded {len(docs)} pages/documents")
        print(f"First page: {docs[0].metadata}")
        print(docs[0].page_content[:500])
    
    return docs

def read_directory(path: str):
    all_files = glob.glob(path + '/*.*')

    with tqdm(total=len(all_files), desc="Loading documents", ncols=80) as pbar:
        counter = -1
        for file in all_files:
            counter += 1

            if file.endswith("csv"):
                loader = CSVLoader(all_files[counter])
            elif file.endswith("doc"):
                loader = UnstructuredWordDocumentLoader(all_files[counter])
            elif file.endswith("docx"):
                loader = UnstructuredWordDocumentLoader(all_files[counter])
            elif file.endswith("epub"):
                loader = UnstructuredEPubLoader(all_files[counter])
            elif file.endswith("html"):
                loader = UnstructuredHTMLLoader(all_files[counter])
            elif file.endswith("md"):
                loader = UnstructuredMarkdownLoader(all_files[counter])
            elif file.endswith("pdf"):
                loader = PyPDFium2Loader(all_files[counter])
            elif file.endswith("ppt"):
                loader = UnstructuredPowerPointLoader(all_files[counter])
            elif file.endswith("pptx"):
                loader = UnstructuredPowerPointLoader(all_files[counter])
            elif file.endswith("txt"):
                loader = TextLoader(all_files[counter])
            elif file.endswith("ipynb"):
                loader = NotebookLoader(all_files[counter])
            elif file.endswith("py"):
                loader = PythonLoader(all_files[counter])

            if counter == 0:
                loader_all = copy.copy(loader)
            elif counter > 0:
                loader_all = MergedDataLoader(loaders=[loader_all, loader])

    docs = loader_all.load()

    return docs

def read_split_doc(doc_type, doc_path, chunk_size, chunk_overlap, mode_print=False):
    global flag_file
    # Read doc using langchain
    if doc_type in ["csv", "doc", "docx", "epub", "html", "md", "pdf", "ppt", "pptx", "txt", "ipynb", "py", "url"]:
        flag_file = True
        docs = read_doc(doc_type, doc_path)
    else:
        flag_file = False
        docs = read_directory(doc_path)
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap)

    docs_split = text_splitter.split_documents(docs)
    
    if mode_print is True:
        print(f"Split into {len(docs_split)} chunks")
        print(f"First chunk: {docs_split[0].metadata}")
        print(docs_split[0].page_content)
    
    return docs_split

def create_db_from_documents(docs_split, embedding):
    # Create vector database from document    
    db = DocArrayInMemorySearch.from_documents(docs_split, embedding=embedding)
    # print(f"Created vector database with {db._collection.count()} documents")
    # db.persist() # save the vectorstore to disk for future use     
    
    return db

def init_embedding():
    # Define embedding
    embedding = HuggingFaceEmbeddings() 
    
    return embedding

def init_llm(llm_name, temperature):
    # Init the LLM
    llm = Ollama(model=llm_name, temperature=temperature)
    
    return llm

def init_qa_chain_from_llm(llm, db):
    # Init memory and chain
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=db.as_retriever(),
        memory=memory
    )
    
    return qa_chain, memory

def init_llm_qa_chain(llm_name, temperature, db):
    # Init LLM and QA chain so it can be modified from the chat interface
    # Init the LLM
    llm = init_llm(llm_name, temperature)
    # Init memory and chain
    qa_chain, memory = init_qa_chain_from_llm(llm, db)
    
    return llm, qa_chain, memory

def qa_call(llm, db, input):
    output = qa_chain({"question": input})
    
    return output

def qa_answer(input):
    # Return the answer from the QA call
    
    return qa_call(input)['answer']

def qa_input_msg_history(input, history):
    # QA function that inputs the answer and the history
    # History managed internally by ChatInterface
    answer = qa_answer(input)
    
    return answer
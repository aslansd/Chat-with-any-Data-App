import gradio as gr
from utils.langchain_helper import init_embedding, read_split_doc, create_db_from_documents, init_llm_qa_chain

# ----------------------------------------------------------------------------
# Gradio Interface
def chat_to_your_data_ui(doc_type, doc_path, folder_path, chunk_size, chunk_overlap,
                         llm_name, temperature, share_gradio):    
    # ----------------------------------------------------------------------------
    # Interface functionality
    # ------------------------
    # When read the document
    def reading_doc_msg(doc_type, doc_path):
        return f"Reading document {doc_path} of type {doc_type} ..."
    def read_doc_msg():
        return "Finished reading the document! Let's chat!"
    def clear_chatbot_after_read_doc():            
        return "", ""
    # -------------------------
    # Init the LLM and read document
    def init_read_doc(doc_type, doc_path, chunk_size, chunk_overlap, temperature):
        global qa_chain_global, db_global
        # Init embedding
        embedding = init_embedding()

        # Read and split document using langchain
        print(f"Reading document {doc_path} of type {doc_type} ...")
        docs_split = read_split_doc(doc_type, doc_path, chunk_size, chunk_overlap)
        # -------------------------
        # Create vector database from data
        db = create_db_from_documents(docs_split, embedding)
        # -------------------------
        # Init the LLM and qa chain
        llm, qa_chain, memory = init_llm_qa_chain(llm_name, temperature, db) 

        qa_chain_global = qa_chain
        db_global = db          

    # When question 
    def qa_input_msg_history(question, chat_history):
        # QA function that inputs the answer and the history
        # History managed internally by ChatInterface      
        answer = qa_chain_global({"question": question})['answer']
        # response = qa_chain({"question": input})
        chat_history.append((question, answer))

        return "", chat_history    
    
    # When clear all (document, chatbot)
    def clear_all():
        global qa_chain_global, db_global
        qa_chain_global = None
        db_global = None
        
        return "Document cleared!", "Document cleared!", "", "", "", ""

    # ----------------------------------------------------------------------------
    # UI
    with gr.Blocks(theme=gr.themes.Glass()) as demo:
        # Description            
        gr.Markdown(
        """
        # Chat with any data with an open source LLM
        Ask questions to the chatbot about your document. The chatbot will find the answer to your question. 
        You can modify the document type and provide its path/link.
        You may also modify some of the advanced options.
        """)
        # -------------------------
        # Parameters and chatbot image
        with gr.Row():
            with gr.Column(scale=2):
                # -------------------------
                # Parameters
                # Temperature and document type
                gr.Markdown(
                """
                ## Select parameters
                Default parameters are already provided.
                """
                )
                # Advanced parameters (hidden)
                with gr.Accordion(label="Advanced options", open=False):
                    gr.Markdown(
                    """
                    The document is split into chunks, keeping semantically related pieces together and with some overlap. 
                    You can modify the chunk size and overlap. The temperature is used to control the randomness of the output.
                    (The lower the temperature the more deterministic the ouput, the higher its value the more random the result, with $temperature\in[0,1]$).
                    """
                    )        
                    sl_temperature = gr.Slider(minimum=0.0, maximum=1.0, value=temperature, label="Temperature", scale=2)
                    with gr.Row():
                        num_chunk_size = gr.Number(value=chunk_size, label="Chunk size", scale=1)
                        num_chunk_overlap = gr.Number(value=chunk_overlap, label="Chunk overlap", scale=1)

        # -------------------------
        # Select and read a document
        gr.Markdown(
        """
        ## Select a document
        Select the document type and provide its path/link (e.g. https://sites.google.com/a/umich.edu/aslansdizaji/home).
        """)
        with gr.Row():
            drop_type_1 = gr.Dropdown(["csv", "doc", "docx", "epub", "html", "md", "pdf", "ppt", "pptx", "txt", "ipynb", "py", "url"],
                                      label="Document Type", value=doc_type, min_width=30, scale=1)
            text_path_1 = gr.Textbox(label="Document Path/URL", placeholder=doc_path, scale=5)
        
        with gr.Row():
            # Read a document
            btn_read_1 = gr.Button("Read Document")
            text_read_output_1 = gr.Textbox(label="Reading State", interactive=False, placeholder="Select the document type and its path!")

        # -------------------------
        # Select and read a folder
        gr.Markdown(
        """
        ## Select a folder
        Provide a path for a folder (e.g. /Users/asataryd/Desktop/Relevant Papers/Interesting Papers).
        """)
        with gr.Row():
            drop_type_2 = gr.Dropdown(["folder"], label="Document Type", value="folder", min_width=30, scale=1)
            text_path_2 = gr.Textbox(label="Folder Path", placeholder=folder_path, scale=5)
        
        with gr.Row():
            # Read a folder
            btn_read_2 = gr.Button("Read Folder")
            text_read_output_2 = gr.Textbox(label="Reading State", interactive=False, placeholder="Select the folder's path!")

        # -------------------------
        # Chatbot
        gr.Markdown("""
        ## Chatbot  
        To chat, introduce a question and press enter.
                    
        Question examples:
                    
         - Hello!
                    
         - What is this document about?
                    
         - Who is Aslan?        
        """
        )
        # Chatbot
        chatbot = gr.Chatbot()
        
        # Input message
        msg = gr.Textbox(label="Question")
        
        # Clear button
        clear = gr.Button("Clear all (document, chatbot)")

        # Init the LLM and read the document/folder with default parameters
        # -------------------------
        # When reading the document (aready read with default parameters)
        btn_read_1.click(reading_doc_msg,                                                               # Reading message 
                            inputs=[drop_type_1, text_path_1], 
                            outputs=text_read_output_1).then(init_read_doc,                             # Init qa chain and read document
                                inputs=[drop_type_1, text_path_1, 
                                        num_chunk_size, num_chunk_overlap,
                                        sl_temperature], 
                                queue=False).then(read_doc_msg,                                         # Finished reading message
                                        outputs=text_read_output_1).then(clear_chatbot_after_read_doc,  # Clear chatbot
                                                outputs=[chatbot, msg], queue=False)
        
        # When reading the folder (aready read with default parameters)
        btn_read_2.click(reading_doc_msg,                                                               # Reading message 
                            inputs=[drop_type_2, text_path_2], 
                            outputs=text_read_output_2).then(init_read_doc,                             # Init qa chain and read document
                                inputs=[drop_type_2, text_path_2, 
                                        num_chunk_size, num_chunk_overlap,
                                        sl_temperature],
                                queue=False).then(read_doc_msg,                                         # Finished reading message
                                        outputs=text_read_output_2).then(clear_chatbot_after_read_doc,  # Clear chatbot
                                                outputs=[chatbot, msg], queue=False) 
        # -------------------------
        # When questioning
        msg.submit(qa_input_msg_history, 
                   inputs=[msg, chatbot], 
                   outputs=[msg, chatbot], queue=False)#.then(bot, chatbot, chatbot)
        
        # -------------------------
        # When clearing
        clear.click(clear_all, 
                    outputs=[text_read_output_1, text_read_output_2, 
                             chatbot, msg, text_path_1, text_path_2], queue=False)
    
    # demo.queue() # To use generator, required for streaming intermediate outputs
    demo.launch(share=share_gradio)
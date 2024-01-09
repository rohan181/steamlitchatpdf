import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub

from transformers import AutoTokenizer
import transformers
import torch
import tqdm 
import accelerate

default_persist_directory = './chroma_HF/'

llm_name1 = "mistralai/Mistral-7B-Instruct-v0.2"
llm_name2 = "mistralai/Mistral-7B-Instruct-v0.1"
llm_name3 = "meta-llama/Llama-2-7b-chat-hf"
llm_name4 = "microsoft/phi-2"
llm_name5 = "mosaicml/mpt-7b-instruct"
llm_name6 = "tiiuae/falcon-7b-instruct"
llm_name7 = "google/flan-t5-xxl"
list_llm = [llm_name1, llm_name2, llm_name3, llm_name4, llm_name5, llm_name6, llm_name7]
list_llm_simple = [os.path.basename(llm) for llm in list_llm]



 Load PDF document and create doc splits
def load_doc(list_file_path, chunk_size, chunk_overlap):
    # Processing for one document only
    # loader = PyPDFLoader(file_path)
    # pages = loader.load()
    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap = 50)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, 
        chunk_overlap = chunk_overlap)
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits


# Create vector database
def create_db(splits):
    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=default_persist_directory
    )
    return vectordb


# Load vector database
def load_db():
    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma(
        persist_directory=default_persist_directory, 
        embedding_function=embedding)
    return vectordb


# Initialize langchain LLM chain
def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    progress(0.1, desc="Initializing HF tokenizer...")
    # HuggingFacePipeline uses local model
    # Warning: it will download model locally...
    # tokenizer=AutoTokenizer.from_pretrained(llm_model)
    # progress(0.5, desc="Initializing HF pipeline...")
    # pipeline=transformers.pipeline(
    #     "text-generation",
    #     model=llm_model,
    #     tokenizer=tokenizer,
    #     torch_dtype=torch.bfloat16,
    #     trust_remote_code=True,
    #     device_map="auto",
    #     # max_length=1024,
    #     max_new_tokens=max_tokens,
    #     do_sample=True,
    #     top_k=top_k,
    #     num_return_sequences=1,
    #     eos_token_id=tokenizer.eos_token_id
    #     )
    # llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': temperature})
    
    # HuggingFaceHub uses HF inference endpoints
    progress(0.5, desc="Initializing HF Hub...")
    llm = HuggingFaceHub(
        repo_id=llm_model, 
        model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens, "top_k": top_k,\
        "trust_remote_code": True, "torch_dtype": "auto"}
    )
    
    progress(0.75, desc="Defining buffer memory...")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )
    # retriever=vector_db.as_retriever(search_type="similarity", search_kwargs={'k': 3})
    retriever=vector_db.as_retriever()
    progress(0.8, desc="Defining retrieval chain...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff", 
        memory=memory,
        # combine_docs_chain_kwargs={"prompt": your_prompt})
        return_source_documents=True,
        # return_generated_question=True,
        # verbose=True,
    )
    progress(0.9, desc="Done!")
    return qa_chain


# Initialize database
def initialize_database(list_file_obj, chunk_size, chunk_overlap, progress=gr.Progress()):
    # Create list of documents (when valid)
    #file_path = file_obj.name
    list_file_path = [x.name for x in list_file_obj if x is not None]
    # print('list_file_path', list_file_path)
    progress(0.25, desc="Loading document...")
    # Load document and create splits
    doc_splits = load_doc(list_file_path, chunk_size, chunk_overlap)
    # Create or load Vector database
    progress(0.5, desc="Generating vector database...")
    # global vector_db
    vector_db = create_db(doc_splits)
    progress(0.9, desc="Done!")
    return vector_db, "Complete!"


def initialize_LLM(llm_option, llm_temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    # print("llm_option",llm_option)
    llm_name = list_llm[llm_option]
    # print("llm_name",llm_name)
    qa_chain = initialize_llmchain(llm_name, llm_temperature, max_tokens, top_k, vector_db, progress)
    return qa_chain, "Complete!"


def format_chat_history(message, chat_history):
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history
    

def conversation(qa_chain, message, history):
    formatted_chat_history = format_chat_history(message, history)
    #print("formatted_chat_history",formatted_chat_history)
   
    # Generate response using QA chain
    response = qa_chain({"question": message, "chat_history": formatted_chat_history})
    response_answer = response["answer"]
    response_sources = response["source_documents"]
    response_source1 = response_sources[0].page_content.strip()
    response_source2 = response_sources[1].page_content.strip()
    # Langchain sources are zero-based
    response_source1_page = response_sources[0].metadata["page"] + 1
    response_source2_page = response_sources[1].metadata["page"] + 1
    # print ('chat response: ', response_answer)
    # print('DB source', response_sources)
    
    # Append user message and response to chat history
    new_history = history + [(message, response_answer)]
    # return gr.update(value=""), new_history, response_sources[0], response_sources[1] 
    return qa_chain, gr.update(value=""), new_history, response_source1, response_source1_page, response_source2, response_source2_page
    

def upload_file(file_obj):
    list_file_path = []
    for idx, file in enumerate(file_obj):
        file_path = file_obj.name
        list_file_path.append(file_path)
    # print(file_path)
    # initialize_database(file_path, progress)
    return list_file_path





def main():
    st.title("PDF-based chatbot (powered by LangChain and open-source LLMs)")
    st.markdown("""
        ## Ask any questions about your PDF documents, along with follow-ups
        **Note:** This AI assistant performs retrieval-augmented generation from your PDF documents. 
        When generating answers, it takes past questions into account (via conversational memory), 
        and includes document references for clarity purposes.
        \n**Warning:** This space uses the free CPU Basic hardware from Hugging Face. Some steps and LLM models used below (free inference endpoints) can take some time to generate an output.
    """)

    # Step 1 - Document pre-processing
    st.header("Step 1 - Document pre-processing")
    uploaded_files = st.file_uploader("Upload your PDF documents (single or multiple)", type="pdf", accept_multiple_files=True)
    db_btn = st.radio("Vector database type", ["ChromaDB"])

    st.slider("Chunk size", 100, 1000, 600, 20, key="chunk_size")
    st.slider("Chunk overlap", 10, 200, 40, 10, key="chunk_overlap")
    
    if st.button("Generating vector database..."):
        # Call your initialization function here using uploaded_files, chunk_size, chunk_overlap

    # Step 2 - QA chain initialization
    st.header("Step 2 - QA chain initialization")
    llm_option = st.radio("LLM models", list_llm_simple)
    st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, key="llm_temperature")
    st.slider("Max Tokens", 224, 4096, 1024, 32, key="max_tokens")
    st.slider("Top-k samples", 1, 10, 3, 1, key="top_k")

    if st.button("Initializing question-answering chain..."):
        # Call your initialization function here using llm_option, llm_temperature, max_tokens, top_k, vector_db

    # Step 3 - Conversation with chatbot
    st.header("Step 3 - Conversation with chatbot")
    msg = st.text_input("Type message", key="message")
    if st.button("Submit"):
        # Call your conversation function here using qa_chain, msg, chatbot, and update UI accordingly

if __name__ == "__main__":
    main()
from langchain import PromptTemplate, LLMChain
from langchain.llms import CTransformers
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from io import BytesIO
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import GPT4AllEmbeddings
import streamlit as st
from pathlib import Path
import os
from langchain import hub
import streamlit as st
from langchain.llms import Ollama

from langchain.chat_models import ChatOpenAI
# os.environ["OPENAI_API_KEY"] = ''
# import lib
from PyPDF2 import PdfWriter

from PyPDF2 import PdfReader
import base64
QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")


def load_llm():
	llm = Ollama(
	model="zephyr",
	verbose=True,	
	)
	return llm

@st.cache_resource
def prepare_llm(chosen_resource,chosen_file):

    llm = load_llm()

    
    embeddings=GPT4AllEmbeddings()
    
    
    load_vector_store = Chroma(persist_directory=f"vectorstores/{chosen_resource}/{chosen_file}", 
                               embedding_function=embeddings)
    retriever = load_vector_store.as_retriever(search_kwargs={"k":2})

    return llm,retriever


    
def get_response(input):
    prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    Question: {question}
    
    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    
    chain_type_kwargs = {"prompt": prompt}


    
    query = input
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", 
                                   retriever=retriever, return_source_documents=True, 
                                   chain_type_kwargs=chain_type_kwargs, verbose=True)
    response = qa(query)
    return response



resource_folders=os.listdir("vectorstores")
if ".DS_Store" in resource_folders:
    resource_folders.remove(".DS_Store")

st.title("PalestineBot - Your Personal Chatbot for all things Palestine")
    
for folder in resource_folders:
    print(folder)
chosen_resource=st.selectbox("Choose consultation resource", resource_folders, index=0)

document_files=os.listdir(f"vectorstores/{chosen_resource}")
if ".DS_Store" in document_files:
    document_files.remove(".DS_Store")
    


    
for file in document_files:
    print(file)
    
chosen_file=st.selectbox("Choose consultation file", document_files, index=0)    

llm,retriever=prepare_llm(chosen_resource,chosen_file)    



st.subheader("I have studied the document")
st.subheader(chosen_file)
st.subheader("From: "+chosen_resource)
st.header("Ask me anything about it")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        
response=None
# React to user input
if prompt := st.chat_input("What is your query?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})    
    
    
    resp=get_response(prompt)    
    print("response is ",resp)
    reference=resp["source_documents"][0].page_content
    reference=reference.replace("\n"," ")
    page=""
    if page in resp["source_documents"][0].metadata:
        page=resp["source_documents"][0].metadata["page"]
    file=""
    if "source" in resp["source_documents"][0].metadata:
        file=resp["source_documents"][0].metadata["source"]
    
    response = f"{resp['result']}\n\n *Page: {page}\n\n *File:{file}"
#     response=prompt
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

        

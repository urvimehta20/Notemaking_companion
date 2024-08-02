import os
import time
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.chroma import Chroma
import streamlit as st 
import base64

st.set_page_config(page_title="Your Python Companion", page_icon=":heart:")

# Loading the API key from the .env file
load_dotenv("apikey.env")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# HUGGINGFACEHUB_API_TOKEN = st.secrets['HUGGINGFACEHUB_API_TOKEN']

# Defining the LLM prompt template which contains the instructions also the context is passed in this template itself
prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, don't try to make up an answer. Politely say you don't know the answer.
Context: {context}
Question: {question}
Only return the helpful answer. Be descriptive in your answers.
Helpful answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["question", "context"])

# Cache the HuggingFaceEmbeddings function
@st.cache_resource
def load_embeddings():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

# Cache the vector store where our preprocessed data is stored
@st.cache_resource
def load_vector_store(_embeddings):
    embeddings = _embeddings
    return Chroma(persist_directory="stores/retrieval_data", embedding_function=embeddings)

# Cache the LlamaCpp model which is the locally saved quantized generation model of the RAG 
@st.cache_resource
def load_llm_model():
    model_path = r"D:\Notemaking_companion\models\gemma-1.1-2b-it.Q2_K.gguf"
    return LlamaCpp(model_path=model_path, temperature=0.2, max_tokens=5000, verbose=False, n_ctx=2048)

# Creating the QA chain
@st.cache_resource
def qa_chain():
    embeddings = load_embeddings()
    vector_store = load_vector_store(embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = load_llm_model()
    
    chain_type_kwargs = {"prompt": prompt}
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )

qa = qa_chain()

# Function to set the background image on streamlit
def set_bg_hack_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("https://cdn5.vectorstock.com/i/1000x1000/36/19/pattern-with-a-rose-flamingo-on-blue-background-vector-22743619.jpg");
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to convert PDF to base64 for displaying it on the sidebar
@st.cache_data
def get_pdf_base64(pdf_path):
    with open(pdf_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# Function to stream the final generated data
def stream_data(response):
    for word in response.split():
        yield word + ' '

### Streamlit app
def main():
    st.title("Python Note-taking Companion")

    # st.subheader(" :blue[Ask any Python-related question, and we'll provide you with helpful and accurate answers!] :sunglasses:")
    text_query = st.text_area(" ", placeholder="Type your python related questions here...")
    generate_response_btn = st.button("Ask")

    set_bg_hack_url()

    with st.sidebar:
        pdf_path = "rag_data/PythonNotesForProfessionals.pdf"
        base64_pdf = get_pdf_base64(pdf_path)
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

    if generate_response_btn and text_query:
        # start_time = time.time()
        response_container = st.empty()
        
        response = qa(text_query)
        # end_time = time.time()
        # st.write(f":green-background[Time Taken: {round(end_time - start_time, 2)} seconds]")
        st.divider()

        if response:
            response_text = response['result']
            stream = stream_data(response_text)
            streamed_text = ""
            for word in stream:
                streamed_text += word
                response_container.markdown(streamed_text)
                time.sleep(0.3) 
            for doc in response['source_documents']:
                    st.markdown(f":green-background[For more information, see page {doc.metadata.get('page', 'N/A')} in the source document.]")
        else:
            st.error("Oops! failed to generate response.")

if __name__ == "__main__":
    main()

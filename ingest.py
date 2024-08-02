from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# defining the open-source embedding model and its configuration 
model_name="sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
    )

# Loading the data from the directory where it is saved
loader = DirectoryLoader('rag_data/', glob="**/*.pdf",
                         show_progress=True,
                         use_multithreading=True,
                         loader_cls=PyPDFLoader)

print("\nPlease wait while the documents are loading..")
documents = loader.load()
print(f"Loaded Docs:{len(documents)}")


text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
texts = text_splitter.split_documents(documents)
print("\nText Splitted Successfully!")


vector_store = Chroma.from_documents(texts, embeddings,
                                     collection_metadata={"hnsw:space": "cosine"},
                                     persist_directory="stores/retrieval_data")

print("VECTOR DB Successfully Created!")
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import HF_REPO, CHROMA_DIR,TIMEOUT, SYLLABUS_PATH
from langchain_community.document_loaders import PyPDFLoader

#Document Load
def load_chunks(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

#generate embedding
def get_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)

#vector store
def get_vector_store(embedding_fn, collection="sample"):
    return Chroma(
        embedding_function=embedding_fn,
        persist_directory=CHROMA_DIR,
        collection_name=collection,
    )

def ingest_documents(vector_store, docs):
    vector_store.add_documents(docs)
    vector_store.persist()


#LLM setup
llm = HuggingFaceEndpoint(repo_id=HF_REPO, task="text_generation", timeout=TIMEOUT)
model = ChatHuggingFace(llm=llm)

#Prompt template
template = """You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])
parser = StrOutputParser()

#Build chain
chain = prompt | model | parser

#Initialize vector store
def initialize():
    docs = load_chunks(SYLLABUS_PATH)
    embedder = get_embedding_model()
    store = get_vector_store(embedder)
    ingest_documents(store, docs)
    return store

# Keep a global store
_store = initialize()

def answer(question, k = 4):
    retriever = _store.as_retriever(search="similarity", search_kwargs={"k": k})
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    return chain.invoke({"context": context, "question": question})

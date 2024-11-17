import streamlit as st
import os
from googlesearch import search
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader 
from langchain_community.vectorstores import chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter


def fetch_top_search_results(query, num_results=10):
    search_results = search(query, num_results=num_results)
    return search_results


# --- LangChain Model and Data Loading ---
def load_and_process_data(data_source_type, data_sources):
    if data_source_type == "URLs":
        return process_urls(data_sources)
    elif data_source_type == "Data Folder":
        return process_data_folder()
    else:
        raise ValueError("Invalid data source type. Choose 'URLs' or 'Data Folder'")
    

def process_urls(urls):
    docs = [WebBaseLoader(url).load() for url in urls]
    return create_vectorstore(docs, "url_docs")

def process_data_folder():
    data_folder = "data"  # Assuming data folder is named "data"
    loader = DirectoryLoader(data_folder)
    docs = loader.load()
    documents = loader.load()
    embedding_model = embeddings.OllamaEmbeddings(model='nomic-embed-text')
    vector_store = chroma.Chroma.from_documents(documents, embedding_model)
    return vector_store.as_retriever()

def create_vectorstore(docs, collection_name):
    doc_list = [item for sublist in docs for item in sublist]
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(doc_list)
    vectorstore = chroma.Chroma.from_documents(
        documents=doc_splits,
        collection_name=collection_name,  
        embedding=embeddings.OllamaEmbeddings(model='nomic-embed-text'),    
    )
    return vectorstore.as_retriever()

def create_conversational_pipeline(retriever):
    template = """Answer the question based on the following context and conversation history:
    Conversation History: {history}
    Context: {context} 
    Question: {question} 
    """
    prompt = ChatPromptTemplate.from_template(template)
    history_runnable = RunnablePassthrough(input=st.session_state["history"])

    return (
        {"history": history_runnable, "context": retriever, "question": RunnablePassthrough()}
        | prompt
        | ChatOllama(model="llama2")  # Ollama model
        | StrOutputParser()
    )

# --- Streamlit App ---
st.title("Ollama Conversational Q&A")

if "history" not in st.session_state:
    st.session_state["history"] = []

st.sidebar.title("Data Options")
data_source_type = st.sidebar.selectbox("Choose your data source:", ["URLs", "Data Folder"])

if data_source_type == "URLs":
    data_source_input = st.sidebar.text_area("Enter URLs (one per line):")
elif data_source_type == "Data Folder":
    add_document = st.sidebar.file_uploader("Add Document to Data Folder")

    if add_document is not None:
        # Save uploaded file to data folder
        with open(os.path.join("data", add_document.name), "wb") as f:
            f.write(add_document.getvalue())
            st.sidebar.success(f"Added {add_document.name} to data folder.")

new_question = st.text_input("Ask your question:")

if st.button("Get Answer"):
    if data_source_type == "URLs":
        if data_source_input and new_question:
            data_sources = data_source_input.splitlines()
            retriever = load_and_process_data(data_source_type, data_sources)
            qa_pipeline = create_conversational_pipeline(retriever)
            answer = qa_pipeline.invoke(new_question)
           
            st.session_state["history"].append(f"User: {new_question} Ollama: {answer}")
            for chat_entry in st.session_state["history"]:
                st.write(chat_entry)
                
            # Fetch relevant URLs based on the model's answer
            relevant_urls = fetch_top_search_results(answer, num_results=5)
            if relevant_urls:
                st.subheader("Relevant URLs:")
                for url in relevant_urls:
                    st.write(url)
            else:
                st.info("No relevant URLs found.")
        else:
            st.warning("Please enter data and a question.")
    elif data_source_type == "Data Folder":
        if new_question:
            retriever = load_and_process_data(data_source_type, [])
            qa_pipeline = create_conversational_pipeline(retriever)
            answer = qa_pipeline.invoke(new_question)
           
            st.session_state["history"].append(f"User: {new_question} Ollama: {answer}")
            for chat_entry in st.session_state["history"]:
                st.write(chat_entry)
                
            # Fetch relevant URLs based on the model's answer
            relevant_urls = fetch_top_search_results(answer, num_results=5)
            if relevant_urls:
                st.subheader("Relevant URLs:")
                for url in relevant_urls:
                    st.write(url)
            else:
                st.info("No relevant URLs found.")
        else:
            st.warning("Please enter a question.")
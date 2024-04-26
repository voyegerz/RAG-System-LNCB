from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from IPython.display import Markdown as md
from dotenv import load_dotenv
import streamlit as st
import os

st.set_page_config(page_title="RAG System 📃", page_icon="🔎", layout="wide")


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


load_dotenv()
API_KEY = os.environ.get("API_KEY")

embedding_model = GoogleGenerativeAIEmbeddings(
    google_api_key=API_KEY, model="models/embedding-001"
)

loader = PyPDFLoader("https://arxiv.org/pdf/2404.07143.pdf")
pages = loader.load_and_split()
# st.write(pages)
text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)

chunks = text_splitter.split_documents(pages)
# st.write(chunks)
db = Chroma.from_documents(chunks, embedding_model, persist_directory="./db_files_rag")
db.persist()
db_connection = Chroma(
    persist_directory="./db_files_rag", embedding_function=embedding_model
)

# chromadb ---> retriever object
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

# cheking
# st.write(retriever.invoke("infinite context"))

# chat template
chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""You are a Helpful AI bot. 
    You take the question and context from user. your answer should be based on the specific context ."""
        ),
        HumanMessagePromptTemplate.from_template(
            """answer the following question based on the given context.
    Context : {context}
    Question : {question}
                                             
    Answer : """
        ),
    ]
)

# Initializing chat model
chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest", google_api_key=API_KEY
)

# output parser
output_parser = StrOutputParser()


# chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)


def set_style():
    st.markdown(
        """
    <style>
    .title {
        font-size: 36px !important;
        text-align: center !important;
        margin-bottom: 30px !important;
    }
    .subtitle {
        font-size: 24px !important;
        text-align: center !important;
        margin-bottom: 20px !important;
    }
    .text {
        font-size: 18px !important;
        margin-bottom: 10px !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


set_style()

st.title("⚙️ RAG System on “Leave No Context Behind” Paper 📃")
st.write(
    "A RAG based (Retrieval-Augmented Generation) system to answer questions based on the 'Leave No Context Behind [infinte-attention]' research paper 2024."
)

input_text = st.text_input("Enter your query :")

if st.button("Generate Answer"):
    with st.spinner("Generating answer..."):
        answer = rag_chain.invoke(input_text)
    st.success("Answer:")
    st.markdown(answer)

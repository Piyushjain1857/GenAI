import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
EMBEDDINGS = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def save_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=EMBEDDINGS)
    vector_store.save_local("faiss_index")


def user_input(user_question):

    new_db = FAISS.load_local(
        "faiss_index",
        EMBEDDINGS,
        allow_dangerous_deserialization=True,
    )

    docs = new_db.similarity_search(user_question)

    prompt = PromptTemplate(
        template="""
    Answer the question as detailed as possible from the provided context. If the answer is not present in the context, reply exactly: \"answer is not available in the context\".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """,
        input_variables=["context", "question"],
    )

    context = "\n\n".join(doc.page_content for doc in docs)

    formatted_prompt = prompt.format(
        context=context,
        question=user_question,
    )

    response = LLM.invoke(formatted_prompt)

    st.write("Reply:", response.content)

def main():
    st.set_page_config(page_title="Chat With Multiple PDF")
    st.header("Chat with Multiple PDF using Gemini🧑‍💻")

    user_question = st.text_input(
        "Ask a Question from the PDF Files"
    )

    if user_question:
        if not os.path.exists("faiss_index/index.faiss"):
            st.warning("Please upload PDF files and click 'Submit & Process' before asking questions.")
        else:
            user_input(user_question)

    with st.sidebar:
        st.title("Menu:")

        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    save_vector_store(text_chunks)
                    st.success("✅ FAISS index created successfully!")

                except Exception as e:
                    st.error("An error occurred while processing the PDF.")
                    st.exception(e)


if __name__ == "__main__":
    main()
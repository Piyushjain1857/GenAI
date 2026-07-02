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


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks


def save_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not present in the context, reply exactly: "answer is not available in the context".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    return model, prompt


def user_input(user_question):
    if not os.path.exists("faiss_index/index.faiss"):
        st.error("No PDF index found. Please upload your PDF files and click 'Submit & Process' first.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    docs = new_db.similarity_search(user_question)

    model, prompt = get_conversational_chain()

    context = "\n\n".join(doc.page_content for doc in docs)

    formatted_prompt = prompt.format(
        context=context,
        question=user_question,
    )

    response = model.invoke(formatted_prompt)

    st.write("Reply:", response.content)

def main():
    st.set_page_config("Chat With Multiple PDF")
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
                    if not pdf_docs:
                        st.warning("Please upload at least one PDF.")
                        st.stop()

                    st.write("📄 Reading PDF...")
                    raw_text = get_pdf_text(pdf_docs)

                    st.write("✂️ Splitting text...")
                    text_chunks = get_text_chunks(raw_text)
                    st.write(f"✅ Created {len(text_chunks)} chunks")

                    st.write("🧠 Creating embeddings and FAISS index...")
                    save_vector_store(text_chunks)

                    st.success("✅ FAISS index created successfully!")

                except Exception as e:
                    st.error("An error occurred while processing the PDF.")
                    st.exception(e)


if __name__ == "__main__":
    main()
#Importing all required Libraries
import streamlit as st
from PyPDF2 import PdfReader #To read pdfs
from langchain.text_splitter import RecursiveCharacterTextSplitter #to split
from langchain_google_genai import GoogleGenerativeAIEmbeddings #Googles Embedding Technique
import google.generativeai as genai #to interact with
from langchain.vectorstores import FAISS #For Vector Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

#Loading the google gemini pro api
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#Extracting Text form pdf
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf) #Reading the pdf
        for page in pdf_reader.pages:
            text+= page.extract_text() #extracting all the text from every page
    return text

#Converting the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000) #dividing text into chunks
    chunks = text_splitter.split_text(text)
    return chunks

#Convert chunks into vectors
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001") #model
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings) #Embedding the text chunks to create vector store
    vector_store.save_local("faiss_index") #saving the vectors to local directory


def get_conversational_chain():
    #prompt that is provided to Google Gemini Pro
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt) #to do internal text summarization

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings) #loading saved faiss_index
    docs = new_db.similarity_search(user_question)#perform similarity search on all the faiss vectors based on the qtn

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])



#Streamlit code
def main():
    st.set_page_config("Chat WIth PDFs !")
    st.header("Ask Questions And Chat With Your PDfs")

    user_question = st.text_input("Ask Questions Based On The Research Papers")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your Research Papers and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()

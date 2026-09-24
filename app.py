import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# App title and presentation
st.title("Let's get the Robot to Read!")
st.write("Versión de Python:", platform.python_version())

# Load and display image
try:
    image = Image.open('cartoon-image-of-a-robot-reading-a-book-vector.jpg')
    st.image(image, width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# Sidebar information
with st.sidebar:
    st.subheader("Your robot buddy will read whatever you need")

# Get API key from user
ke = st.text_input('Insert your OpenAI password, please', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Please insert your OpenAI password to continue")

# PDF uploader
pdf = st.file_uploader("Upload your PDF file here", type="pdf")

# Process the PDF if uploaded
if pdf is not None and ke:
    try:
        # Extract text from PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.info(f"Text extracted: {len(text)} characters")
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"Document divided into {len(chunks)} fragments")
        
        # Create embeddings and knowledge base
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # User question interface
        st.subheader("Escribe qué quieres saber sobre el documento")
        user_question = st.text_area(" ", placeholder="Ask it anything...")
        
        # Process question when submitted
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            # Use a current model instead of deprecated text-davinci-003
            # Options: "gpt-3.5-turbo-instruct" or "gpt-4o" depending on your API access
            llm = OpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18")
            
            # Load QA chain
            chain = load_qa_chain(llm, chain_type="stuff")
            
            # Run the chain
            response = chain.run(input_documents=docs, question=user_question)
            
            # Display the response
            st.markdown("### Respuesta:")
            st.markdown(response)
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
        # Add detailed error for debugging
        import traceback
        st.error(traceback.format_exc())
elif pdf is not None and not ke:
    st.warning("Please input your OpenAI password to continue")
else:
    st.info("Please upload a PDF file to continue")

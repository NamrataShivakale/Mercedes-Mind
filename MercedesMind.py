import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import Ollama  # Correct import
from langchain.prompts import PromptTemplate

# Load the FAISS vector store
def load_vector_store():
    # embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")
    # Allow dangerous deserialization (only do this if you trust the source)
    vector_store = FAISS.load_local("faiss_index_baai", embedding_model, allow_dangerous_deserialization=True)
    return vector_store

# Ask a question using Mistral 7B Instruct model via Ollama
def ask_question(query):
    vector_store = load_vector_store()

    # Search for the most relevant documents
    docs = vector_store.similarity_search(query, k=3)

    # Set up the Ollama model
    llm = Ollama(model="mistral:7b-instruct-v0.2-q4_0")

    # Create the QA chain
    chain = load_qa_chain(llm, chain_type="stuff")

    # Get the answer from the model
    answer = chain.run(input_documents=docs, question=query)

    return answer

# Streamlit UI
def create_ui():
    # Set background image
    st.markdown(
        """
        <style>
        .stApp {
            background-color:  #05214C;
            height: 100vh;  
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            .stTitle {
            color: white !important;  /* Use '!important' to ensure it overrides other styles */
        }
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.title("Car Manual Q&A System")

    # Initialize the session state for messages (conversation history)
    if 'message' not in st.session_state:
        st.session_state.message = []

    # Display the conversation history (messages)
    for message in st.session_state.message:
        st.chat_message(message['role']).markdown(message['content'])

    # Ask the user to input a question
    prompt = st.chat_input("Ask a Question about the Car Manual")

    # If a prompt is provided, display it as the user's message and store it
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.message.append({"role": "user", "content": prompt})

        # Get the answer from the model
        answer = ask_question(prompt)

        # Display the answer
        st.chat_message('assistant').markdown(answer)
        st.session_state.message.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    create_ui()

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ================== CONFIGURACIÓN ==================
CARPETA_DOCUMENTOS = "documentos"
CARPETA_DB = "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
GROQ_MODEL = "llama-3.3-70b-versatile"   # muy rápido y bueno en español

# ================== CARGAR O CREAR BASE VECTORIAL (caché) ==================
@st.cache_resource
def cargar_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    if os.path.exists(CARPETA_DB) and len(os.listdir(CARPETA_DB)) > 0:
        st.success("📂 Base de conocimiento cargada")
        return Chroma(persist_directory=CARPETA_DB, embedding_function=embeddings)
    
    st.info("🔨 Creando base de conocimiento por primera vez (puede tardar 1-2 minutos)...")
    documentos = []
    for archivo in os.listdir(CARPETA_DOCUMENTOS):
        ruta = os.path.join(CARPETA_DOCUMENTOS, archivo)
        if archivo.lower().endswith(".pdf"):
            loader = PyPDFLoader(ruta)
        elif archivo.lower().endswith(".docx"):
            loader = Docx2txtLoader(ruta)
        else:
            continue
        documentos.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documentos)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CARPETA_DB
    )
    st.success(f"✅ ¡Base creada con {len(chunks)} fragmentos!")
    return vectorstore

# ================== CADENA RAG ==================
def crear_chain(vectorstore, groq_api_key):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.3,
        api_key=groq_api_key
    )
    
    prompt = ChatPromptTemplate.from_template("""
Eres un asistente jurídico experto en derecho colombiano.
Responde de forma clara, precisa y profesional.
Usa SOLO la información del contexto.
Si no tienes suficiente información, di: "No tengo información suficiente en los documentos para responder esto."

Contexto:
{context}

Pregunta: {question}

Respuesta:
""")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ================== INTERFAZ STREAMLIT ==================
st.set_page_config(page_title="⚖️ Asistente Jurídico IA", layout="centered")
st.title("⚖️ Asistente Jurídico Local + Groq")
st.caption("Tu base de conocimiento privada • Respuestas ultrarrápidas")

# Clave API (se guarda en sesión)
if "groq_key" not in st.session_state:
    st.session_state.groq_key = ""

groq_key = st.text_input("🔑 Pega tu Groq API Key aquí:", 
                        value=st.session_state.groq_key, 
                        type="password")

if groq_key:
    st.session_state.groq_key = groq_key

vectorstore = cargar_vectorstore()

if groq_key:
    chain = crear_chain(vectorstore, groq_key)
    
    # Historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Mostrar mensajes anteriores
    for mensaje in st.session_state.messages:
        with st.chat_message(mensaje["role"]):
            st.markdown(mensaje["content"])
    
    # Input del usuario
    if pregunta := st.chat_input("Escribe tu consulta jurídica..."):
        st.session_state.messages.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)
        
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                respuesta = chain.invoke(pregunta)
                st.markdown(respuesta)
        
        st.session_state.messages.append({"role": "assistant", "content": respuesta})
else:
    st.warning("👉 Ingresa tu Groq API Key para empezar a chatear")

st.sidebar.info("✅ Base de conocimiento: tus documentos en la carpeta 'documentos'")
st.sidebar.caption("Modelo: Llama 3.3 70B vía Groq (ultrarrápido)")
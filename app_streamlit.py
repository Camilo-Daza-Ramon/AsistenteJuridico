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
GROQ_MODEL = "llama-3.3-70b-versatile"

GROQ_API_KEY = "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"

NOMBRE_ASISTENTE = "JusticIA"

# ================== PÁGINA (DEBE SER LA PRIMERA LLAMADA) ==================
st.set_page_config(
    page_title=f"{NOMBRE_ASISTENTE} · Asistente Jurídico IA",
    page_icon="⚖️",
    layout="centered"
)

# ================== ESTILOS Y ANIMACIONES ==================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;700;900&display=swap');

/* ── BASE ── */
.stApp { background-color: #020c1b !important; min-height: 100vh; }
.main .block-container { position: relative; z-index: 1; padding-top: 0 !important; max-width: 860px; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── FIX: AREA DE INPUT INTEGRADA AL FONDO ── */
[data-testid="stBottom"] {
    background: transparent !important;
    border-top: none !important;
    box-shadow: none !important;
}
[data-testid="stBottomBlockContainer"] {
    background: transparent !important;
    padding-bottom: 1rem !important;
}
/* Gradiente que funde el chat con el input */
[data-testid="stBottom"]::before {
    content: '';
    position: absolute;
    top: -70px; left: 0; right: 0;
    height: 70px;
    background: linear-gradient(to bottom, transparent 0%, #020c1b 100%);
    pointer-events: none;
    z-index: 0;
}

/* ── ORBES DE FONDO ── */
.bg-orbs { position: fixed; inset: 0; pointer-events: none; z-index: 0; overflow: hidden; }
.orb { position: absolute; border-radius: 50%; filter: blur(100px); animation: orbDrift ease-in-out infinite alternate; }
.orb1 { width: 600px; height: 600px; background: radial-gradient(circle, rgba(29,78,216,0.30) 0%, transparent 70%); top: -200px; left: -180px; animation-duration: 20s; }
.orb2 { width: 460px; height: 460px; background: radial-gradient(circle, rgba(14,165,233,0.20) 0%, transparent 70%); top: 30%; right: -130px; animation-duration: 24s; animation-delay: -9s; }
.orb3 { width: 520px; height: 520px; background: radial-gradient(circle, rgba(88,28,135,0.24) 0%, transparent 70%); bottom: -150px; left: 10%; animation-duration: 28s; animation-delay: -5s; }
.orb4 { width: 360px; height: 360px; background: radial-gradient(circle, rgba(5,150,105,0.13) 0%, transparent 70%); top: 50%; left: -80px; animation-duration: 32s; animation-delay: -18s; }
@keyframes orbDrift {
    0%   { transform: translate(0,0) scale(1); }
    33%  { transform: translate(45px,65px) scale(1.07); }
    66%  { transform: translate(-30px,35px) scale(0.95); }
    100% { transform: translate(25px,-40px) scale(1.04); }
}

/* ── SPLASH SCREEN ── */
#jia-splash {
    position: fixed; inset: 0;
    display: none; align-items: center; justify-content: center;
    background: radial-gradient(ellipse at 35% 40%, rgba(29,78,216,0.13) 0%, transparent 55%),
                radial-gradient(ellipse at 70% 65%, rgba(88,28,135,0.10) 0%, transparent 50%),
                #020c1b;
    z-index: 9999;
    transition: opacity 0.8s ease;
}
#jia-splash.visible  { display: flex; }
#jia-splash.fade-out { opacity: 0; pointer-events: none; }

#sp-logo-wrap {
    text-align: center;
    will-change: transform;
    transition: transform 1s cubic-bezier(0.4, 0, 0.2, 1);
    animation: spawnIn 0.9s cubic-bezier(0.34,1.4,0.64,1) both;
}
@keyframes spawnIn { from{opacity:0;transform:translateY(35px) scale(0.85);} to{opacity:1;transform:none;} }

#sp-logo-wrap.fly {
    transform: translateY(calc(-50vh + 80px)) scale(0.22);
}
/* Ocultar textos de soporte durante el vuelo */
#sp-logo-wrap.fly .sp-sub,
#sp-logo-wrap.fly .sp-bar-wrap,
#sp-logo-wrap.fly .sp-tagline,
#sp-logo-wrap.fly .sp-dots { opacity: 0; transition: opacity 0.25s ease; }

.sp-icon {
    font-size: 8rem;
    display: block;
    margin-bottom: 1.6rem;
    animation: iconPulse 2.2s ease-in-out infinite;
}
@keyframes iconPulse {
    0%,100% { filter: drop-shadow(0 0 18px rgba(59,130,246,0.55)) drop-shadow(0 0 40px rgba(59,130,246,0.2)); transform: scale(1) rotate(0deg); }
    50%     { filter: drop-shadow(0 0 55px rgba(96,165,250,1.0)) drop-shadow(0 0 90px rgba(96,165,250,0.45)); transform: scale(1.12) rotate(4deg); }
}
.sp-title {
    font-size: 6rem; font-weight: 900; letter-spacing: -3.5px; margin: 0 0 0.5rem; line-height: 1;
    background: linear-gradient(135deg,#f1f5f9 10%,#60a5fa 50%,#a78bfa 90%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    animation: titleReveal 0.85s cubic-bezier(0.34,1.4,0.64,1) 0.2s both;
}
@keyframes titleReveal { from{opacity:0;transform:scale(0.75) translateY(18px);} to{opacity:1;transform:none;} }
.sp-ia-grad {
    background: linear-gradient(135deg,#60a5fa,#a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.sp-sub {
    font-size: 0.88rem; color: #4b5e78; letter-spacing: 4.5px; text-transform: uppercase;
    margin-bottom: 2.8rem;
    animation: fadeUp 0.6s ease 0.6s both; transition: opacity 0.3s ease;
}
@keyframes fadeUp { from{opacity:0;transform:translateY(12px);} to{opacity:1;transform:none;} }
.sp-bar-wrap {
    width: 270px; height: 3px; background: rgba(255,255,255,0.06);
    border-radius: 99px; margin: 0 auto 2rem; overflow: hidden;
    animation: fadeUp 0.5s ease 0.8s both; transition: opacity 0.3s ease;
}
.sp-bar {
    height: 100%; border-radius: 99px; width: 0%;
    background: linear-gradient(90deg,#3b82f6,#8b5cf6,#06b6d4,#3b82f6); background-size: 300% 100%;
    animation: barFill 15s cubic-bezier(0.4,0,0.6,1) 0.8s forwards, shimmer 2.5s linear 0.8s infinite;
}
@keyframes barFill { 0%{width:0%;} 20%{width:28%;} 45%{width:55%;} 75%{width:78%;} 100%{width:92%;} }
@keyframes shimmer { 0%{background-position:100% 0;} 100%{background-position:-100% 0;} }
.sp-tagline {
    font-size: 0.72rem; color: #1c3252; letter-spacing: 2px; text-transform: uppercase;
    animation: fadeUp 0.6s ease 1s both; transition: opacity 0.3s ease;
}
.sp-dots { margin-top: 2.2rem; animation: fadeUp 0.5s ease 1.1s both; transition: opacity 0.3s ease; }
.sp-dot {
    display: inline-block; width: 7px; height: 7px;
    border-radius: 50%; margin: 0 5px;
    background: rgba(96,165,250,0.35);
    animation: dotBounce 1.3s ease-in-out infinite;
}
.sp-dot:nth-child(2) { animation-delay: 0.22s; }
.sp-dot:nth-child(3) { animation-delay: 0.44s; }
@keyframes dotBounce {
    0%,80%,100% { transform:scale(0.8); opacity:0.35; }
    40% { transform:scale(1.4); opacity:1; background:rgba(96,165,250,0.95); }
}

/* ── EMPUJAR CONTENIDO BAJO EL HEADER FIJO ── */
.main .block-container { padding-top: 110px !important; }

/* ── HEADER NAVBAR: FIJO, ANCHO TOTAL, UNA SOLA LINEA ── */
.app-header {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 50;
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0 2.5rem;
    height: 90px;
    background: rgba(2,12,27,0.92);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(59,130,246,0.12);
    box-shadow: 0 4px 32px rgba(0,0,0,0.4);
    pointer-events: none;
}
.app-icon {
    font-size: 2.6rem;
    line-height: 1;
    flex-shrink: 0;
    animation: iconHeaderPulse 3s ease-in-out infinite;
}
@keyframes iconHeaderPulse {
    0%,100% { filter: drop-shadow(0 0 8px rgba(96,165,250,0.5)); }
    50%     { filter: drop-shadow(0 0 22px rgba(96,165,250,1.0)) drop-shadow(0 0 44px rgba(167,139,250,0.5)); }
}
.app-title {
    font-family: 'Outfit', 'Inter', sans-serif;
    font-size: 1.8rem !important;
    font-weight: 900;
    letter-spacing: -2px;
    margin: 0;
    line-height: 1;
    white-space: nowrap;
    color: White;
    text-shadow: 0 2px 6px rgba(0,0,0,0.5);
}
.app-ia {
    color: #60a5fa;
    text-shadow:
        0 0 12px rgba(96,165,250,1.0),
        0 0 30px rgba(96,165,250,0.8),
        0 0 60px rgba(96,165,250,0.45);
    animation: iaNeon 3s ease-in-out infinite alternate;
}
@keyframes iaNeon {
    0%   { text-shadow: 0 0 10px rgba(96,165,250,0.9), 0 0 25px rgba(96,165,250,0.6); }
    100% { text-shadow: 0 0 16px rgba(96,165,250,1.0), 0 0 45px rgba(96,165,250,0.9), 0 0 80px rgba(167,139,250,0.5); }
}
.app-sep {
    color: rgba(96,165,250,0.35);
    font-size: 1.5rem;
    font-weight: 200;
    flex-shrink: 0;
    margin: 0 0.15rem;
}
.app-subtitle {
    font-family: 'Outfit', 'Inter', sans-serif;
    font-size: 0.88rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #f1f5f9;
    text-shadow: 0 0 18px rgba(255,255,255,0.25), 0 1px 4px rgba(0,0,0,0.5);
    white-space: nowrap;
    margin: 0;
}
.app-divider { display: none; }

/* ── MENSAJES DEL CHAT ── */
[data-testid="stChatMessage"] {
    animation: msgSlide 0.4s cubic-bezier(0.34,1.1,0.64,1) both !important;
    border-radius: 18px !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    backdrop-filter: blur(10px);
    margin-bottom: 0.7rem !important;
    background: rgba(10,20,40,0.6) !important;
}
@keyframes msgSlide { from{opacity:0;transform:translateY(16px) scale(0.97);} to{opacity:1;transform:none;} }

/* ── INPUT ── */
[data-testid="stChatInputContainer"] {
    background: rgba(10,18,35,0.88) !important;
    border: 1px solid rgba(59,130,246,0.22) !important;
    border-radius: 18px !important;
    backdrop-filter: blur(18px);
    box-shadow: 0 4px 28px rgba(0,0,0,0.35);
    transition: border-color 0.25s, box-shadow 0.25s;
}
[data-testid="stChatInputContainer"]:focus-within {
    border-color: rgba(96,165,250,0.55) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12), 0 4px 28px rgba(0,0,0,0.35) !important;
}

/* ── SPINNER ── */
.stSpinner > div { border-top-color: #60a5fa !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(96,165,250,0.4); }
</style>

<!-- ORBES DE FONDO ANIMADOS -->
<div class="bg-orbs">
  <div class="orb orb1"></div>
  <div class="orb orb2"></div>
  <div class="orb orb3"></div>
  <div class="orb orb4"></div>
</div>

<!-- SPLASH: logo grande en el centro, vuela al header cuando carga -->
<div id="jia-splash">
  <div id="sp-logo-wrap">
    <span class="sp-icon">&#x2696;&#xFE0F;</span>
    <p class="sp-title">Justic<span class="sp-ia-grad">IA</span></p>
    <p class="sp-sub">Asistente Jur&iacute;dico Inteligente</p>
    <div class="sp-bar-wrap"><div class="sp-bar"></div></div>
    <p class="sp-tagline">Powered by AI &middot; Derecho Colombiano</p>
    <div class="sp-dots">
      <span class="sp-dot"></span>
      <span class="sp-dot"></span>
      <span class="sp-dot"></span>
    </div>
  </div>
</div>

<script>
(function(){
    var splash = document.getElementById('jia-splash');
    var wrap   = document.getElementById('sp-logo-wrap');
    if (!splash || !wrap) return;

    /* Si ya cargó en esta sesión: solo mostrar el header sin splash */
    if (sessionStorage.getItem('jia_v2')) return;
    sessionStorage.setItem('jia_v2', '1');

    splash.style.display = 'flex';

    /* Polling: espera hasta que Streamlit haya renderizado el chat */
    var attempts = 0;
    var timer = setInterval(function(){
        attempts++;
        var ready = document.querySelector('[data-testid="stChatMessage"]')
                 || document.querySelector('[data-testid="stChatInputContainer"]');
        if (ready || attempts > 200) {
            clearInterval(timer);
            /* FASE 1: el logo vuela hacia arriba y se encoge */
            wrap.classList.add('fly');
            /* FASE 2: desvanece el splash completo */
            setTimeout(function(){
                splash.classList.add('fade-out');
                setTimeout(function(){ splash.style.display = 'none'; }, 850);
            }, 700);
        }
    }, 200);
})();
</script>
""", unsafe_allow_html=True)

# ── HEADER PRINCIPAL ──
st.markdown("""
<div class="app-header">
  <span class="app-icon">&#x2696;&#xFE0F;</span>
  <p class="app-title">Justic<span class="app-ia">IA</span></p>
  <span class="app-sep">|</span>
  <p class="app-subtitle">Asistente Jur&iacute;dico &middot; Derecho Colombiano</p>
  <div class="app-divider"></div>
</div>
""", unsafe_allow_html=True)

# ================== INICIALIZAR APP — vectorstore + chain (todo cacheado) ==================
@st.cache_resource(show_spinner="⚖️  Iniciando JusticIA…")
def inicializar_app():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(CARPETA_DB) and len(os.listdir(CARPETA_DB)) > 0:
        vectorstore = Chroma(persist_directory=CARPETA_DB, embedding_function=embeddings)
    else:
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

    # k=4 es suficiente y reduce tiempo de búsqueda
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.3, api_key=GROQ_API_KEY)

    prompt = ChatPromptTemplate.from_template("""
Eres JusticIA, un asistente jurídico experto en derecho colombiano.
Responde de forma clara, precisa, profesional y en español.
Usa SOLO la información del contexto.
Si no tienes suficiente información, responde: "No tengo información suficiente en los documentos para responder esta consulta."

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

# ================== CHAT ==================
chain = inicializar_app()

if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    bienvenida = """¡Hola! Soy **JusticIA** ⚖️

Tu asistente jurídico impulsado por Inteligencia Artificial, especializado en derecho colombiano.

Estoy listo para ayudarte con consultas sobre códigos, procedimientos, derechos, jurisprudencia y cualquier tema de tu base de conocimiento legal.

¿En qué puedo asistirte hoy?"""
    st.session_state.messages.append({"role": "assistant", "content": bienvenida})

for mensaje in st.session_state.messages:
    if mensaje["role"] == "assistant":
        with st.chat_message("assistant", avatar="⚖️"):
            st.markdown(mensaje["content"])
    else:
        with st.chat_message("user", avatar="👤"):
            st.markdown(mensaje["content"])

if pregunta := st.chat_input("Escribe tu consulta jurídica aquí..."):
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user", avatar="👤"):
        st.markdown(pregunta)

    with st.chat_message("assistant", avatar="⚖️"):
        with st.spinner("Analizando documentos..."):
            respuesta = chain.invoke(pregunta)
            st.markdown(respuesta)

    st.session_state.messages.append({"role": "assistant", "content": respuesta})
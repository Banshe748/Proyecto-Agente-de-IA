import os
import glob
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # solo tiene efecto en local, si existe un archivo .env

# ============================================================
# CONFIGURACIÓN
# ============================================================
NOMBRE_EMPRESA = "Santos Pegasus Soluciones"
CARPETA_DOCS = "docs"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"   # servido vía Hugging Face Inference Providers (no se carga localmente)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

RAG_TEMPLATE = """<s>[INST] Eres un asistente que responde preguntas usando SOLO el siguiente contexto. Si la respuesta no está en el contexto, dilo claramente en vez de inventarla. Responde en español, de forma breve y directa.

Contexto:
{context}

Pregunta: {input} [/INST]
"""

st.set_page_config(page_title=f"Asistente de {NOMBRE_EMPRESA}", page_icon="📄", layout="wide")


def obtener_hf_token():
    """Busca el token primero en los Secrets de Streamlit y luego en variables de entorno (uso local)."""
    try:
        if "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        pass
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY", "")


HF_TOKEN = obtener_hf_token()


# ============================================================
# RECURSOS CACHEADOS
# Streamlit vuelve a ejecutar todo el script en cada interacción del chat,
# así que sin @st.cache_resource el modelo y el índice se recargarían en
# cada pregunta. El guion bajo en "_embeddings" le dice a Streamlit que no
# intente hashear ese argumento.
# ============================================================

@st.cache_resource(show_spinner="Conectando con el modelo de lenguaje...")
def cargar_llm():
    if not HF_TOKEN:
        st.error(
            "Falta configurar HF_TOKEN. En Streamlit Cloud: Settings → Secrets. "
            "En local: archivo .env con HF_TOKEN=tu_token."
        )
        st.stop()
    return HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        provider="auto",  # deja que Hugging Face elija el proveedor disponible para el modelo
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=250,      # <-- CAMBIO 1: Bajamos de 300 a 250
        task="text-generation",  # <-- CAMBIO 2: Añadimos la tarea explícita
        do_sample=False,
        repetition_penalty=1.1,
    )


@st.cache_resource(show_spinner="Cargando modelo de embeddings...")
def cargar_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _dividir_en_chunks(documentos):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    return splitter.split_documents(documentos)


@st.cache_resource(show_spinner="Indexando la base de conocimiento...")
def construir_base_conocimiento(_embeddings):
    """Carga TODOS los PDF de la carpeta docs/ (no solo uno) y arma el índice FAISS."""
    rutas = sorted(glob.glob(os.path.join(CARPETA_DOCS, "*.pdf")))
    if not rutas:
        return None, []

    documentos = []
    for ruta in rutas:
        documentos.extend(PyPDFLoader(ruta).load())

    chunks = _dividir_en_chunks(documentos)
    vectorstore = FAISS.from_documents(chunks, _embeddings)
    nombres = [os.path.basename(r) for r in rutas]
    return vectorstore, nombres


def indexar_archivos_subidos(_embeddings, archivos):
    """Indexa PDFs subidos manualmente desde la barra lateral (solo para la sesión actual)."""
    documentos = []
    for archivo in archivos:
        ruta_temp = os.path.join("/tmp", f"{uuid.uuid4().hex}_{archivo.name}")
        with open(ruta_temp, "wb") as f:
            f.write(archivo.getbuffer())
        documentos.extend(PyPDFLoader(ruta_temp).load())

    chunks = _dividir_en_chunks(documentos)
    return FAISS.from_documents(chunks, _embeddings)


def _formatear_documentos(docs):
    texto_limpio = []
    for doc in docs:
        contenido = doc.page_content.replace("\x00", "").strip()
        if contenido:
            texto_limpio.append(contenido)
    
    texto_unido = "\n\n".join(texto_limpio)
    return texto_unido[:1500]

def construir_cadena_rag(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    prompt = PromptTemplate.from_template(RAG_TEMPLATE)
    
    cadena = (
        {"context": retriever | _formatear_documentos, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return cadena


# ============================================================
# INTERFAZ
# ============================================================

st.title(f"📄 Asistente de {NOMBRE_EMPRESA}")
st.caption("Pregunta en lenguaje natural sobre los documentos internos, sin necesidad de abrirlos.")

embeddings = cargar_embeddings()
llm = cargar_llm()
vectorstore_base, nombres_base = construir_base_conocimiento(embeddings)

with st.sidebar:
    st.header("Base de conocimiento")
    if nombres_base:
        st.success(f"{len(nombres_base)} documento(s) cargados")
        for nombre in nombres_base:
            st.caption(f"• {nombre}")
    else:
        st.warning(f"No se encontraron PDFs en `{CARPETA_DOCS}/`. Agrégalos al repositorio.")

    st.divider()
    st.subheader("Documentos adicionales")
    archivos_extra = st.file_uploader(
        "Sube PDFs solo para esta sesión",
        type="pdf",
        accept_multiple_files=True,
    )

    st.divider()
    if st.button("🗑️ Borrar historial de chat"):
        st.session_state.historial = []
        st.rerun()

if archivos_extra:
    with st.spinner("Indexando documentos adicionales..."):
        vectorstore_activo = indexar_archivos_subidos(embeddings, archivos_extra)
        if vectorstore_base is not None:
            # Fusionamos la base cacheada DENTRO de esta copia nueva y
            # desechable, nunca al revés. vectorstore_base la devuelve
            # @st.cache_resource y la comparten todas las sesiones; si
            # mutáramos ese objeto, los PDF que un usuario suba en su sesión
            # quedarían mezclados en la base de conocimiento de todos los demás.
            vectorstore_activo.merge_from(vectorstore_base)
else:
    vectorstore_activo = vectorstore_base

if vectorstore_activo is None:
    st.info(
        f"Agrega al menos un PDF a la carpeta `{CARPETA_DOCS}/` de tu repositorio, "
        "o súbelo desde la barra lateral, para comenzar."
    )
    st.stop()

if "historial" not in st.session_state:
    st.session_state.historial = []

for autor, mensaje in st.session_state.historial:
    with st.chat_message(autor):
        st.markdown(mensaje)

pregunta = st.chat_input("Escribe tu pregunta...")
if pregunta:
    st.session_state.historial.append(("user", pregunta))
    with st.chat_message("user"):
        st.markdown(pregunta)

with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            cadena = construir_cadena_rag(vectorstore_activo, llm)
            try:
                resultado = cadena.invoke(pregunta)
                # Validamos que el servidor haya devuelto texto real y no un objeto None
                if resultado:
                    respuesta = str(resultado).strip()
                else:
                    respuesta = "⚠️ El servidor gratuito de Hugging Face devolvió una respuesta vacía por saturación temporal o por un texto demasiado largo. Intenta reformular tu pregunta."
            except Exception as e:
                respuesta = f"Ocurrió un error al consultar el modelo: {e}"
            st.markdown(respuesta)

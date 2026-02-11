import streamlit as st
import requests
from dotenv import load_dotenv
import os

load_dotenv()

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG V8 (Reranker BGE)", page_icon="🧠", layout="wide")

# Rutas locales
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_IMAGENES_LOCAL = os.path.join(BASE_DIR, "data", "imagenes")

def reparar_ruta(ruta_db, filename):
    if os.path.exists(ruta_db): return ruta_db
    for root, dirs, files in os.walk(RUTA_IMAGENES_LOCAL):
        if filename in files: return os.path.join(root, filename)
    return None

st.markdown("""
<style>
    .stImage > img { max-height: 250px; object-fit: contain; }
    .debug-box { font-size: 11px; color: #555; background: #eee; padding: 2px 5px; border-radius: 3px; margin: 2px 0; }
    .debug-header { font-size: 13px; font-weight: bold; color: #333; margin-bottom: 5px; }
    .rerank-box { font-size: 11px; color: #064e3b; background: #d1fae5; padding: 4px; border: 1px solid #10b981; border-radius: 3px; margin: 2px 0; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("Configuración")
    if st.button("🗑️ Borrar Memoria"):
        st.session_state.messages = []
        st.rerun()

st.title("Tutor IA Avanzado (Reranking BGE-M3)")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hola. Ahora uso BGE-M3 para seleccionar los mejores textos."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Pregunta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    historial_envio = st.session_state.messages[:-1]
    if len(historial_envio) > 3: historial_envio = historial_envio[-3:]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Reranking inteligente en proceso..."):
            try:
                payload = {"pregunta": prompt, "history": historial_envio}
                response = requests.post(f"{API_URL}/ask", json=payload, timeout=300)
                
                if response.status_code == 200:
                    data = response.json()
                    respuesta = data["respuesta"]
                    placeholder.markdown(respuesta)
                    
                    debug_info = data.get("debug_info", {})
                    
                    if debug_info:
                        with st.expander("🧠 Cerebro del Sistema (Pipeline de Búsqueda)", expanded=False):
                            
                            st.caption("Fase 1: Retrieval (Casting) -> Fase 2: Reranking (Selección Final)")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("<div class='debug-header'>1. Vector (Top 10)</div>", unsafe_allow_html=True)
                                for item in debug_info.get('vector_qwen', []):
                                    st.markdown(f"<div class='debug-box'>{item}</div>", unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("<div class='debug-header'>2. BM25 (Top 10)</div>", unsafe_allow_html=True)
                                for item in debug_info.get('lexico_bm25', []):
                                    st.markdown(f"<div class='debug-box'>{item}</div>", unsafe_allow_html=True)
                                    
                            with col3:
                                st.markdown("<div class='debug-header'>🚀 3. BGE Reranker (Top 4)</div>", unsafe_allow_html=True)
                                st.caption("Ordenados por relevancia real")
                                for item in debug_info.get('reranker_bge', []):
                                    # Formato especial verde para los ganadores
                                    st.markdown(f"<div class='rerank-box'>{item}</div>", unsafe_allow_html=True)

                    # Fuentes
                    docs = data.get("fuentes_texto", [])
                    imgs = data.get("imagenes", [])
                    if docs or imgs:
                        with st.expander("📚 Referencias", expanded=True):
                            if docs:
                                st.markdown("###### 📄 Documentos")
                                for d in docs: st.caption(f"• {d}")
                            if imgs:
                                st.divider()
                                st.markdown("###### 🖼️ Imágenes")
                                cols = st.columns(len(imgs))
                                for idx, img_data in enumerate(imgs):
                                    ruta_final = reparar_ruta(img_data.get('path'), img_data.get('filename'))
                                    col = cols[idx] if idx < len(cols) else st.container()
                                    with col:
                                        if ruta_final:
                                            st.image(ruta_final, use_container_width=True)
                                        else:
                                            st.error(f"Perdida: {img_data.get('filename')}")

                    st.session_state.messages.append({"role": "assistant", "content": respuesta})
                else:
                    st.error(f"Error API: {response.text}")
            except Exception as e:
                st.error(f"Error: {e}")
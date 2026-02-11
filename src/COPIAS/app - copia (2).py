import streamlit as st
import requests
from dotenv import load_dotenv
import os

load_dotenv()

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG V7 (Debug Total)", page_icon="🧪", layout="wide")

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
    .debug-header { font-size: 14px; font-weight: bold; color: #333; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("Configuración")
    if st.button("🗑️ Borrar Memoria"):
        st.session_state.messages = []
        st.rerun()

st.title("Tutor IA Híbrido Multimodal")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hola. Pregunta y verás el debug de Texto e Imágenes."}]

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
        with st.spinner("Buscando Híbridamente..."):
            try:
                payload = {"pregunta": prompt, "history": historial_envio}
                response = requests.post(f"{API_URL}/ask", json=payload, timeout=300)
                
                if response.status_code == 200:
                    data = response.json()
                    respuesta = data["respuesta"]
                    placeholder.markdown(respuesta)
                    
                    # --- ZONA DE DEBUG ---
                    debug_info = data.get("debug_info", {})
                    
                    if debug_info:
                        with st.expander("🕵️ Debug: Comparativa de Motores (Texto vs Imagen)", expanded=False):
                            
                            # SECCIÓN 1: TEXTO
                            st.markdown("#### 📄 TEXTO (Documentos)")
                            col_t1, col_t2, col_t3 = st.columns(3)
                            with col_t1:
                                st.markdown("<div class='debug-header'>1. Vector (Qwen)</div>", unsafe_allow_html=True)
                                for item in debug_info.get('vector_qwen', []):
                                    st.markdown(f"<div class='debug-box'>{item}</div>", unsafe_allow_html=True)
                            with col_t2:
                                st.markdown("<div class='debug-header'>2. Léxico (BM25)</div>", unsafe_allow_html=True)
                                for item in debug_info.get('lexico_bm25', []):
                                    st.markdown(f"<div class='debug-box'>{item}</div>", unsafe_allow_html=True)
                            with col_t3:
                                st.markdown("<div class='debug-header'>3. Fusión Final</div>", unsafe_allow_html=True)
                                for item in debug_info.get('fusion_final', []):
                                    st.markdown(f"<div class='debug-box' style='background:#dcfce7; border:1px solid #86efac'>{item}</div>", unsafe_allow_html=True)

                            st.divider()

                            # SECCIÓN 2: IMÁGENES
                            st.markdown("#### 🖼️ IMÁGENES (Gráficos)")
                            col_i1, col_i2, col_i3 = st.columns(3)
                            with col_i1:
                                st.markdown("<div class='debug-header'>1. Vector (CLIP)</div>", unsafe_allow_html=True)
                                for item in debug_info.get('img_vector', []):
                                    st.markdown(f"<div class='debug-box'>{item}</div>", unsafe_allow_html=True)
                            with col_i2:
                                st.markdown("<div class='debug-header'>2. Léxico (BM25)</div>", unsafe_allow_html=True)
                                for item in debug_info.get('img_bm25', []):
                                    st.markdown(f"<div class='debug-box'>{item}</div>", unsafe_allow_html=True)
                            with col_i3:
                                st.markdown("<div class='debug-header'>3. Fusión Final</div>", unsafe_allow_html=True)
                                for item in debug_info.get('img_fusion', []):
                                    st.markdown(f"<div class='debug-box' style='background:#dcfce7; border:1px solid #86efac'>{item}</div>", unsafe_allow_html=True)

                    # --- ZONA DE VISUALIZACIÓN ---
                    docs = data.get("fuentes_texto", [])
                    imgs = data.get("imagenes", [])
                    if docs or imgs:
                        with st.expander("📚 Fuentes y Evidencias", expanded=True):
                            if docs:
                                st.markdown("###### 📄 Texto Usado")
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
                                            # Mostrar si vino más por CLIP o BM25 es difícil sin más datos, pero mostramos el score general
                                            st.caption(f"Score Fusión: {img_data.get('score')}")
                                        else:
                                            st.error(f"Perdida: {img_data.get('filename')}")

                    st.session_state.messages.append({"role": "assistant", "content": respuesta})
                else:
                    st.error(f"Error API: {response.text}")
            except Exception as e:
                st.error(f"Error: {e}")
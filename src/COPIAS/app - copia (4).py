import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Configuración dinámica
API_URL = f"http://{os.getenv('API_HOST', '127.0.0.1')}:{os.getenv('API_PORT', '8000')}"
RUTA_IMAGENES_LOCAL = os.getenv("DATA_PATH_IMAGENES", "./data/imagenes")

st.set_page_config(page_title=f"RAG ({os.getenv('LLM_PROVIDER', 'Auto')})", page_icon="🎓", layout="wide")

def reparar_ruta(ruta_db, filename):
    """Intenta encontrar la imagen en local si la ruta de la DB falla"""
    if os.path.exists(ruta_db): return ruta_db
    
    # Intento 1: Ruta desde .env
    ruta_env = os.path.join(RUTA_IMAGENES_LOCAL, filename)
    if os.path.exists(ruta_env): return ruta_env
    
    # Intento 2: Búsqueda recursiva
    for root, dirs, files in os.walk(RUTA_IMAGENES_LOCAL):
        if filename in files: return os.path.join(root, filename)
    return None

st.markdown("""
<style>
    .stImage > img { max-height: 250px; object-fit: contain; }
    .meta-tag { font-size: 11px; background: #e0f2fe; color: #0369a1; padding: 2px 6px; border-radius: 4px; font-weight: bold; border: 1px solid #bae6fd; }
    .debug-box { font-size: 10px; color: #444; background: #f4f4f5; padding: 3px; border: 1px solid #e4e4e7; margin-bottom: 2px; border-radius: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
    .winner-box { font-size: 11px; color: #065f46; background: #d1fae5; padding: 4px; border: 1px solid #10b981; margin-bottom: 3px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("Tutor IA")
    st.caption(f"Provider: **{os.getenv('LLM_PROVIDER', 'unknown').upper()}**")
    if st.button("🗑️ Borrar Memoria"):
        st.session_state.messages = []
        st.rerun()

st.title("Asistente Universitario Multimodal")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy tu tutor personal. Pregúntame sobre tus apuntes."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Escribe tu duda..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    historial_envio = st.session_state.messages[:-1]
    if len(historial_envio) > 3: historial_envio = historial_envio[-3:]

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        meta_data = {}
        
        try:
            # Conexión Streaming
            with requests.post(
                f"{API_URL}/ask", 
                json={"pregunta": prompt, "history": historial_envio}, 
                stream=True, timeout=120
            ) as response:
                
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            chunk = json.loads(line.decode('utf-8'))
                            
                            # --- FASE 1: METADATA (Llega al instante) ---
                            if chunk["type"] == "metadata":
                                meta_data = chunk
                                debug = meta_data.get("debug_info", {})
                                
                                # Renderizamos Trazabilidad
                                with st.expander("🕵️ Debug: Pipeline de Decisión", expanded=False):
                                    if "query_rewritten" in debug:
                                        st.info(f"✨ **Rewriting:** {debug['query_rewritten']}")
                                    
                                    c1, c2, c3 = st.columns([1, 1, 1.5])
                                    with c1:
                                        st.caption("Vector (Qwen)")
                                        for item in debug.get('step1_text_vec', [])[:5]: st.markdown(f"<div class='debug-box'>{item}</div>", unsafe_allow_html=True)
                                    with c2:
                                        st.caption("Léxico (BM25)")
                                        for item in debug.get('step1_text_bm25', [])[:5]: st.markdown(f"<div class='debug-box'>{item}</div>", unsafe_allow_html=True)
                                    with c3:
                                        st.caption("Reranker (Ganadores)")
                                        for item in debug.get('step2_text_final', []): st.markdown(f"<div class='winner-box'>{item}</div>", unsafe_allow_html=True)
                                    
                                    st.divider()
                                    st.caption("Imágenes Analizadas (Pipeline visual oculto por espacio)")

                                # Renderizamos Fuentes Finales
                                docs = meta_data.get("fuentes_texto", [])
                                imgs = meta_data.get("imagenes", [])
                                
                                if docs or imgs:
                                    with st.expander("📚 Fuentes Verificadas", expanded=True):
                                        if docs:
                                            st.markdown("**Documentos:**")
                                            for d in docs: st.caption(f"• {d}")
                                        if imgs:
                                            st.divider()
                                            cols = st.columns(len(imgs))
                                            for idx, img in enumerate(imgs):
                                                ruta = reparar_ruta(img.get('path'), img.get('filename'))
                                                with cols[idx] if idx < len(cols) else st.container():
                                                    if ruta:
                                                        st.image(ruta, use_container_width=True)
                                                        st.caption(f"Score: {img['score']}%")

                            # --- FASE 2: TEXTO (Efecto máquina de escribir) ---
                            elif chunk["type"] == "content":
                                delta = chunk.get("delta", "")
                                full_response += delta
                                message_placeholder.markdown(full_response + "▌")
                            
                            # --- FASE 3: ERROR ---
                            elif chunk["type"] == "error":
                                st.error(f"Error Backend: {chunk['message']}")

                    # Finalizar
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error(f"Error API: {response.status_code}")
        except Exception as e:
            st.error(f"Error conexión: {e}")
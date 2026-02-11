import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURACIÓN DE IMÁGENES ---
LOGO_ESTATICO = "https://png.pngtree.com/png-vector/20240516/ourlarge/pngtree-wet-dog-with-damp-hair-posing-against-a-dark-background-png-image_12480396.png" 
LOGO_ANIMADO = "https://media.tenor.com/ddggwHIbHIMAAAAM/mad-angry.gif" 
LOGO_SIDEBAR = "IA.png" 
# -------------------------------

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}"
RUTA_IMAGENES_LOCAL = os.getenv("DATA_PATH_IMAGENES", "./data/imagenes")

st.set_page_config(page_title=f"RAG ({os.getenv('LLM_PROVIDER', 'Auto')})", page_icon="🎓", layout="wide")

def reparar_ruta(ruta_db, filename):
    if not filename: return None
    if os.path.exists(ruta_db): return ruta_db
    ruta_env = os.path.join(RUTA_IMAGENES_LOCAL, filename)
    if os.path.exists(ruta_env): return ruta_env
    for root, dirs, files in os.walk(RUTA_IMAGENES_LOCAL):
        if filename in files: return os.path.join(root, filename)
    return None

st.markdown("""
<style>
    .stImage > img { max-height: 250px; object-fit: contain; }
    .meta-tag { font-size: 11px; background: #e0f2fe; color: #0369a1; padding: 2px 6px; border-radius: 4px; font-weight: bold; border: 1px solid #bae6fd; }
    .debug-box { font-size: 10px; color: #444; background: #f4f4f5; padding: 3px; border: 1px solid #e4e4e7; margin-bottom: 2px; border-radius: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
    .winner-box { font-size: 11px; color: #065f46; background: #d1fae5; padding: 4px; border: 1px solid #10b981; margin-bottom: 3px; border-radius: 4px; font-weight: bold; }
    
    /* CAMBIO 1: Tamaño dinámico (Responsivo) */
    .stChatMessage .stChatMessageAvatar { 
        width: 12vw;  /* Ocupa el 12% del ancho de la ventana */
        height: 12vw; 
        max-width: 150px; /* Tope máximo para pantallas gigantes */
        max-height: 150px;
        min-width: 60px;  /* Tope mínimo para móviles */
        min-height: 60px;
    }
    
    .stChatMessage .stImage {
        width: 100%;
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image(LOGO_SIDEBAR)
    st.title("Tutor IA")
    st.caption(f"Provider: **{os.getenv('LLM_PROVIDER', 'unknown').upper()}**")
    if st.button("🗑️ Borrar Memoria"):
        st.session_state.messages = []
        st.rerun()

st.title("Asistente Universitario Multimodal")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy TutorIA, tu profesor experto virtual. "}]

# 1. BUCLE HISTORIAL
for msg in st.session_state.messages:
    icono = LOGO_ESTATICO if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=icono):
        st.markdown(msg["content"])

if prompt := st.chat_input("Escribe tu duda..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    historial_envio = st.session_state.messages[:-1]
    if len(historial_envio) > 3: historial_envio = historial_envio[-3:]

    # 2. BLOQUE DE RESPUESTA ACTIVA
    with st.chat_message("assistant", avatar=LOGO_ANIMADO):
        message_placeholder = st.empty()
        full_response = ""
        meta_data = {}
        
        try:
            with requests.post(
                f"{API_URL}/ask", 
                json={"pregunta": prompt, "history": historial_envio}, 
                stream=True, timeout=120
            ) as response:
                
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            chunk = json.loads(line.decode('utf-8'))
                            
                            if chunk["type"] == "metadata":
                                meta_data = chunk
                                debug = meta_data.get("debug_info", {})
                                
                                with st.expander("🕵️ Debug: Pipeline de Decisión", expanded=False):
                                    if "query_rewritten" in debug:
                                        st.info(f"✨ **Rewriting:** {debug['query_rewritten']}")
                                    
                                    c1, c2, c3 = st.columns([1, 1, 1.5])
                                    with c1:
                                        st.caption("Vector")
                                        for item in debug.get('step1_text_vec', [])[:5]: st.markdown(f"<div class='debug-box'>{item}</div>", unsafe_allow_html=True)
                                    with c2:
                                        st.caption("Léxico")
                                        for item in debug.get('step1_text_bm25', [])[:5]: st.markdown(f"<div class='debug-box'>{item}</div>", unsafe_allow_html=True)
                                    with c3:
                                        st.caption("Reranker")
                                        for item in debug.get('step2_text_final', []): st.markdown(f"<div class='winner-box'>{item}</div>", unsafe_allow_html=True)

                                docs = meta_data.get("fuentes_texto", [])
                                raw_imgs = meta_data.get("imagenes", [])
                                
                                # CAMBIO 2: Filtrado de imágenes por Score > 40
                                imgs_filtradas = [img for img in raw_imgs if img.get('score', 0) > 40]
                                
                                if docs or imgs_filtradas:
                                    with st.expander("📚 Fuentes Verificadas", expanded=True):
                                        if docs:
                                            st.markdown("**Documentos:**")
                                            for d in docs: st.caption(f"• {d}")
                                            
                                        if imgs_filtradas:
                                            st.divider()
                                            
                                            num_cols = 2 
                                            cols = st.columns(num_cols)
                                            
                                            # Iteramos sobre la lista ya filtrada
                                            for idx, img in enumerate(imgs_filtradas):
                                                ruta = reparar_ruta(img.get('path'), img.get('filename'))
                                                score = img.get('score', 0)
                                                
                                                with cols[idx % num_cols]:
                                                    if ruta:
                                                        st.image(ruta, use_container_width=True)
                                                        color = "green" if score > 70 else "orange"
                                                        st.markdown(f"**:{color}[Certeza: {score}%]**")

                            elif chunk["type"] == "content":
                                delta = chunk.get("delta", "")
                                full_response += delta
                                message_placeholder.markdown(full_response + "▌")
                            
                            elif chunk["type"] == "error":
                                st.error(f"Error Backend: {chunk['message']}")

                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error(f"Error API: {response.status_code}")
        except Exception as e:
            st.error(f"Error conexión: {e}")
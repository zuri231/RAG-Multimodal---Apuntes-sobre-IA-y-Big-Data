import streamlit as st
import requests
from dotenv import load_dotenv
import os

load_dotenv()

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Universitario V4", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    .stImage > img { max-height: 250px; object-fit: contain; }
    .meta-tag { 
        font-size: 12px; 
        background-color: #f0f2f6; 
        padding: 4px 8px; 
        border-radius: 4px; 
        color: #31333F;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=90)
with col2:
    st.title("Tutor IA Multimodal")
    st.caption("Categorización Automática • Nivel de Certeza • Memoria Contextual")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Pregunta sobre tus asignaturas..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    historial_envio = st.session_state.messages[:-1]
    if len(historial_envio) > 3: historial_envio = historial_envio[-3:]

    with st.chat_message("assistant"):
        with st.spinner("Buscando en apuntes y diagramas..."):
            try:
                payload = {"pregunta": prompt, "history": historial_envio}
                response = requests.post(f"{API_URL}/ask", json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    st.markdown(data["respuesta"])

                    docs = data.get("fuentes_texto", [])
                    imgs = data.get("imagenes", []) # Ahora es una lista de diccionarios

                    if docs or imgs:
                        with st.expander("📚 Fuentes y Análisis Visual", expanded=True):
                            
                            # 1. DOCUMENTOS
                            if docs:
                                st.markdown("###### 📄 Referencias de Texto")
                                for d in docs:
                                    st.caption(f"• {d}")
                            
                            # 2. IMÁGENES (Con Metadatos y Score)
                            if imgs:
                                st.divider()
                                st.markdown("###### 🖼️ Evidencias Gráficas Encontradas")
                                
                                # Usamos columnas para mostrar imágenes en paralelo
                                cols = st.columns(len(imgs))
                                
                                for idx, img_data in enumerate(imgs):
                                    ruta = img_data['path']
                                    score = img_data['score']
                                    asignatura = img_data['asignatura'].upper()
                                    tema = img_data['tema']
                                    
                                    # Determinar color de la barra de certeza
                                    color_score = "green" if score > 75 else "orange" if score > 50 else "red"

                                    if os.path.exists(ruta):
                                        with cols[idx] if idx < len(cols) else st.container():
                                            # Imagen
                                            st.image(ruta, use_container_width=True)
                                            
                                            # Metadatos (Asignatura > Tema)
                                            st.markdown(f"<span class='meta-tag'>{asignatura}</span> • {tema}", unsafe_allow_html=True)
                                            
                                            # Barra de Certeza
                                            st.progress(score / 100)
                                            st.caption(f"🎯 Certeza: :{color_score}[{score}%]")
                                    else:
                                        st.error(f"Img no encontrada: {ruta}")

                    st.session_state.messages.append({"role": "assistant", "content": data["respuesta"]})
                else:
                    st.error(f"Error API: {response.text}")

            except Exception as e:
                st.error(f"Error conexión: {e}")
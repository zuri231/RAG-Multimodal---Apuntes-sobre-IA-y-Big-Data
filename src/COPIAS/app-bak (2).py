import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Cargar entorno
load_dotenv()

# URL de tu API FastAPI (Localhost por defecto)
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Universitario", page_icon="🎓", layout="centered")

# --- CABECERA ---
st.title("🎓 Asistente de Apuntes con IA")
st.markdown("Pregunta sobre tus PDFs y Diagramas de clase.")

# --- ESTADO DE LA API ---
try:
    health = requests.get(f"{API_URL}/health", timeout=2)
    if health.status_code == 200:
        st.success("✅ API Conectada y Base de Datos lista")
    else:
        st.error("⚠️ La API devuelve error.")
except:
    st.error("❌ No se detecta la API. Ejecuta 'uvicorn src.api:app --reload' en otra terminal.")

# --- CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input de usuario
if prompt := st.chat_input("¿Qué quieres saber hoy?"):
    # 1. Guardar y mostrar pregunta usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Llamar a la API
    with st.chat_message("assistant"):
        with st.spinner("Consultando apuntes y razonando..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask", 
                    json={"pregunta": prompt},
                    timeout=60 # Damos tiempo al LLM
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["respuesta"]
                    sources = data["fuentes"]
                    
                    st.markdown(answer)
                    
                    # Mostrar fuentes en un desplegable
                    if sources:
                        with st.expander("📚 Fuentes consultadas"):
                            for s in sources:
                                st.write(f"- {s}")
                    
                    # Guardar historial
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
                else:
                    st.error(f"Error de la API: {response.text}")
            
            except Exception as e:
                st.error(f"Error de conexión: {e}")
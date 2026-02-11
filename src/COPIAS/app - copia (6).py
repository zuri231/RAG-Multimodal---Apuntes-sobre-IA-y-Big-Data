import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURACIÓN DE ASSETS ---
AVATAR_CHICO = "./img/chico.png" 
AVATAR_CHICA = "./img/chica.png"
AVATAR_USER_CHICO = "./img/azul.png"    
AVATAR_USER_CHICA = "./img/morado.png"  
LOGO_APP = "./img/IA.png" 

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}"
RUTA_IMAGENES_LOCAL = os.getenv("DATA_PATH_IMAGENES", "./data/imagenes")

# Configuración de página
st.set_page_config(page_title="Tutor IA", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

def reparar_ruta(ruta_db, filename):
    if not filename: return None
    if os.path.exists(ruta_db): return ruta_db
    ruta_env = os.path.join(RUTA_IMAGENES_LOCAL, filename)
    if os.path.exists(ruta_env): return ruta_env
    for root, dirs, files in os.walk(RUTA_IMAGENES_LOCAL):
        if filename in files: return os.path.join(root, filename)
    return None

# --- MOTOR DE ESTILOS INTELIGENTE (LIGHT/DARK AUTO) ---
def get_theme_css(persona):
    if persona == "Chico":
        # === ARÍA: TECH THEME ===
        return """
        <style>
            /* --- FUENTES GLOBALES --- */
            .stApp { font-family: 'Courier New', monospace; }
            
            /* =========================================
               MODO OSCURO (DEFAULT - DEEP NAVY)
               ========================================= */
            .stApp {
                background-color: #0f172a; 
                color: #e2e8f0; 
            }
            [data-testid="stSidebar"] {
                background-color: #1e293b; 
                border-right: 1px solid #38bdf8; 
            }
            [data-testid="stSidebar"] * { color: #bae6fd !important; }
            
            h1, h2, h3 { 
                color: #38bdf8 !important; 
                text-transform: uppercase; 
                text-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
            }
            .stTextInput input {
                background-color: #1e293b !important;
                color: #7dd3fc !important; 
                border: 1px solid #38bdf8 !important;
            }
            .stTextInput label { color: #38bdf8 !important; }
            .stChatMessage {
                background-color: rgba(30, 41, 59, 0.6); 
                border-left: 4px solid #38bdf8;
                border-radius: 8px;
            }
            .debug-box { background: #020617; border: 1px dashed #38bdf8; color: #7dd3fc; }

            /* =========================================
               MODO CLARO (AUTO-DETECTADO - LAB TECH)
               ========================================= */
            @media (prefers-color-scheme: light) {
                .stApp {
                    background-color: #f8fafc !important; /* Gris muy claro técnico */
                    color: #0f172a !important; /* Azul marino casi negro */
                }
                [data-testid="stSidebar"] {
                    background-color: #e0f2fe !important; /* Azul cielo muy pálido */
                    border-right: 2px solid #0284c7 !important; 
                }
                [data-testid="stSidebar"] * { color: #0369a1 !important; }
                
                h1, h2, h3 { 
                    color: #0284c7 !important; /* Azul ingeniero fuerte */
                    text-shadow: none !important;
                }
                .stTextInput input {
                    background-color: #ffffff !important;
                    color: #0c4a6e !important; 
                    border: 2px solid #0284c7 !important;
                }
                .stTextInput label { color: #0284c7 !important; }
                
                .stChatMessage {
                    background-color: #ffffff !important; 
                    border-left: 5px solid #0284c7 !important;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
                    color: #334155 !important;
                }
                /* Ajuste de textos dentro del chat para modo claro */
                .stChatMessage p { color: #334155 !important; }
                
                .debug-box { background: #f0f9ff !important; border: 1px solid #0ea5e9 !important; color: #0369a1 !important; }
            }

            /* --- ESTILOS COMUNES (AVATARES Y BOTONES) --- */
            .stChatMessageAvatar {
                border: 2px solid #38bdf8;
                box-shadow: 0 0 8px rgba(56, 189, 248, 0.3);
                border-radius: 50% !important;
                object-fit: cover;
            }
            @media (prefers-color-scheme: light) {
                .stChatMessageAvatar { border-color: #0284c7 !important; box-shadow: none !important; }
            }

            div[role="radiogroup"] label { color: inherit !important; font-weight: bold; }
            .stButton button {
                border: 1px solid #38bdf8;
                color: #38bdf8;
                background-color: transparent;
                border-radius: 6px;
            }
            .stButton button:hover {
                border-color: #38bdf8;
                color: white;
                background-color: #38bdf8;
            }
            /* Botón modo claro */
            @media (prefers-color-scheme: light) {
                .stButton button {
                    border: 2px solid #0284c7 !important;
                    color: #0284c7 !important;
                }
                .stButton button:hover {
                    background-color: #0284c7 !important;
                    color: white !important;
                }
            }
        </style>
        """
    else:
        # === LEXÍA: ELEGANT THEME ===
        return """
        <style>
            /* --- FUENTES GLOBALES --- */
            .stApp { font-family: 'Verdana', sans-serif; }

            /* =========================================
               MODO OSCURO (DEFAULT - VIOLET SLATE)
               ========================================= */
            .stApp {
                background-color: #1e1b2e; 
                color: #e9d5ff; 
            }
            [data-testid="stSidebar"] {
                background-color: #262238;
                border-right: 1px solid #d8b4fe;
            }
            [data-testid="stSidebar"] * { color: #e9d5ff !important; }
            
            h1, h2, h3 { 
                color: #f0abfc !important; 
                font-family: 'Segoe UI', serif; 
                font-weight: 600;
            }
            .stTextInput input {
                background-color: #2d2a45 !important;
                color: #fce7f3 !important; 
                border: 1px solid #d8b4fe !important;
            }
            .stTextInput label { color: #f0abfc !important; }
            .stChatMessage {
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                border: 1px solid #581c87;
            }
            .debug-box { background: #2e1065; border: 1px solid #d8b4fe; color: #d8b4fe; }

            /* =========================================
               MODO CLARO (AUTO-DETECTADO - ACADEMIC PAPER)
               ========================================= */
            @media (prefers-color-scheme: light) {
                .stApp {
                    background-color: #fffbf0 !important; /* Crema muy suave */
                    color: #4a044e !important; /* Morado muy oscuro */
                }
                [data-testid="stSidebar"] {
                    background-color: #faf5ff !important; /* Lavanda blanquecino */
                    border-right: 2px solid #d8b4fe !important;
                }
                [data-testid="stSidebar"] * { color: #6b21a8 !important; }
                
                h1, h2, h3 { 
                    color: #7e22ce !important; /* Púrpura real */
                }
                .stTextInput input {
                    background-color: #ffffff !important;
                    color: #581c87 !important; 
                    border: 2px solid #d8b4fe !important;
                }
                .stTextInput label { color: #7e22ce !important; }
                
                .stChatMessage {
                    background-color: #ffffff !important;
                    border: 1px solid #e9d5ff !important;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
                    color: #2e1065 !important;
                }
                .stChatMessage p { color: #2e1065 !important; }
                
                .debug-box { background: #f3e8ff !important; border: 1px solid #a855f7 !important; color: #6b21a8 !important; }
            }

            /* --- ESTILOS COMUNES (AVATARES Y BOTONES) --- */
            .stChatMessageAvatar {
                border: 2px solid #f0abfc;
                box-shadow: 0 0 8px rgba(240, 171, 252, 0.3);
                border-radius: 50% !important;
                object-fit: cover;
            }
            @media (prefers-color-scheme: light) {
                .stChatMessageAvatar { border-color: #d8b4fe !important; box-shadow: none !important; }
            }

            div[role="radiogroup"] label { color: inherit !important; font-weight: bold; }
            .stButton button {
                background-color: #4c1d95;
                color: white;
                border-radius: 8px;
                border: none;
            }
            .stButton button:hover { background-color: #6b21a8; }
            
            @media (prefers-color-scheme: light) {
                .stButton button {
                    background-color: #f3e8ff !important;
                    color: #6b21a8 !important;
                    border: 1px solid #d8b4fe !important;
                }
                .stButton button:hover {
                    background-color: #d8b4fe !important;
                    color: white !important;
                }
            }
        </style>
        """

# --- SIDEBAR ---
with st.sidebar:
    st.image(LOGO_APP, width=80)
    st.title("Configuración")
    st.markdown("---")
    
    st.subheader("🎓 Elige a tu Profesor")
    
    tutor_mode = st.radio(
        "Selecciona:",
        ["Chico", "Chica"],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # LÓGICA DE AVATARES
    if tutor_mode == "Chico":
        nombre_ia = "ArIA"
        avatar_ia_actual = AVATAR_CHICO
        avatar_user_actual = AVATAR_USER_CHICO
        desc_ia = "Experto Técnico"
        st.info(f"👨‍💻 **{nombre_ia}** Activo.\n\n*Modo: Técnico / Estructurado.*")
    else:
        nombre_ia = "LexIA"
        avatar_ia_actual = AVATAR_CHICA
        avatar_user_actual = AVATAR_USER_CHICA
        desc_ia = "Experta Técnica"
        st.success(f"👩‍💻 **{nombre_ia}** Activa.\n\n*Modo: Didáctico / Detallista.*")

    st.markdown("---")
    if st.button("🔄 Reiniciar Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.caption(f"Engine: {os.getenv('LLM_PROVIDER', 'GROQ').upper()}")

# --- APLICAR CSS INTELIGENTE ---
st.markdown(get_theme_css(tutor_mode), unsafe_allow_html=True)

# --- HEADER ---
c1, c2 = st.columns([0.1, 0.9])
with c1:
    st.image(avatar_ia_actual, width=70)
with c2:
    if tutor_mode == "Chico":
        st.markdown(f"# >_ SYSTEM: {nombre_ia}")
    else:
        st.markdown(f"# ✨ Clase con {nombre_ia}")

# --- HISTORIAL ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    saludo = f"Sistema ArIA online. Esperando tu pregunta.👓" if tutor_mode == "Chico" else f"¡Hola! Soy {nombre_ia}. ¿En qué puedo ayudarte hoy?💻"
    st.session_state.messages.append({"role": "assistant", "content": saludo})

# Renderizar Mensajes
for msg in st.session_state.messages:
    icono = avatar_ia_actual if msg["role"] == "assistant" else avatar_user_actual
    with st.chat_message(msg["role"], avatar=icono):
        st.markdown(msg["content"])

# --- INPUT ---
placeholder = f"Ingresa comando para {nombre_ia}..." if tutor_mode == "Chico" else f"Pregúntale algo a {nombre_ia}..."

if prompt := st.chat_input(placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=avatar_user_actual):
        st.markdown(prompt)

    historial_envio = st.session_state.messages[:-4]

    with st.chat_message("assistant", avatar=avatar_ia_actual):
        message_placeholder = st.empty()
        full_response = ""
        
        loading_txt = "⚡ Procesando..." if tutor_mode == "Chico" else "✨ Consultando apuntes..."
        message_placeholder.markdown(f"_{loading_txt}_")
        
        try:
            with requests.post(
                f"{API_URL}/ask", 
                json={"pregunta": prompt, "history": historial_envio, "persona": tutor_mode.lower()}, 
                stream=True, timeout=120
            ) as response:
                
                if response.status_code == 200:
                    message_placeholder.empty()
                    for line in response.iter_lines():
                        if line:
                            chunk = json.loads(line.decode('utf-8'))
                            
                            if chunk["type"] == "metadata":
                                meta = chunk.get("debug_info", {})
                                imgs = chunk.get("imagenes", [])
                                docs = chunk.get("fuentes_texto", [])
                                
                                label_debug = "⚙️ KERNEL" if tutor_mode == "Chico" else "🧠 Razonamiento"
                                with st.expander(label_debug, expanded=False):
                                    c1, c2, c3 = st.columns(3)
                                    with c1:
                                        st.caption("Vector")
                                        for i in meta.get('step1_text_vec', [])[:2]: st.markdown(f"<div class='debug-box'>{i}</div>", unsafe_allow_html=True)
                                    with c2:
                                        st.caption("Keywords")
                                        for i in meta.get('step1_text_bm25', [])[:2]: st.markdown(f"<div class='debug-box'>{i}</div>", unsafe_allow_html=True)
                                    with c3:
                                        st.caption("Top")
                                        for i in meta.get('step2_text_final', [])[:2]: st.markdown(f"<div class='winner-box'>{i}</div>", unsafe_allow_html=True)

                                imgs_ok = [i for i in imgs if i.get('score', 0) > 40]
                                if imgs_ok or docs:
                                    st.markdown("---")
                                    if imgs_ok:
                                        cols = st.columns(3)
                                        for idx, img in enumerate(imgs_ok):
                                            path = reparar_ruta(img.get('path'), img.get('filename'))
                                            with cols[idx % 3]:
                                                if path: st.image(path, caption=f"{img['score']}%", use_container_width=True)
                                    
                                    if docs and tutor_mode == "Chica":
                                        with st.expander("Ver fuentes"):
                                            for d in docs: st.caption(f"📄 {d}")

                            elif chunk["type"] == "content":
                                full_response += chunk.get("delta", "")
                                message_placeholder.markdown(full_response + "▌")
                            
                            elif chunk["type"] == "error":
                                st.error(chunk['message'])

                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error(f"Error {response.status_code}")
        except Exception as e:
            st.error(f"Error de conexión: {e}")
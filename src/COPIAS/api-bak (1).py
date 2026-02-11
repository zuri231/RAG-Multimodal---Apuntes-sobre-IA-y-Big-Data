import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# Cargar entorno
load_dotenv()

app = FastAPI(title="API RAG Universitario", version="2.0")

# --- CONFIGURACIÓN ---
DB_PATH = os.getenv("DB_PATH")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-2.0-flash-lite-preview-02-05:free")

# MODELOS DE EMBEDDING (Deben coincidir con los usados al crear la DB)
MODELO_TEXTO_NAME = "Qwen/Qwen3-Embedding-0.6B"       # 1024 dimensiones
MODELO_EMBEDDING = "clip-ViT-B-32"                  # 512 dimensiones (El que da el error si no se usa)

# Cliente OpenRouter
client_llm = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Variables globales
chroma_client = None
model_texto = None
model_imagen = None
col_text = None
col_img = None

class QueryRequest(BaseModel):
    pregunta: str

@app.on_event("startup")
def startup_event():
    global chroma_client, model_texto, model_imagen, col_text, col_img
    print(f"🚀 Iniciando API... Conectando a {DB_PATH}")
    
    # 1. Cargar Modelo de TEXTO (Qwen)
    print(f"⏳ Cargando modelo TEXTO: {MODELO_TEXTO_NAME}...")
    model_texto = SentenceTransformer(MODELO_TEXTO_NAME, trust_remote_code=True)
    
    # 2. Cargar Modelo de IMAGEN (CLIP)
    print(f"⏳ Cargando modelo IMAGEN: {MODELO_EMBEDDING}...")
    model_imagen = SentenceTransformer(MODELO_EMBEDDING)
    
    # 3. Conectar a ChromaDB
    if not os.path.exists(DB_PATH):
        raise RuntimeError(f"❌ No encuentro la DB en: {DB_PATH}")
    
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    
    # Cargar colecciones
    try:
        col_text = chroma_client.get_collection("text_knowledge")
        print("✅ Colección de Textos conectada.")
    except:
        print("⚠️ No se encontró colección de textos.")

    try:
        col_img = chroma_client.get_collection("multimodal_knowledge")
        print("✅ Colección de Imágenes conectada.")
    except:
        print("⚠️ No se encontró colección de imágenes.")

@app.post("/ask")
def ask_question(request: QueryRequest):
    if not model_texto or not model_imagen:
        raise HTTPException(status_code=503, detail="Los modelos aún se están cargando")

    query = request.pregunta
    contexto_parts = []
    fuentes = []

    # A. BÚSQUEDA EN TEXTO (Usando Qwen - 1024 dims)
    if col_text:
        try:
            # Vectorizamos con Qwen
            vec_texto = model_texto.encode(query).tolist()
            res = col_text.query(query_embeddings=[vec_texto], n_results=3)
            
            for i, doc in enumerate(res['documents'][0]):
                meta = res['metadatas'][0][i]
                source = meta.get('source', 'Desconocido')
                asignatura = meta.get('asignatura', 'General')
                contexto_parts.append(f"[DOC: {source} ({asignatura})]: {doc}")
                fuentes.append(f"📄 {source}")
        except Exception as e:
            print(f"Error buscando en texto: {e}")

    # B. BÚSQUEDA EN IMÁGENES (Usando CLIP - 512 dims)
    if col_img:
        try:
            # Vectorizamos con CLIP (Aquí arreglamos el error de dimensión)
            vec_imagen = model_imagen.encode(query).tolist()
            res = col_img.query(query_embeddings=[vec_imagen], n_results=2)
            
            for i, doc in enumerate(res['documents'][0]):
                meta = res['metadatas'][0][i]
                source = meta.get('source', 'Desconocido')
                contexto_parts.append(f"[IMG: {source}]: {doc}")
                fuentes.append(f"🖼️ {source}")
        except Exception as e:
            print(f"Error buscando en imágenes: {e}")

    contexto_final = "\n\n".join(contexto_parts)

    # C. Generar Respuesta con LLM
    if not contexto_final:
        return {"respuesta": "No encontré información relevante en tus apuntes.", "fuentes": []}

    prompt_sistema = f"""
    Eres un profesor universitario experto. Responde usando SOLO el siguiente contexto.
    
    CONTEXTO RECUPERADO:
    {contexto_final}
    
    INSTRUCCIONES:
    1. Si usas información de una imagen, menciona "Como se ve en la imagen...".
    2. Sé didáctico y claro.
    """

    try:
        completion = client_llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": query},
            ]
        )
        respuesta_llm = completion.choices[0].message.content
        return {"respuesta": respuesta_llm, "fuentes": list(set(fuentes))}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "models": "Qwen + CLIP"}
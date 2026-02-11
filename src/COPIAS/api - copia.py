import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="API RAG Universitario", version="4.0")

# --- CONFIGURACIÓN ---
DB_PATH = os.getenv("DB_PATH")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-2.0-flash-lite-preview-02-05:free")

MODELO_TEXTO_NAME = "Qwen/Qwen3-Embedding-0.6B"
MODELO_IMAGEN_NAME = "clip-ViT-B-32"

client_llm = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

chroma_client = None
model_texto = None
model_imagen = None
col_text = None
col_img = None

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    pregunta: str
    history: List[Message] = []

@app.on_event("startup")
def startup_event():
    global chroma_client, model_texto, model_imagen, col_text, col_img
    print(f"🚀 Iniciando API v4.0 (Con Score y Metadatos)... BD: {DB_PATH}")
    
    model_texto = SentenceTransformer(MODELO_TEXTO_NAME, trust_remote_code=True)
    model_imagen = SentenceTransformer(MODELO_IMAGEN_NAME)
    
    if not os.path.exists(DB_PATH):
        raise RuntimeError(f"❌ No encuentro la DB en: {DB_PATH}")
    
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    try: col_text = chroma_client.get_collection("text_knowledge")
    except: pass
    try: col_img = chroma_client.get_collection("multimodal_knowledge")
    except: pass
    print("✅ Recursos cargados.")

@app.post("/ask")
def ask_question(request: QueryRequest):
    if not model_texto or not model_imagen:
        raise HTTPException(status_code=503, detail="Cargando modelos...")

    query = request.pregunta
    contexto_parts = []
    fuentes_texto = [] 
    
    # Nueva lista para objetos ricos (Ruta, Score, Asignatura, Tema)
    imagenes_info: List[Dict[str, Any]] = []

    # 1. BÚSQUEDA TEXTO
    if col_text:
        try:
            vec_texto = model_texto.encode(query).tolist()
            res = col_text.query(query_embeddings=[vec_texto], n_results=3)
            for i, doc in enumerate(res['documents'][0]):
                meta = res['metadatas'][0][i]
                source = meta.get('source', 'Doc')
                asignatura = meta.get('asignatura', 'General')
                tema = meta.get('tema', 'General')
                
                # Le pasamos la Asignatura/Tema al LLM para que sepa de qué habla
                contexto_parts.append(f"[TEXTO - {asignatura}/{tema}]: {doc}")
                fuentes_texto.append(f"{source} ({asignatura})")
        except Exception as e:
            print(f"Error texto: {e}")

    # 2. BÚSQUEDA IMÁGENES (Con Score y Metadatos)
    if col_img:
        try:
            vec_imagen = model_imagen.encode(query).tolist()
            # Pedimos 'distances' explícitamente, aunque query() las suele devolver por defecto
            res = col_img.query(
                query_embeddings=[vec_imagen], 
                n_results=3, 
                include=["documents", "metadatas", "distances"]
            )
            
            for i, doc in enumerate(res['documents'][0]):
                meta = res['metadatas'][0][i]
                path = meta.get('path', '')
                source = meta.get('source', 'Img')
                asignatura = meta.get('asignatura', 'General')
                tema = meta.get('tema', 'General')
                
                # CÁLCULO DE SCORE (Certeza)
                # Chroma devuelve Distancia Cosino (0 a 2).
                # Convertimos a Similitud (0 a 1). Fórmula aproximada: 1 - distancia
                distancia = res['distances'][0][i]
                score = max(0.0, 1.0 - distancia) # Evitamos negativos
                
                # Al LLM le damos el contexto completo
                contexto_parts.append(f"[IMAGEN - {asignatura}/{tema} - {source}]: {doc}")
                
                # Guardamos TODOS los datos para el Frontend
                if path:
                    imagenes_info.append({
                        "path": path,
                        "filename": source,
                        "asignatura": asignatura,
                        "tema": tema,
                        "score": round(score * 100, 1) # Convertimos a porcentaje (Ej: 85.5)
                    })

        except Exception as e:
            print(f"Error imagen: {e}")

    # 3. LLM
    contexto_str = "\n".join(contexto_parts)
    if not contexto_str: contexto_str = "No hay información."

    system_prompt = f"""
    Eres un profesor experto. Responde usando el CONTEXTO y MEMORIA.
    
    CONTEXTO (Incluye Texto e Imágenes):
    {contexto_str}
    
    INSTRUCCIONES:
    1. Menciona explícitamente si la información viene de una imagen ("Como vemos en el gráfico de [Asignatura]...").
    2. Usa el nombre de la asignatura para dar contexto.
    """

    messages_payload = [{"role": "system", "content": system_prompt}]
    for msg in request.history:
        messages_payload.append({"role": msg.role, "content": msg.content})
    messages_payload.append({"role": "user", "content": query})

    try:
        completion = client_llm.chat.completions.create(
            model=LLM_MODEL,
            messages=messages_payload
        )
        return {
            "respuesta": completion.choices[0].message.content,
            "fuentes_texto": list(set(fuentes_texto)),
            "imagenes": imagenes_info # Devolvemos la lista de objetos completa
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
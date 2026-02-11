import os
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("API_RAG")

load_dotenv()

app = FastAPI(title="API RAG Universitario", version="7.0-FullHybrid")

# Config
DB_PATH = os.getenv("DB_PATH")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-2.0-flash-lite-preview-02-05:free")
MODELO_TEXTO_NAME = "Qwen/Qwen3-Embedding-0.6B"
MODELO_IMAGEN_NAME = "clip-ViT-B-32"

client_llm = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# Globales
chroma_client = None
model_texto = None
model_imagen = None
col_text = None
col_img = None

# Índices BM25 (Ahora tenemos dos)
bm25_text_index = None
bm25_text_docs = []
bm25_text_metadatas = []

bm25_img_index = None
bm25_img_docs = []
bm25_img_metadatas = []

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    pregunta: str
    history: List[Message] = []

def reciprocal_rank_fusion(results_dict: Dict[str, Dict], k=60):
    fused_scores = {}
    for method, doc_list in results_dict.items():
        for rank, (doc_content, meta) in enumerate(doc_list):
            # Usamos el path o source como ID único para evitar duplicados
            doc_id = meta.get('path') or meta.get('source') or doc_content[:50]
            
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"score": 0, "doc": doc_content, "meta": meta}
            fused_scores[doc_id]["score"] += 1 / (rank + k)
    return sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)

@app.on_event("startup")
def startup_event():
    global chroma_client, model_texto, model_imagen, col_text, col_img
    global bm25_text_index, bm25_text_docs, bm25_text_metadatas
    global bm25_img_index, bm25_img_docs, bm25_img_metadatas
    
    logger.info("🚀 Iniciando API Híbrida TOTAL (Texto + Imágenes)...")

    model_texto = SentenceTransformer(MODELO_TEXTO_NAME, trust_remote_code=True)
    model_imagen = SentenceTransformer(MODELO_IMAGEN_NAME)
    
    if not os.path.exists(DB_PATH):
        raise RuntimeError(f"❌ DB no encontrada en: {DB_PATH}")
    
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    try: col_text = chroma_client.get_collection("text_knowledge")
    except: pass
    try: col_img = chroma_client.get_collection("multimodal_knowledge")
    except: pass

    # 1. Índice BM25 para TEXTO
    if col_text:
        logger.info("⚡ Creando índice BM25 (Texto)...")
        all_text = col_text.get()
        bm25_text_docs = all_text['documents']
        bm25_text_metadatas = all_text['metadatas']
        tokenized_text = [doc.lower().split(" ") for doc in bm25_text_docs]
        bm25_text_index = BM25Okapi(tokenized_text)

    # 2. Índice BM25 para IMÁGENES (Nuevo)
    if col_img:
        logger.info("⚡ Creando índice BM25 (Imágenes)...")
        all_img = col_img.get()
        bm25_img_docs = all_img['documents'] # Aquí están las descripciones generadas por la IA
        bm25_img_metadatas = all_img['metadatas']
        tokenized_img = [doc.lower().split(" ") for doc in bm25_img_docs]
        bm25_img_index = BM25Okapi(tokenized_img)

    logger.info("✅ Todo listo.")

@app.post("/ask")
def ask_question(request: QueryRequest):
    if not model_texto or not bm25_text_index:
        raise HTTPException(status_code=503, detail="Cargando...")
    
    query = request.pregunta
    
    # Debug Lists
    debug_text_vec = []
    debug_text_bm25 = []
    debug_text_final = []
    
    debug_img_vec = []
    debug_img_bm25 = []
    debug_img_final = []

    # ==========================================
    # 1. PROCESAMIENTO DE TEXTO (HÍBRIDO)
    # ==========================================
    candidates_text_vec = []
    candidates_text_bm25 = []

    # A. Vectorial Texto
    if col_text:
        try:
            vec = model_texto.encode(query).tolist()
            res = col_text.query(query_embeddings=[vec], n_results=5)
            for i, doc in enumerate(res['documents'][0]):
                meta = res['metadatas'][0][i]
                candidates_text_vec.append((doc, meta))
                debug_text_vec.append(f"{meta.get('source')} ({meta.get('asignatura')})")
        except: pass

    # B. BM25 Texto
    if bm25_text_index:
        try:
            tokenized_query = query.lower().split(" ")
            top_docs = bm25_text_index.get_top_n(tokenized_query, bm25_text_docs, n=5)
            for doc in top_docs:
                idx = bm25_text_docs.index(doc)
                meta = bm25_text_metadatas[idx]
                candidates_text_bm25.append((doc, meta))
                debug_text_bm25.append(f"{meta.get('source')} ({meta.get('asignatura')})")
        except: pass

    # C. Fusión Texto
    fusion_text = reciprocal_rank_fusion({"vec": candidates_text_vec, "bm25": candidates_text_bm25})
    final_text_docs = fusion_text[:4]

    # ==========================================
    # 2. PROCESAMIENTO DE IMÁGENES (HÍBRIDO - NUEVO)
    # ==========================================
    candidates_img_vec = []
    candidates_img_bm25 = []

    # A. Vectorial Imágenes (CLIP)
    if col_img:
        try:
            vec_img = model_imagen.encode(query).tolist()
            # Pedimos más candidatos (10) para tener margen de cruce
            res = col_img.query(query_embeddings=[vec_img], n_results=10, include=["documents", "metadatas", "distances"])
            for i, doc in enumerate(res['documents'][0]):
                meta = res['metadatas'][0][i]
                
                # Calculamos score vectorial para usarlo de referencia si queremos, 
                # pero RRF usa solo el ranking (posición)
                distancia = res['distances'][0][i]
                score_vec = max(0.0, 1.0 - distancia)
                
                # Guardamos score en metadata temporalmente para visualizarlo luego si es necesario
                meta['_score_vec'] = score_vec 
                
                candidates_img_vec.append((doc, meta))
                debug_img_vec.append(f"{meta.get('source')}")
        except: pass

    # B. BM25 Imágenes (Sobre las descripciones)
    if bm25_img_index:
        try:
            tokenized_query = query.lower().split(" ")
            # Buscamos en las descripciones de las fotos
            top_docs_img = bm25_img_index.get_top_n(tokenized_query, bm25_img_docs, n=10)
            for doc in top_docs_img:
                idx = bm25_img_docs.index(doc)
                meta = bm25_img_metadatas[idx]
                candidates_img_bm25.append((doc, meta))
                debug_img_bm25.append(f"{meta.get('source')}")
        except: pass

    # C. Fusión Imágenes
    fusion_img = reciprocal_rank_fusion({"vec": candidates_img_vec, "bm25": candidates_img_bm25})
    final_img_docs = fusion_img[:3] # Nos quedamos con las 3 mejores fotos fusionadas

    # ==========================================
    # 3. CONSTRUCCIÓN DE RESPUESTA
    # ==========================================
    contexto_parts = []
    fuentes_texto = []
    imagenes_info = []

    # Texto final
    for item in final_text_docs:
        meta = item['meta']
        debug_text_final.append(f"{meta.get('source')} ({meta.get('asignatura')})")
        contexto_parts.append(f"[TEXTO - {meta.get('asignatura')}/{meta.get('tema')}]: {item['doc']}")
        fuentes_texto.append(f"{meta.get('source')} ({meta.get('asignatura')})")

    # Imágenes finales
    for item in final_img_docs:
        meta = item['meta']
        doc = item['doc']
        debug_img_final.append(f"{meta.get('source')}")
        
        # Recuperamos el score. Si vino de BM25 puro, no tiene score vectorial, ponemos uno alto figurado o 0.
        # Para RRF el score absoluto no importa tanto, pero para la UI sí.
        # Si existe _score_vec (vino de CLIP), lo usamos. Si no, calculamos uno falso basado en BM25 (posición).
        raw_score = meta.get('_score_vec', 0.9) # 0.9 por defecto si lo encontró BM25 por palabra exacta
        
        contexto_parts.append(f"[IMAGEN - {meta.get('source')}]: {doc}")
        if meta.get('path'):
            imagenes_info.append({
                "path": meta.get('path'), 
                "filename": meta.get('source'),
                "asignatura": meta.get('asignatura', 'Gral'), 
                "tema": meta.get('tema', '-'),
                "score": round(raw_score * 100, 1)
            })

    # LLM
    contexto_str = "\n".join(contexto_parts) or "Sin información."
    
    system_prompt = f"""
    Eres un profesor experto. Responde usando el CONTEXTO y MEMORIA.
    CONTEXTO (Híbrido Multimodal):
    {contexto_str}
    """

    messages = [{"role": "system", "content": system_prompt}]
    for msg in request.history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": query})

    try:
        completion = client_llm.chat.completions.create(model=LLM_MODEL, messages=messages)
        return {
            "respuesta": completion.choices[0].message.content,
            "fuentes_texto": list(set(fuentes_texto)),
            "imagenes": imagenes_info,
            "debug_info": {
                "vector_qwen": debug_text_vec,
                "lexico_bm25": debug_text_bm25,
                "fusion_final": debug_text_final,
                # Debug de Imágenes también
                "img_vector": debug_img_vec,
                "img_bm25": debug_img_bm25,
                "img_fusion": debug_img_final
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
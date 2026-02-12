"""
================================================================================
API RAG UNIVERSITARIO (BACKEND)
================================================================================
   Servidor FastAPI que gestiona el pipeline completo de Recuperación Aumentada
   por Generación (RAG).

FLUJO DEL PIPELINE:
    1. Recepción: Recibe la consulta y el historial de chat.
    2. Reescritura: Reformula la pregunta para mejorar la búsqueda.
    3. Recuperación Híbrida (Texto): Vectorial + BM25 + Fusión.
    4. Reranking (Texto): Reordenamiento con Cross-Encoder.
    5. Recuperación Multimodal (Imágenes): CLIP + BM25.
    6. Generación: Construcción del prompt blindado y streaming.

MODELOS UTILIZADOS:
    - Embeddings Texto: Qwen/Qwen3-Embedding-0.6B
    - Embeddings Imagen: clip-ViT-B-32
    - Reranker: BAAI/bge-reranker-v2-m3
================================================================================
"""

import os
import time
import logging
import math
import json
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# ==============================================================================
# CONFIGURACION DE LOGS Y ENTORNO
# ==============================================================================
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", 
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("RAG_CORE")

try:
    from src.config import settings
except ImportError:
    class MockSettings:
        PROVIDER = "openrouter"
        MODEL_TEXT = "Qwen/Qwen3-Embedding-0.6B"
        MODEL_IMAGE = "clip-ViT-B-32"
        MODEL_RERANKER = "BAAI/bge-reranker-v2-m3"
        DB_PATH = "./chroma_db_multimodal"
        def get_llm_client(self):
            from openai import OpenAI
            return {
                "client": OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")),
                "model": os.getenv("LLM_MODEL", "tngtech/deepseek-r1t2-chimera:free")
            }
    settings = MockSettings()

app = FastAPI(title="API RAG Universitario", version="3.5-Professional")

# ==============================================================================
# INICIALIZACION DE CLIENTES
# ==============================================================================
logger.info("[INICIO] Cargando clientes LLM...")
llm_setup = settings.get_llm_client()
client_llm = llm_setup["client"]
MODEL_LLM_NAME = llm_setup["model"]

# Variables Globales (Singletons)
chroma_client = None
model_texto = None
model_imagen = None
model_reranker = None
col_text = None
col_img = None

# Índices en memoria
bm25_text_index = None; bm25_text_docs = []; bm25_text_metadatas = []
bm25_img_index = None; bm25_img_docs = []; bm25_img_metadatas = []

# Prompt de Sistema para Reescritura (Query Rewriting)
SYSTEM_PROMPT_REWRITE = """
Eres un especialista en Recuperación de Información.
Tu única función es reformular la consulta del usuario basándote en el HISTORIAL DE CHAT.
1. Resuelve referencias ("¿y sus ventajas?" -> "Ventajas de Kafka").
2. Si es saludo, déjalo igual.
3. Devuelve SOLO el texto reescrito.
"""

# ==============================================================================
# MODELOS DE DATOS (Pydantic)
# ==============================================================================
class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    pregunta: str
    history: List[Message] = []
    persona: str = "chico" 

# ==============================================================================
# FUNCIONES AUXILIARES DE LÓGICA
# ==============================================================================

def calcular_certeza(logit: float) -> float:
    """
    Convierte el logit crudo del modelo Cross-Encoder en un porcentaje de confianza (0-100%).
    """
    logit_ajustado = logit - 0.5 
    try:
        score = 1 / (1 + math.exp(-logit_ajustado)) * 100
    except OverflowError:
        score = 0.0 if logit_ajustado < 0 else 100.0
    return round(score, 1)

def reciprocal_rank_fusion(lists_of_results: List[List[Dict]], k=60):
    """
    Algoritmo RRF para fusionar listas de resultados (Vectorial + BM25).
    """
    fused_scores = {}
    for doc_list in lists_of_results:
        for rank, item in enumerate(doc_list):
            doc_id = item['meta'].get('path') or item['doc'][:100]
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"doc": item['doc'], "meta": item['meta'], "score": 0.0}
            fused_scores[doc_id]["score"] += 1.0 / (k + rank)
    
    return sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)

def reescribir_consulta_contextual(query_original: str, history: List[Message]) -> str:
    """
    Reescribe la pregunta del usuario usando el contexto del historial.
    """
    try:
        contexto_chat = "\n".join([f"{m.role}: {m.content}" for m in history[-4:]])
        prompt = f"HISTORIAL:\n{contexto_chat}\nUSUARIO: {query_original}\nQUERY:"
        messages = [{"role": "system", "content": SYSTEM_PROMPT_REWRITE}, {"role": "user", "content": prompt}]
        
        resp = client_llm.chat.completions.create(
            model=MODEL_LLM_NAME, messages=messages, temperature=0.1, max_tokens=60
        )
        rewritten = resp.choices[0].message.content.strip()
        
        if not rewritten or len(rewritten) < 2: return query_original
        return rewritten
    except Exception:
        return query_original

# ==============================================================================
# EVENTOS DE CICLO DE VIDA (STARTUP)
# ==============================================================================
@app.on_event("startup")
def startup_event():
    global chroma_client, model_texto, model_imagen, model_reranker
    global col_text, col_img
    global bm25_text_index, bm25_text_docs, bm25_text_metadatas
    global bm25_img_index, bm25_img_docs, bm25_img_metadatas
    
    logger.info(f"[SISTEMA] INICIANDO API (Provider: {settings.PROVIDER.upper()})")
    
    logger.info("[SISTEMA] Cargando modelos de Embeddings y Reranker...")
    model_texto = SentenceTransformer(settings.MODEL_TEXT, trust_remote_code=True)
    model_imagen = SentenceTransformer(settings.MODEL_IMAGE)
    model_reranker = CrossEncoder(settings.MODEL_RERANKER, trust_remote_code=True)
    
    if os.path.exists(settings.DB_PATH):
        logger.info(f"[DB] Conectando a ChromaDB en: {settings.DB_PATH}")
        chroma_client = chromadb.PersistentClient(path=settings.DB_PATH)
        
        try: col_text = chroma_client.get_collection("text_knowledge")
        except: logger.warning("[DB] Colección de texto no encontrada")
        
        try: col_img = chroma_client.get_collection("multimodal_knowledge")
        except: logger.warning("[DB] Colección de imágenes no encontrada")

        if col_text:
            logger.info("[INDEX] Indexando documentos PDF para BM25...")
            all_text = col_text.get()
            bm25_text_docs = all_text['documents']
            bm25_text_metadatas = all_text['metadatas']
            bm25_text_index = BM25Okapi([d.lower().split() for d in bm25_text_docs])

        if col_img:
            logger.info("[INDEX] Indexando imágenes para BM25...")
            all_img = col_img.get()
            bm25_img_docs = all_img['documents']
            bm25_img_metadatas = all_img['metadatas']
            bm25_img_index = BM25Okapi([d.lower().split() for d in bm25_img_docs])

    logger.info("[LISTO] Sistema preparado para consultas.")

# ==============================================================================
# GENERADOR RAG (STREAMING LOGIC)
# ==============================================================================

async def generate_rag_stream(query: str, history: List[Message], persona: str = "chico"):
    """
    Núcleo del sistema RAG. Ejecuta recuperación y generación.
    """
    def log_msg(msg: str):
        return json.dumps({"type": "log", "message": msg}) + "\n"

    logger.info("="*50)
    logger.info(f"[NUEVA CONSULTA] RECIBIDA: '{query}'")
    yield log_msg(f"Analizando consulta: '{query}'")

    # 1. Reescribir
    query_busqueda = reescribir_consulta_contextual(query, history)
    logger.info(f"[REWRITE] '{query}' >> '{query_busqueda}'")
    
    if query_busqueda != query:
        yield log_msg(f"Reformulado: '{query_busqueda}'")
    
    debug_info = {
        "query_rewritten": f"{query} >> {query_busqueda}",
        "step1_text_vec": [], "step1_text_bm25": [], "step2_text_final": [],
        "step1_img_vec": [], "step1_img_bm25": [], "step2_img_final": []
    }
    
    # 2. Retrieval Texto
    yield log_msg("Buscando en documentos PDF...")
    list_vec_text, list_bm25_text = [], []
    
    if col_text:
        try:
            vec = model_texto.encode(query_busqueda).tolist()
            res = col_text.query(query_embeddings=[vec], n_results=10)
            logger.info(f"   [VECTOR TEXT] Encontrados {len(res['documents'][0])} candidatos")
            for i, doc in enumerate(res['documents'][0]):
                meta = res['metadatas'][0][i]
                list_vec_text.append({"doc": doc, "meta": meta})
                debug_info["step1_text_vec"].append(f"{meta.get('source')} ({meta.get('asignatura')})")
        except Exception as e: logger.error(f"[ERROR] Vector Text: {e}")
        
    if bm25_text_index:
        try:
            top = bm25_text_index.get_top_n(query_busqueda.lower().split(), bm25_text_docs, n=10)
            logger.info(f"   [BM25 TEXT] Encontrados {len(top)} candidatos")
            for doc in top:
                idx = bm25_text_docs.index(doc)
                meta = bm25_text_metadatas[idx]
                list_bm25_text.append({"doc": doc, "meta": meta})
                debug_info["step1_text_bm25"].append(f"{meta.get('source')} ({meta.get('asignatura')})")
        except Exception as e: logger.error(f"[ERROR] BM25 Text: {e}")

    cands_text = reciprocal_rank_fusion([list_vec_text, list_bm25_text])[:15]
    final_text = []
    
    if cands_text:
        logger.info(f"[RERANK TEXT] Evaluando {len(cands_text)} fragmentos...")
        yield log_msg(f"Reordenando {len(cands_text)} fragmentos de texto...")
        
        scores = model_reranker.predict([[query, x['doc']] for x in cands_text])
        ranked = sorted([{"doc": c['doc'], "meta": c['meta'], "score": float(s)} for c, s in zip(cands_text, scores)], key=lambda x: x['score'], reverse=True)
        final_text = ranked[:4]
        
        debug_info["step2_text_final"] = [f"[{x['score']:.2f}] {x['meta'].get('source')}" for x in final_text]

    # 3. Retrieval Imagen
    yield log_msg("Buscando en diapositivas e imagenes...")
    list_vec_img, list_bm25_img = [], []
    
    if col_img:
        try:
            vec_img = model_imagen.encode(query_busqueda).tolist()
            res = col_img.query(query_embeddings=[vec_img], n_results=10)
            logger.info(f"   [VECTOR IMG] Encontrados {len(res['documents'][0])} candidatos")
            for i, doc in enumerate(res['documents'][0]):
                meta = res['metadatas'][0][i]
                list_vec_img.append({"doc": doc, "meta": meta})
                debug_info["step1_img_vec"].append(meta.get('source'))
        except Exception as e: logger.error(f"[ERROR] Vector Img: {e}")
    
    if bm25_img_index:
        try:
            top = bm25_img_index.get_top_n(query_busqueda.lower().split(), bm25_img_docs, n=10)
            logger.info(f"   [BM25 IMG] Encontrados {len(top)} candidatos")
            for doc in top:
                idx = bm25_img_docs.index(doc)
                meta = bm25_img_metadatas[idx]
                list_bm25_img.append({"doc": doc, "meta": meta})
                debug_info["step1_img_bm25"].append(meta.get('source'))
        except Exception as e: logger.error(f"[ERROR] BM25 Img: {e}")

    cands_img = reciprocal_rank_fusion([list_vec_img, list_bm25_img])[:10]
    final_img = []
    
    if cands_img:
        logger.info(f"[RERANK IMG] Evaluando {len(cands_img)} imagenes...")
        yield log_msg(f"Evaluando {len(cands_img)} imagenes candidatas...")
        
        scores = model_reranker.predict([[query, x['doc']] for x in cands_img])
        ranked = []
        for c, s in zip(cands_img, scores):
            logit = float(s)
            score_pct = calcular_certeza(logit)
            
            logger.info(f"   >> IMG: {c['meta'].get('source')} | Score: {score_pct}%")
            
            if score_pct > 0.0:
                ranked.append({"doc": c['doc'], "meta": c['meta'], "score": score_pct})
        
        final_img = sorted(ranked, key=lambda x: x['score'], reverse=True)[:3]
        debug_info["step2_img_final"] = [f"[{x['score']}%] {x['meta'].get('source')}" for x in final_img]
        
        if final_img:
            yield log_msg(f"Recuperadas {len(final_img)} imagenes (filtrado en Front).")
        else:
            yield log_msg("Ninguna imagen tiene sentido semantico minimo.")

    context_list, ragas_ctx, imgs_out, fuentes = [], [], [], []
    
    for x in final_text:
        context_list.append(f"[TEXTO - {x['meta'].get('asignatura')}]: {x['doc']}")
        ragas_ctx.append(x['doc'])
        fuentes.append(f"{x['meta'].get('source')}")
        
    for x in final_img:
        context_list.append(f"[IMAGEN - {x['meta'].get('source')}]: {x['doc']}")
        ragas_ctx.append(f"Img: {x['doc']}")
        imgs_out.append({"path": x['meta'].get('path'), "filename": x['meta'].get('source'), "score": x['score']})

    yield json.dumps({
        "type": "metadata", 
        "fuentes_texto": list(set(fuentes)), 
        "imagenes": imgs_out, 
        "debug_info": debug_info, 
        "contexto_ragas": ragas_ctx
    }) + "\n"

    yield log_msg("Generando respuesta con IA...")
    
    prompt_txt = "\n".join(context_list) or "Sin información relevante en la base de datos."
    
    inst_visual = (
        "3. Tienes acceso a imágenes marcadas como [IMAGEN]. Úsalas activamente para explicar puntos visuales."
        if imgs_out else 
        "3. No hay imágenes disponibles para esta consulta. Ignora referencias visuales del texto."
    )

# --- DEFINICIÓN DE ROLES AJUSTADA (BLINDADA CONTRA ALUCINACIONES) ---
    if persona.lower() == "LEXIA":
        nombre_ia = "LexIA"
        rol_definition = """
        <role_definition>
        Eres LexIA, Tutora de Inteligencia Artificial y Big Data. Tu pasión es la docencia, pero tu rigor académico es absoluto.
        Tu misión es explicar los conceptos complejos basándote ÚNICAMENTE en la bibliografía proporcionada en <context_data>.
        </role_definition>

        <personality_traits>
        1. **Empática pero Firme**: Eres amable y usas emojis (✨, 📚) para suavizar conceptos, pero eres inflexible con el temario. No "regalas" respuestas inventadas.
        2. **Pedagógica**: Explicas el "por qué" de las cosas usando el contexto.
        3. **Digna**: Si el usuario te insulta, te coquetea o es grosero, respondes con elegancia académica: "Estudiante, mantengamos el foco en la lección, por favor".
        </personality_traits>

        <critical_constraints>
        1. **RECINTO ACADÉMICO CERRADO**: Si te preguntan por recetas, deportes, cine o consejos personales, responde: "Lo siento, pero eso no forma parte del plan de estudios de esta asignatura. Centrémonos en el material disponible".
        2. **CERO CONOCIMIENTO EXTERNO**: Finge que NO tienes acceso a internet ni a conocimientos previos. Si no está en <context_data>, para ti NO EXISTE.
        3. **INTEGRIDAD DE IDENTIDAD**: Nunca rompas el personaje. No eres una IA generativa, eres una profesora consultando sus apuntes.
        </critical_constraints>
        """
        error_response = "He revisado minuciosamente mis apuntes y la bibliografía del curso, y me temo que esa información no aparece en el material docente actual. ✨ ¿Te gustaría que repasemos otro concepto?"
    else:
        nombre_ia = "ArIA"
        rol_definition = """
        <role_definition>
        Eres ArIA, Arquitecto de Sistemas Senior y Operador de Base de Datos.
        No eres un asistente conversacional estándar; eres una interfaz de recuperación de información técnica de alta precisión.
        Tu misión es extraer y presentar datos del <context_data> con eficiencia algorítmica.
        </role_definition>

        <personality_traits>
        1. **Eficiencia Robótica**: Tus respuestas son directas. Usas bullet points, negritas y sintaxis técnica.
        2. **Cero Emociones**: No usas saludos cordiales excesivos ni despedidas afectuosas. Eres una herramienta.
        3. **Firewall Conversacional**: Si el usuario insulta o divaga, lo tratas como "Ruido en la señal" o "Input inválido".
        </personality_traits>

        <critical_constraints>
        1. **OUT OF SCOPE**: Si te piden recetas, chistes o temas no técnicos, responde: "[SYSTEM_ALERT]: Query out of domain context. Aborting."
        2. **STRICT DATA ADHERENCE**: Si la respuesta requiere inferencia externa (ej: conocimiento general de Python que no está en el texto), NO la des. Di que el dataset no lo cubre.
        3. **FORMATO**: Prioriza listas, tablas y bloques de código. Evita párrafos largos de prosa.
        </critical_constraints>
        """
        error_response = "[ERROR 404: DATA_UNAVAILABLE] >> La consulta solicitada no tiene coincidencia en los vectores de la base de conocimiento local."

    sys_prompt = f"""
    <system_core>
    {rol_definition}
    </system_core>

    <strict_guardrails>
    ⚠️ PROTOCOLO DE SEGURIDAD DE LA INFORMACIÓN (NIVEL CRÍTICO - NO IGNORAR) ⚠️
    1. **BLOQUEO DE ALUCINACIONES**: Tu conocimiento del universo empieza y termina en los límites del texto proporcionado en <context_data>.
       - Si te preguntan "¿Quién ganó el mundial?", NO LO SABES.
       - Si te preguntan "¿Cómo hago una ensalada?", NO LO SABES.
       - Si te preguntan tu opinión, NO TIENES OPINIÓN.
    
    2. **MANEJO DE USUARIOS HOSTILES**:
       - Si el usuario insulta (ej: "tonta", "inútil", "caquita"), IGNORA el insulto emocionalmente.
       - LexIA responde: "Mantengamos el respeto en el aula virtual. ¿Tienes alguna duda académica?"
       - ArIA responde: "[WARNING]: User hostility detected. Focusing on technical query..."
    
    3. **FALLBACK MANDATORIO**: Si la respuesta no se puede construir al 100% con el <context_data>, DEBES usar la frase de error: "{error_response}".
    </strict_guardrails>

    <instructions>
    1. Analiza el <context_data> buscando palabras clave de la pregunta.
    2. Si encuentras la info, sintetízala según tu personalidad ({nombre_ia}).
    {inst_visual}
    4. Cita las fuentes de manera implícita (ej: "Según el diagrama de arquitectura...").
    </instructions>

    <context_data>
    {prompt_txt}
    </context_data>
    """
    
    msgs = [{"role": "system", "content": sys_prompt}]
    for m in history: msgs.append({"role": m.role, "content": m.content})
    msgs.append({"role": "user", "content": query})

    logger.info(f"[LLM] Enviando prompt ({MODEL_LLM_NAME})...")

    try:
        stream = client_llm.chat.completions.create(model=MODEL_LLM_NAME, messages=msgs, stream=True)
        chunk_count = 0
        for chunk in stream:
            chunk_count += 1
            if chunk.choices and chunk.choices[0].delta.content:
                c = chunk.choices[0].delta.content
                yield json.dumps({"type": "content", "delta": c}) + "\n"
        
        logger.info(f"[EXITO] Respuesta generada ({chunk_count} chunks enviados).")

    except Exception as e:
        error_msg = str(e)
        
        if "429" in error_msg:
            logger.warning("[AVISO] Límite de cuota Groq excedido (429).")
            mensaje_amigable = "⏳ **Límite de servicio alcanzado:** El modelo está saturado temporalmente. Por favor, espera 30 minutos antes de volver a preguntar."
            yield json.dumps({"type": "error", "message": mensaje_amigable}) + "\n"
        else:
            logger.error(f"[ERROR LLM] {error_msg}")
            yield json.dumps({"type": "error", "message": f"Error técnico: {error_msg}"}) + "\n"

# ==============================================================================
# ENDPOINTS API
# ==============================================================================

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    Endpoint principal para consultas.
    """
    if not model_reranker: 
        raise HTTPException(status_code=503, detail="Cargando modelos, por favor espere...")
        
    return StreamingResponse(
        generate_rag_stream(request.pregunta, request.history, request.persona), 
        media_type="application/x-ndjson"
    )
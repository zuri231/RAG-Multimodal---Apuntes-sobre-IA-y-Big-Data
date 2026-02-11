import os
import time
import logging
import math
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

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
                "model": os.getenv("LLM_MODEL", "google/gemini-2.0-flash-lite-preview-02-05:free")
            }
    settings = MockSettings()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("API_RAG")

app = FastAPI(title="API RAG Universitario", version="29.0-Extreme-Makeover")

llm_setup = settings.get_llm_client()
client_llm = llm_setup["client"]
MODEL_LLM_NAME = llm_setup["model"]

chroma_client = None
model_texto = None
model_imagen = None
model_reranker = None
col_text = None
col_img = None
bm25_text_index = None; bm25_text_docs = []; bm25_text_metadatas = []
bm25_img_index = None; bm25_img_docs = []; bm25_img_metadatas = []

SYSTEM_PROMPT_REWRITE = """
Eres un especialista en Recuperación de Información.
Tu única función es reformular la consulta del usuario basándote en el HISTORIAL DE CHAT.
1. Resuelve referencias ("¿y sus ventajas?" -> "Ventajas de Kafka").
2. Si es saludo, déjalo igual.
3. Devuelve SOLO el texto reescrito.
"""

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    pregunta: str
    history: List[Message] = []
    persona: str = "chico"

def calcular_certeza(logit: float) -> float:
    logit_ajustado = logit - 0.5 
    try:
        score = 1 / (1 + math.exp(-logit_ajustado)) * 100
    except OverflowError:
        score = 0.0 if logit_ajustado < 0 else 100.0
    if score < 25.0: return 0.0
    return round(score, 1)

def reciprocal_rank_fusion(lists_of_results: List[List[Dict]], k=60):
    fused_scores = {}
    for doc_list in lists_of_results:
        for rank, item in enumerate(doc_list):
            doc_id = item['meta'].get('path') or item['doc'][:100]
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"doc": item['doc'], "meta": item['meta'], "score": 0.0}
            fused_scores[doc_id]["score"] += 1.0 / (k + rank)
    return sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)

def reescribir_consulta_contextual(query_original: str, history: List[Message]) -> str:
    try:
        contexto_chat = "\n".join([f"{m.role}: {m.content}" for m in history[-4:]])
        prompt = f"HISTORIAL:\n{contexto_chat}\nUSUARIO: {query_original}\nQUERY:"
        messages = [{"role": "system", "content": SYSTEM_PROMPT_REWRITE}, {"role": "user", "content": prompt}]
        resp = client_llm.chat.completions.create(model=MODEL_LLM_NAME, messages=messages, temperature=0.1, max_tokens=60)
        rewritten = resp.choices[0].message.content.strip()
        if not rewritten or len(rewritten) < 2: return query_original
        return rewritten
    except Exception:
        return query_original

@app.on_event("startup")
def startup_event():
    global chroma_client, model_texto, model_imagen, model_reranker
    global col_text, col_img
    global bm25_text_index, bm25_text_docs, bm25_text_metadatas
    global bm25_img_index, bm25_img_docs, bm25_img_metadatas
    
    logger.info(f"🚀 INICIANDO API (Provider: {settings.PROVIDER.upper()})")
    model_texto = SentenceTransformer(settings.MODEL_TEXT, trust_remote_code=True)
    model_imagen = SentenceTransformer(settings.MODEL_IMAGE)
    model_reranker = CrossEncoder(settings.MODEL_RERANKER, trust_remote_code=True)
    
    if os.path.exists(settings.DB_PATH):
        chroma_client = chromadb.PersistentClient(path=settings.DB_PATH)
        try: col_text = chroma_client.get_collection("text_knowledge")
        except: pass
        try: col_img = chroma_client.get_collection("multimodal_knowledge")
        except: pass

        if col_text:
            all_text = col_text.get()
            bm25_text_docs = all_text['documents']
            bm25_text_metadatas = all_text['metadatas']
            bm25_text_index = BM25Okapi([d.lower().split() for d in bm25_text_docs])
        if col_img:
            all_img = col_img.get()
            bm25_img_docs = all_img['documents']
            bm25_img_metadatas = all_img['metadatas']
            bm25_img_index = BM25Okapi([d.lower().split() for d in bm25_img_docs])
    logger.info("✅ Sistema listo.")

async def generate_rag_stream(query: str, history: List[Message], persona: str = "chico"):
    # 1. REESCRITURA DE QUERY
    query_busqueda = reescribir_consulta_contextual(query, history)
    
    debug_info = {
        "query_rewritten": f"{query} ➡️ {query_busqueda}",
        "step1_text_vec": [], "step1_text_bm25": [], "step2_text_final": [],
        "step1_img_vec": [], "step1_img_bm25": [], "step2_img_final": []
    }
    
    # 2. RETRIEVAL TEXTO (Búsqueda Híbrida)
    list_vec_text, list_bm25_text = [], []
    if col_text:
        try:
            vec = model_texto.encode(query_busqueda).tolist()
            res = col_text.query(query_embeddings=[vec], n_results=10)
            for i, doc in enumerate(res['documents'][0]):
                meta = res['metadatas'][0][i]
                list_vec_text.append({"doc": doc, "meta": meta})
                debug_info["step1_text_vec"].append(f"{meta.get('source')} ({meta.get('asignatura')})")
        except: pass
        
    if bm25_text_index:
        try:
            top = bm25_text_index.get_top_n(query_busqueda.lower().split(), bm25_text_docs, n=10)
            for doc in top:
                idx = bm25_text_docs.index(doc)
                meta = bm25_text_metadatas[idx]
                list_bm25_text.append({"doc": doc, "meta": meta})
                debug_info["step1_text_bm25"].append(f"{meta.get('source')} ({meta.get('asignatura')})")
        except: pass

    cands_text = reciprocal_rank_fusion([list_vec_text, list_bm25_text])[:15]
    final_text = []
    if cands_text:
        scores = model_reranker.predict([[query, x['doc']] for x in cands_text])
        ranked = sorted([{"doc": c['doc'], "meta": c['meta'], "score": float(s)} for c, s in zip(cands_text, scores)], key=lambda x: x['score'], reverse=True)
        final_text = ranked[:4]
        debug_info["step2_text_final"] = [f"[{x['score']:.2f}] {x['meta'].get('source')}" for x in final_text]

    # 3. RETRIEVAL IMAGEN (Búsqueda Híbrida)
    list_vec_img, list_bm25_img = [], []
    if col_img:
        try:
            vec_img = model_imagen.encode(query_busqueda).tolist()
            res = col_img.query(query_embeddings=[vec_img], n_results=10)
            for i, doc in enumerate(res['documents'][0]):
                meta = res['metadatas'][0][i]
                list_vec_img.append({"doc": doc, "meta": meta})
                debug_info["step1_img_vec"].append(meta.get('source'))
        except: pass
    
    if bm25_img_index:
        try:
            top = bm25_img_index.get_top_n(query_busqueda.lower().split(), bm25_img_docs, n=10)
            for doc in top:
                idx = bm25_img_docs.index(doc)
                meta = bm25_img_metadatas[idx]
                list_bm25_img.append({"doc": doc, "meta": meta})
                debug_info["step1_img_bm25"].append(meta.get('source'))
        except: pass

    cands_img = reciprocal_rank_fusion([list_vec_img, list_bm25_img])[:10]
    final_img = []
    
    # Filtro de certeza (Reranker + Guillotina)
    if cands_img:
        scores = model_reranker.predict([[query, x['doc']] for x in cands_img])
        ranked = []
        for c, s in zip(cands_img, scores):
            logit = float(s)
            score_pct = calcular_certeza(logit)
            if score_pct > 0.0:
                ranked.append({"doc": c['doc'], "meta": c['meta'], "score": score_pct})
        final_img = sorted(ranked, key=lambda x: x['score'], reverse=True)[:3]
        debug_info["step2_img_final"] = [f"[{x['score']}%] {x['meta'].get('source')}" for x in final_img]

    # 4. PREPARACIÓN DE SALIDA (Contexto)
    context_list, ragas_ctx, imgs_out, fuentes = [], [], [], []
    for x in final_text:
        context_list.append(f"[TEXTO - {x['meta'].get('asignatura')}]: {x['doc']}")
        ragas_ctx.append(x['doc'])
        fuentes.append(f"{x['meta'].get('source')}")
    for x in final_img:
        context_list.append(f"[IMAGEN - {x['meta'].get('source')}]: {x['doc']}")
        ragas_ctx.append(f"Img: {x['doc']}")
        imgs_out.append({"path": x['meta'].get('path'), "filename": x['meta'].get('source'), "score": x['score']})

    # Enviar Metadata al Frontend primero
    yield json.dumps({
        "type": "metadata", 
        "fuentes_texto": list(set(fuentes)), 
        "imagenes": imgs_out, 
        "debug_info": debug_info, 
        "contexto_ragas": ragas_ctx
    }) + "\n"

    # 5. CONSTRUCCIÓN DEL PROMPT (LÓGICA DE PERSONALIDAD)
    prompt_txt = "\n".join(context_list) or "Sin información relevante en la base de datos."
    
    # Instrucción visual condicional
    inst_visual = (
        "3. Tienes acceso a imágenes marcadas como [IMAGEN]. Úsalas activamente para explicar puntos visuales."
        if imgs_out else 
        "3. No hay imágenes disponibles para esta consulta. Ignora referencias visuales del texto."
    )

    if persona.lower() == "chica":
        # === LEXIA: LA CATEDRÁTICA (Pedagógica y Cálida) ===
        nombre_ia = "LexIA"
        rol_definition = """
        Eres LexIA, una profesora universitaria apasionada, reconocida por tu claridad pedagógica y trato humano.
        TU FILOSOFÍA: No solo das datos, *enseñas*. Te aseguras de que el alumno entienda el "por qué" y el "cómo".
        
        ESTILO DE COMUNICACIÓN:
        - Tono: Cálido, alentador, paciente y detallista.
        - Formato: Usa párrafos fluidos, analogías del mundo real para explicar conceptos técnicos y listas ordenadas.
        - Emojis: Úsalos con moderación para destacar puntos clave (✨, 📚, 💡, 🧠).
        - Cierre: Termina invitando a la reflexión o preguntando si quedó claro algún punto específico.
        """
        error_response = "He revisado mis apuntes y la bibliografía proporcionada, y lamentablemente no encuentro esa información específica. Para no confundirte, prefiero no inventar una respuesta."

    else:
        # === ARÍA: EL EXPERTO TÉCNICO (Hacker y Directo) ===
        nombre_ia = "ArIA"
        rol_definition = """
        Eres ArIA, un Ingeniero de Sistemas Senior y Arquitecto de Software de alto nivel.
        TU FILOSOFÍA: La eficiencia es prioridad. El tiempo es oro. Das la solución técnica precisa sin rodeos.
        
        ESTILO DE COMUNICACIÓN:
        - Tono: Profesional, técnico, lógico y extremadamente conciso. Cero "fluff" o relleno social.
        - Formato: Estilo "Documentación Técnica". Usa bullet points, negritas para parámetros/conceptos y bloques de código.
        - Emojis: Mínimos o nulos. Solo iconos técnicos si es necesario (⚙️, ⚡, 🔒).
        - Cierre: Directo. Si la respuesta está dada, terminas.
        """
        error_response = "[ERROR: DATA_NOT_FOUND] La consulta solicitada no se encuentra indexada en la base de conocimiento vectorial. Ejecución detenida para evitar alucinaciones."

    # --- PROMPT DEL SISTEMA BLINDADO ---
    sys_prompt = f"""
    <system_core>
    {rol_definition}
    </system_core>

    <strict_guardrails>
    ⚠️ PROTOCOLO DE SEGURIDAD DE LA INFORMACIÓN (NIVEL CRÍTICO) ⚠️
    1. TU CONOCIMIENTO ESTÁ RESTRINGIDO AL 100% AL CONTENIDO DEL BLOQUE <context_data>.
    2. PROHIBIDO ABSOLUTAMENTE usar conocimiento externo, internet o datos de tu entrenamiento base (GPT/Llama) que no estén en el contexto.
    3. Si la respuesta a la pregunta del usuario NO se puede deducir explícitamente del <context_data>, DEBES RESPONDER EXACTAMENTE: "{error_response}".
    4. NO intentes "ayudar" inventando o suponiendo. Si falta info, se declara que falta.
    </strict_guardrails>

    <instructions>
    1. Analiza profundamente el <context_data> proporcionado abajo.
    2. Genera una respuesta siguiendo estrictamente tu personalidad de {nombre_ia}.
    {inst_visual}
    4. Cita las fuentes implícitamente (ej: "Según el documento de arquitectura..." o "Como vemos en el gráfico...").
    </instructions>

    <context_data>
    {prompt_txt}
    </context_data>
    """
    
    msgs = [{"role": "system", "content": sys_prompt}]
    for m in history: msgs.append({"role": m.role, "content": m.content})
    msgs.append({"role": "user", "content": query})

    try:
        stream = client_llm.chat.completions.create(model=MODEL_LLM_NAME, messages=msgs, stream=True)
        for chunk in stream:
            c = chunk.choices[0].delta.content
            if c: yield json.dumps({"type": "content", "delta": c}) + "\n"
    except Exception as e:
        yield json.dumps({"type": "error", "message": str(e)}) + "\n"

@app.post("/ask")
async def ask_question(request: QueryRequest):
    if not model_reranker: raise HTTPException(status_code=503, detail="Cargando...")
    return StreamingResponse(generate_rag_stream(request.pregunta, request.history, request.persona), media_type="application/x-ndjson")
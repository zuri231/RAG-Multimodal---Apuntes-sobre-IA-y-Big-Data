"""
================================================================================
EVALUACION DE RETRIEVAL VISUAL (TORNEO DE MODELOS)
================================================================================
   Script de benchmarking para medir la eficacia del sistema RAG Multimodal.
   Simula usuarios humanos describiendo imágenes y comprueba si el sistema
   es capaz de encontrar la imagen original basándose en esa descripción.

MECANICA DEL TORNEO:
    1. Muestreo: Selecciona imágenes aleatorias de la base de datos.
    2. Simulación (El Juez): Un modelo de Visión (VLM) mira la foto y genera
       una "búsqueda de usuario" (ej: "Gráfico de barras de ventas").
    3. Búsqueda: El modelo competidor (CLIP) vectoriza esa frase y busca en la BD.
    4. Veredicto: Si la imagen original aparece en el Top-3, es un ACIERTO.

UTILIDAD:
    - Comparar si funciona mejor la búsqueda en español o en inglés.
    - Validar si el modelo CLIP está alineado con las descripciones generadas.
================================================================================
"""

import os
import random
import logging
import chromadb
import ollama
from tqdm import tqdm
from PIL import Image
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ==============================================================================
# CONFIGURACION Y LOGS
# ==============================================================================
load_dotenv()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("evaluacion_visual")
VLM_JUDGE = "llava-phi3" 

MODELO_A = {
    "nombre": "Modelo A (CLIP Base - Spanish DB)",
    "db_path": "./chroma_db_multimodal(casa_llava_qwen)buena_spanish", 
    "embedding_model": "clip-ViT-B-32",
    "collection_name": "multimodal_knowledge"
}

MODELO_B = {
    "nombre": "Modelo B (CLIP Large - English DB)",
    "db_path": "./chroma_db_multimodal(casa_llava_qwen)buena_no_spanish", 
    "embedding_model": "clip-ViT-B-32",
    "collection_name": "multimodal_knowledge"
}

# ==============================================================================
# FUNCIONES AUXILIARES (EL JUEZ)
# ==============================================================================

def generar_busqueda_simulada(ruta_imagen):
    if not os.path.exists(ruta_imagen):
        return None
    
    prompt = (
        "Imagina que eres un estudiante buscando este diagrama o foto en Google. "
        "Escribe una frase de búsqueda CORTA y precisa (en español) para encontrarla. "
        "Solo la frase, sin explicaciones."
    )
    
    try:
        res = ollama.chat(model=VLM_JUDGE, messages=[
            {'role': 'user', 'content': prompt, 'images': [ruta_imagen]}
        ])

        return res['message']['content'].strip().replace('"', '')
    except Exception as e:
        logger.warning(f"[ALERTA] Fallo del Juez VLM en {ruta_imagen}: {e}")
        return None

# ==============================================================================
# LOGICA DE EVALUACION
# ==============================================================================

def evaluar_retrieval_visual(config):
    print(f"\n{'='*60}")
    print(f" EVALUANDO BASE: {config['nombre']}")
    print(f"{'='*60}")

    if not os.path.exists(config['db_path']):
        logger.error(f"[ERROR] Ruta DB no encontrada: {config['db_path']}")
        return 0

    try:
        client = chromadb.PersistentClient(path=config['db_path'])
        collection = client.get_collection(name=config['collection_name'])
        model = SentenceTransformer(config['embedding_model'], trust_remote_code=True)
    except Exception as e:
        logger.error(f"[ERROR] Inicializando recursos: {e}")
        return 0

    datos = collection.get()
    if not datos['ids']:
        logger.warning("[AVISO] La colección está vacía.")
        return 0
    items_validos = []
    for i, meta in enumerate(datos['metadatas']):
        if meta and 'path' in meta and os.path.exists(meta['path']):
            items_validos.append((datos['ids'][i], meta['path']))

    if not items_validos:
        logger.warning("[AVISO] No se encontraron rutas de imágenes válidas en los metadatos.")
        return 0

    sample_size = min(20, len(items_validos))
    muestra = random.sample(items_validos, sample_size)
    
    aciertos = 0
    print(f" Realizando {sample_size} pruebas ciegas...")

    for doc_id_real, ruta_img in tqdm(muestra, desc="Evaluando"):
        
        query_simulada = generar_busqueda_simulada(ruta_img)
        if not query_simulada: 
            continue
        
        query_emb = model.encode(query_simulada).tolist()
        
        res = collection.query(query_embeddings=[query_emb], n_results=3)
        
        if doc_id_real in res['ids'][0]:
            aciertos += 1
        else:
            pass

    score = (aciertos / sample_size) * 100
    print(f"\n TASA DE ACIERTO (Recall@3): {score:.2f}%")
    return score

# ==============================================================================
# EJECUCION PRINCIPAL
# ==============================================================================

def main():
    logger.info(" Verificando disponibilidad del Juez Visual...")
    os.system(f"ollama pull {VLM_JUDGE}")

    # Evaluar Modelo A
    score_a = evaluar_retrieval_visual(MODELO_A)
    
    # Evaluar Modelo B

    score_b = 0 

    print("\n" + "="*40)
    print(" TABLA DE RESULTADOS FINAL")
    print("="*40)
    print(f" 1. {MODELO_A['nombre']}: {score_a:.2f}%")
    if score_b > 0:
        print(f" 2. {MODELO_B['nombre']}: {score_b:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
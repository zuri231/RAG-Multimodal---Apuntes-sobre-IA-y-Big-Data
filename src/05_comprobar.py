"""
================================================================================
EVALUACION A/B: IMPACTO DEL IDIOMA EN RETRIEVAL DE IMAGENES
================================================================================
   Script de benchmarking diseñado para validar la hipótesis:
   "Traducir las descripciones de las imágenes al español mejora la 
   capacidad de recuperación cuando las consultas son en español".

METODOLOGIA:
    1. Se configuran dos bases de datos competidoras:
       - MODELO A: Descripciones procesadas en Español.
       - MODELO B: Descripciones originales (Inglés/Sin procesar).
    2. Se seleccionan imágenes aleatorias.
    3. Un Juez (VLM) genera una consulta de búsqueda simulada en ESPAÑOL.
    4. Se lanza esa misma consulta en español contra ambas bases de datos.
    5. Se compara la Tasa de Acierto (Hit Rate) de cada una.

HIPOTESIS ESPERADA:
    El Modelo A debería superar al modelo B
================================================================================
"""

import os
import random
import logging
import chromadb
import ollama
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ==============================================================================
# CONFIGURACION Y LOGS
# ==============================================================================
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("evaluacion_multimodal")
VLM_JUDGE = "llava-phi3" 

MODELO_A_CONFIG = {
    "nombre": "Base de Datos (Descripciones en ESPAÑOL)",
    "db_path": "./chroma_db_multimodal(casa_llava_qwen)buena_spanish",
    "embedding_model": "clip-ViT-B-32",
    "collection_name": "multimodal_knowledge"
}

MODELO_B_CONFIG = {
    "nombre": "Base de Datos (Descripciones en INGLES/RAW)",
    "db_path": "./chroma_db_multimodal(casa_llava_qwen)buena_no_spanish",
    "embedding_model": "clip-ViT-B-32",
    "collection_name": "multimodal_knowledge"
}

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def generar_consulta_espanol(ruta_imagen):
    if not os.path.exists(ruta_imagen):
        return None
    
    prompt = (
        "Imagina que eres un estudiante hispanohablante buscando esta imagen en un buscador. "
        "Escribe una frase de búsqueda CORTA y EXCLUSIVAMENTE EN ESPAÑOL que usarías para encontrarla. "
        "Solo la frase."
    )
    
    try:
        res = ollama.chat(model=VLM_JUDGE, messages=[
            {'role': 'user', 'content': prompt, 'images': [ruta_imagen]}
        ])
        return res['message']['content'].strip().replace('"', '').replace('.', '')
    except Exception as e:
        logger.warning(f"Error generando consulta con VLM: {e}")
        return None

# ==============================================================================
# MOTOR DE EVALUACION
# ==============================================================================

def evaluar_base_datos(config):
    print(f"\n{'='*60}")
    print(f" EVALUANDO: {config['nombre']}")
    print(f" DB Path:   {config['db_path']}")
    print(f"{'='*60}")
    
    if not os.path.exists(config['db_path']):
        logger.error("Ruta de base de datos no encontrada.")
        return 0.0

    try:
        client = chromadb.PersistentClient(path=config['db_path'])
        collection = client.get_collection(name=config['collection_name'])
        model = SentenceTransformer(config['embedding_model'])
    except Exception as e:
        logger.error(f"Error inicializando modelo/DB: {e}")
        return 0.0

    datos = collection.get()
    if not datos['ids']:
        logger.warning("La base de datos está vacía.")
        return 0.0
    items_validos = []
    for i, meta in enumerate(datos['metadatas']):
        if meta and 'path' in meta and os.path.exists(meta['path']):
            items_validos.append((datos['ids'][i], meta['path']))

    if not items_validos:
        logger.warning("No se encontraron rutas de imágenes válidas.")
        return 0.0

    sample_size = min(20, len(items_validos))
    muestra = random.sample(items_validos, sample_size)
    
    aciertos = 0
    
    print(f" Realizando {sample_size} pruebas de recuperación con consultas en ESPAÑOL...")
    
    for doc_id_real, ruta_img in tqdm(muestra, desc="Progreso"):
        query_es = generar_consulta_espanol(ruta_img)
        if not query_es:
            continue

        query_emb = model.encode(query_es).tolist()

        resultados = collection.query(
            query_embeddings=[query_emb],
            n_results=3
        )

        if doc_id_real in resultados['ids'][0]:
            aciertos += 1
    
    score = (aciertos / sample_size) * 100
    print(f" TASA DE ACIERTO: {score:.2f}%")
    return score

# ==============================================================================
# EJECUCION PRINCIPAL
# ==============================================================================

def main():
    print(" Verificando disponibilidad de modelo VLM (Ollama)...")
    os.system(f"ollama pull {VLM_JUDGE}")

    score_a = evaluar_base_datos(MODELO_A_CONFIG)
    score_b = evaluar_base_datos(MODELO_B_CONFIG)
    
    print("\n" + "="*50)
    print(" RESULTADOS DEL EXPERIMENTO DE IDIOMA")
    print("="*50)
    print(f" 1. {MODELO_A_CONFIG['nombre']}: {score_a:.2f}%")
    print(f" 2. {MODELO_B_CONFIG['nombre']}: {score_b:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    main()
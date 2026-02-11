"""
================================================================================
BENCHMARK DE ESTRATEGIAS RAG (CHUNK SIZE vs RERANKING)
================================================================================
   Script de evaluación cuantitativa para determinar la configuración óptima
   de la base de datos vectorial.

OBJETIVO:
   Comparar empíricamente cómo afectan dos variables al rendimiento del sistema:
   1. Tamaño del Chunk (300, 500, 800, 1000 tokens).
   2. Uso de Reranker (Cross-Encoder) vs. Búsqueda Vectorial Pura.

METRICAS:
   - Hit Rate @3: Porcentaje de veces que el documento correcto aparece en el Top 3.
   - MRR @3 (Mean Reciprocal Rank): Mide qué tan arriba aparece el resultado correcto.
     (1.0 = posición 1; 0.5 = posición 2; 0.33 = posición 3).
   - Latencia: Tiempo promedio de procesamiento por consulta.
================================================================================
"""

import time
import os
import pandas as pd
import logging
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# ==============================================================================
# CONFIGURACION Y LOGS
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("benchmark_rag")
MODELO_EMB = "Qwen/Qwen3-Embedding-0.6B"
MODELO_RERANKER = "BAAI/bge-reranker-v2-m3"
GOLDEN_DATASET = [
    # --- BIG DATA & KAFKA ---
    {
        "q": "¿Que es Kafka?", 
        "expected_doc": "apuntes_kafka.pdf"
    },
    {
        "q": "¿Cuales son los componentes de la arquitectura Kafka?", 
        "expected_doc": "apuntes_kafka.pdf" 
    },
    {
        "q": "¿Como funciona un broker en Kafka?", 
        "expected_doc": "apuntes_kafka.pdf" 
    },
    {
        "q": "¿Cuales son los componentes de Hadoop?", 
        "expected_doc": "introduccion_hadoop.pdf" 
    },

    # --- BASES DE DATOS (NoSQL / SQL) ---
    {
        "q": "¿Que es un indice en MongoDB?", 
        "expected_doc": "MongoDB.pdf"
    },
    {
        "q": "¿Que es un raid?", 
        "expected_doc": "almacenamiento_de_datos.pdf" 
    },
    {
        "q": "¿Que tipos de datos tiene Hive?", 
        "expected_doc": "apache_hive.pdf"
    },

    # --- MACHINE LEARNING (Supervisado) ---
    {
        "q": "¿Cual es el objetivo del aprendizaje supervisado?", 
        "expected_doc": "Explicacion_Modelos-Supervisado.pdf"
    },
    {
        "q": "¿Tipos de clasificación de la regresión logistica?", 
        "expected_doc": "Regresión Logística 2025-2026.pdf"
    },
    {
        "q": "¿Cuando se detiene un arbol de decision?", 
        "expected_doc": "arbol de decision_2025_2026.pdf"
    },

    # --- MACHINE LEARNING (No Supervisado / Deep Learning) ---
    {
        "q": "¿Cual es el objetivo del aprendizaje no supervisado?", 
        "expected_doc": "Kmeans_no_supervisado.pdf"
    },
    {
        "q": "¿Que es un centroide en K-Means?", 
        "expected_doc": "Kmeans_no_supervisado.pdf"
    },
    {
        "q": "Que es la convolucion?", 
        "expected_doc": "CNN_2025_2026.pdf"
    }
]
BASES_DE_DATOS = ["db_300", "db_500", "db_800", "db_1000"]

# ==============================================================================
# MOTOR DE EVALUACION
# ==============================================================================

def evaluar_configuracion(nombre_config, db_folder_path, usar_reranker=False):
    logger.info(f"Evaluando: {nombre_config}...")
    if not os.path.exists(db_folder_path):
        logger.error(f"No se encuentra la carpeta: {db_folder_path}")
        return {
            "Configuración": nombre_config,
            "Hit Rate @3": "N/A",
            "MRR @3": 0.0,
            "Latencia (s)": 0.0
        }
    try:
        client = chromadb.PersistentClient(path=db_folder_path)
        coleccion = client.get_collection("text_knowledge")
        model = SentenceTransformer(MODELO_EMB, trust_remote_code=True)
        
        reranker = None
        if usar_reranker:
            reranker = CrossEncoder(MODELO_RERANKER)
            
    except Exception as e:
        logger.error(f"Error inicializando recursos para {nombre_config}: {e}")
        return {
            "Configuración": nombre_config,
            "Hit Rate @3": "Error",
            "MRR @3": 0.0,
            "Latencia (s)": 0.0
        }

    hits = 0
    mrr_sum = 0
    start_time = time.time()

    for item in GOLDEN_DATASET:
        query = item["q"]
        target = item["expected_doc"].lower()

        emb_query = model.encode(query).tolist()
        res = coleccion.query(query_embeddings=[emb_query], n_results=10)

        if not res["documents"]:
            continue

        metas = res["metadatas"][0]
        docs = res["documents"][0]
        top_sources = []

        if usar_reranker and reranker:
            pares = [[query, d] for d in docs]
            scores = reranker.predict(pares)
            
            ranking = sorted(zip(metas, scores), key=lambda x: x[1], reverse=True)
            
            top_sources = [m.get("source", "").lower() for m, s in ranking[:3]]
        else:
            top_sources = [m.get("source", "").lower() for m in metas[:3]]

        found_at_index = -1
        
        for idx, source_found in enumerate(top_sources):
            nombre_archivo = os.path.basename(source_found)
            if target in nombre_archivo:
                found_at_index = idx
                break
        
        if found_at_index != -1:
            hits += 1
            mrr_sum += 1 / (found_at_index + 1)

    total_cases = len(GOLDEN_DATASET)
    latencia_total = time.time() - start_time
    latencia_promedio = latencia_total / total_cases if total_cases > 0 else 0

    return {
        "Configuración": nombre_config,
        "Hit Rate @3": f"{(hits/total_cases)*100:.1f}%",
        "MRR @3": round(mrr_sum/total_cases, 2),
        "Latencia (s)": round(latencia_promedio, 3)
    }

# ==============================================================================
# EJECUCION PRINCIPAL
# ==============================================================================

def main():
    print("="*60)
    print(" INICIANDO BENCHMARK DE RENDIMIENTO RAG")
    print("="*60)
    print(f" Modelos: {MODELO_EMB} / {MODELO_RERANKER}")
    print(f" Casos de prueba: {len(GOLDEN_DATASET)}")
    print("-" * 60)

    resultados_finales = []

    for db_name in BASES_DE_DATOS:
        res_base = evaluar_configuracion(
            nombre_config=f"{db_name} (Base)", 
            db_folder_path=db_name, 
            usar_reranker=False
        )
        resultados_finales.append(res_base)

        res_rerank = evaluar_configuracion(
            nombre_config=f"{db_name} (+Reranker)", 
            db_folder_path=db_name, 
            usar_reranker=True
        )
        resultados_finales.append(res_rerank)

    df = pd.DataFrame(resultados_finales)
    
    print("\n" + "="*60)
    print(" RESULTADOS CONSOLIDADOS")
    print("="*60)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.to_string(index=False))
    print("="*60)
    try:
        df.to_csv("benchmark_results.csv", index=False)
        print("[INFO] Resultados guardados en 'benchmark_results.csv'")
    except:
        pass

if __name__ == "__main__":
    main()
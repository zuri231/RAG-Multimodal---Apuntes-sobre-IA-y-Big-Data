"""
================================================================================
EVALUACION DE CALIDAD GENERATIVA (METRICAS RAGAS)
================================================================================
   Script de auditoría de calidad que utiliza un "Juez IA" para evaluar
   la precisión y relevancia de las respuestas generadas por el sistema RAG.

METODOLOGIA (RAGAS - Retrieval Augmented Generation Assessment):
    1. Generación: Se envían preguntas de control (Golden Data) a la API.
    2. Recolección: Se capturan la respuesta generada y el contexto recuperado
       (fragmentos de PDF/Imágenes) mediante el stream de la API.
    3. Juicio: Un modelo LLM potente (el Juez) analiza la coherencia entre:
       - Pregunta vs. Respuesta (Relevancia)
       - Contexto vs. Respuesta (Fidelidad/Faithfulness)
       - Contexto vs. Ground Truth (Precisión del Contexto)

METRICAS:
    - Faithfulness: ¿La respuesta se inventa cosas o se basa en el contexto?
    - Answer Relevancy: ¿La respuesta contesta realmente a lo que se preguntó?
    - Context Precision: ¿El sistema recuperó la información correcta?
================================================================================
"""

import requests
import pandas as pd
import time
import json
import os
import logging
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


# ==============================================================================
# CONFIGURACION Y LOGS
# ==============================================================================
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("evaluacion_ragas")

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}/ask"
PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# ==============================================================================
# 1. CONFIGURACION DEL JUEZ (MODELO EVALUADOR)
# ==============================================================================

def configurar_juez():
    logger.info(f"Configurando Juez RAGAS (Modo: {PROVIDER.upper()})...")
    
    if PROVIDER == "groq":
        if not ChatGroq:
            logger.warning("La librería 'langchain-groq' no está instalada. Usando OpenAI como fallback.")
            return _configurar_openai()
            
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Falta GROQ_API_KEY en el archivo .env")
            
        logger.info("Juez: GROQ (Llama-3-70b) - Configurado para alto rendimiento.")
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0
        )
    else:
        return _configurar_openai()

def _configurar_openai():
    logger.info("Juez: OpenAI (GPT-4) - Configurado estándar.")
    return ChatOpenAI(model="gpt-4", temperature=0)

# ==============================================================================
# 2. DATOS DE PRUEBA (GOLDEN DATA)
# ==============================================================================

GOLDEN_DATA = [
    {
        "question": "¿Qué es Apache Kafka?",
        "ground_truth": "Apache Kafka es una plataforma distribuida de transmisión de datos que permite publicar, suscribirse, almacenar y procesar flujos de registros en tiempo real."
    },
    {
        "question": "¿Cuáles son los componentes principales de la arquitectura de Kafka?",
        "ground_truth": "Los componentes principales son Productores, Consumidores, Brokers, Topics, Particiones y ZooKeeper."
    },
    {
        "question": "¿Qué es el aprendizaje supervisado?",
        "ground_truth": "Es un tipo de aprendizaje automático donde el modelo se entrena con un conjunto de datos etiquetado."
    },
    {
        "question": "¿Qué diferencia hay entre regresión y clasificación?",
        "ground_truth": "La regresión predice valores continuos, mientras que la clasificación predice categorías."
    },
    {
        "question": "¿Qué muestra el diagrama de la imagen acid.png?",
        "ground_truth": "Muestra un esquema relacionado con las propiedades ACID en bases de datos."
    }
]

# ==============================================================================
# 3. FUNCIONES DE EJECUCION
# ==============================================================================

def obtener_respuestas_sistema():
    """
    Itera sobre el Golden Data, envía las preguntas a la API y reconstruye
    la respuesta y el contexto desde el stream de datos.
    """
    questions, answers, contexts, ground_truths = [], [], [], []

    logger.info(f"Iniciando ciclo de preguntas ({len(GOLDEN_DATA)} casos)...")

    for i, item in enumerate(GOLDEN_DATA):
        q = item["question"]
        logger.info(f"[{i+1}/{len(GOLDEN_DATA)}] Preguntando: '{q}'")
        
        try:
            with requests.post(API_URL, json={"pregunta": q, "history": []}, stream=True, timeout=120) as response:
                
                if response.status_code == 200:
                    full_answer = ""
                    retrieved_context = []

                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line.decode('utf-8'))
                                
                                if chunk["type"] == "metadata":
                                    retrieved_context = chunk.get("contexto_ragas", [])
                                
                                elif chunk["type"] == "content":
                                    full_answer += chunk.get("delta", "")
                                    
                            except json.JSONDecodeError:
                                continue
                    
                    questions.append(q)
                    answers.append(full_answer)
                    
                    if not retrieved_context:
                        retrieved_context = ["Sin contexto recuperado."]
                    contexts.append(retrieved_context)
                    
                    ground_truths.append(item["ground_truth"])
                    logger.info("Respuesta recibida y procesada correctamente.")
                    
                else:
                    logger.error(f"Error HTTP API: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error de conexión con la API: {e}")

        time.sleep(1)

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }

def main():
    judge_llm = configurar_juez()
    
    logger.info("Cargando modelo de embeddings para métricas (Local)...")
    eval_embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
    
    data_dict = obtener_respuestas_sistema()
    
    if not data_dict["question"]:
        logger.error("No se obtuvieron datos para evaluar. Abortando.")
        return

    dataset = Dataset.from_dict(data_dict)

    max_workers = 2 if PROVIDER == "groq" else 1
    
    logger.info(f"El Juez IA está evaluando (Workers: {max_workers})...")

    try:
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=judge_llm,
            embeddings=eval_embeddings,
            raise_exceptions=False,
            run_config=RunConfig(max_workers=max_workers, timeout=180)
        )

        print("\n" + "="*50)
        print("RESULTADOS DE LA EVALUACION")
        print("="*50)
        print(results)
        print("="*50)
        
        filename = "evaluacion_ragas_resultados.xlsx"
        df = results.to_pandas()
        df.to_excel(filename, index=False)
        logger.info(f"Informe Excel generado: {filename}")

    except Exception as e:
        logger.error(f"Error crítico durante la evaluación RAGAS: {e}")
        pd.DataFrame(data_dict).to_excel("datos_crudos_emergencia.xlsx", index=False)
        logger.warning("Se guardaron los datos crudos en 'datos_crudos_emergencia.xlsx'.")

if __name__ == "__main__":
    main()
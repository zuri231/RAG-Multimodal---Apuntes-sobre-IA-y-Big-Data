# ====================================================
# 01_multimodal_ingest_smart.py - INGESTA DE IMÁGENES 
# ====================================================
"""
================================================================================
PIPELINE DE INGESTA MULTIMODAL (Imágenes + IA Generativa)
================================================================================
   Sistema avanzado de ingesta que procesa imágenes para RAG Multimodal:
    1. Escaneo: Detecta imágenes en carpetas anidadas (Asignatura/Tema).
    2. Visión (VLM): Usa un modelo de Visión (LLaVA/Phi-3) para "ver" y describir la imagen.
    3. Embedding: Vectoriza la imagen visualmente con CLIP.
    4. Almacenamiento: Guarda Vector + Descripción + Metadatos en ChromaDB.

FLUJO COMPLETO:
    Imagen en disco
        ↓
    [1] VLM (Ollama) → Genera descripción textual detallada (Captioning)
        ↓
    [2] CLIP Model → Genera embedding visual de la imagen
        ↓
    [3] Metadatos → Extrae Asignatura/Tema de la estructura de carpetas
        ↓
    [4] ChromaDB → Guarda todo para futura recuperación
================================================================================
"""
import os
import chromadb
from sentence_transformers import SentenceTransformer
from PIL import Image
import ollama
from tqdm import tqdm
from dotenv import load_dotenv
import logging
import torch 


# =============================================================================
# 1. CONFIGURACIÓN GENERAL
# =============================================================================

# Configuración de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# Variables de Configuración
DB_PATH = os.getenv("DB_PATH", "./chroma_db_multimodal(casa_llava_qwen)buena_spanish")
CLIP_MODEL = "clip-ViT-B-32"
VISION_MODEL = "llava" 
DATA_DIR = "./data/imagenes"


# =============================================================================
# 2. FUNCIÓN DE DESCRIPCIÓN DE IMÁGENES (MODELO VISUAL)
# =============================================================================

def describir_imagen(ruta):
    """
    Usa LLaVA (Ollama) para generar una descripción detallada en español.
    """
    try:
        PROMPT = """
        Actúa como un profesor experto de universidad hispanohablante. Analiza esta imagen con el MÁXIMO DETALLE posible.
        
        IMPORTANTE: TU RESPUESTA DEBE SER 100% EN ESPAÑOL.
        Si la imagen contiene texto en inglés, TRADÚCELO al español en tu explicación. NO escribas en inglés.
        
        Tu objetivo es que un estudiante ciego hispanohablante pueda entender perfectamente esta diapositiva o esquema.
        
        Sigue esta estructura para evaluar:
        1. TIPO DE IMAGEN: Indica si es código, diagrama, gráfico estadístico, diapositiva de texto o esquema conceptual.
        2. TEXTO LITERAL (TRADUCIDO): Transcribe el título y el texto importante, pero TRADÚCELO AL ESPAÑOL si está en otro idioma.
        3. ANÁLISIS VISUAL: Describe paso a paso las relaciones, flechas, colores y formas en español. Si es un gráfico, describe los ejes y la tendencia.
        4. EXPLICACIÓN TÉCNICA: Explica qué concepto de Inteligencia Artificial o Big Data se está enseñando aquí.
        
        REGLA DE ORO: Todo el output debe estar en CASTELLANO, incluso si el contenido original está en inglés.
        """

        response = ollama.chat(
            model=VISION_MODEL,
            messages=[{'role': 'user', 'content': PROMPT, 'images': [ruta]}]
        )
        return response['message']['content']

    except Exception as e:
        print(f"\n ERROR OLLAMA en {os.path.basename(ruta)}: {e}")
        return "Sin descripción"


# =============================================================================
# 3. PROGRAMA PRINCIPAL
# =============================================================================

def main():
    logger.info(f"Iniciando Ingesta con doble nivel (Asignatura/Tema) en {DATA_DIR}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Dispositivo seleccionado: {device.upper()}")
    
    if device == "cpu":
        logger.warning("CUIDADO: Se está usando CPU.")

    model = SentenceTransformer(CLIP_MODEL, device=device)
    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        client.delete_collection("multimodal_knowledge")
    except:
        pass
    
    collection = client.create_collection(
        name="multimodal_knowledge",
        metadata={"hnsw:space": "cosine"}
    )

    ids, embeddings, metadatas, documents = [], [], [], []
    archivos_encontrados = []

    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                
                ruta_completa = os.path.join(root, file)
                relativa = os.path.relpath(ruta_completa, DATA_DIR)
                partes = relativa.split(os.sep)
                
                asignatura = "general"
                tema = "general"
                
                if len(partes) > 1:
                    asignatura = partes[0]
                if len(partes) > 2:
                    tema = partes[1]
                
                archivos_encontrados.append({
                    "path": ruta_completa,
                    "filename": file,
                    "asignatura": asignatura,
                    "tema": tema
                })

    if not archivos_encontrados:
        logger.warning(f"No hay imágenes en {DATA_DIR}")
        return

    logger.info(f"Encontradas {len(archivos_encontrados)} imágenes.")

    seen_ids = set()

    for item in tqdm(archivos_encontrados, desc="Procesando"):
        try:
            base_id = f"img_{item['filename']}"
            unique_id = base_id
            counter = 1
            
            while unique_id in seen_ids:
                unique_id = f"{base_id}_{counter}"
                counter += 1
            
            seen_ids.add(unique_id)

            image = Image.open(item["path"])
            emb = model.encode(image).tolist()

            descripcion = describir_imagen(item["path"])

            ids.append(unique_id)
            embeddings.append(emb)
            documents.append(descripcion)
            metadatas.append({
                "type": "image",
                "path": item["path"],
                "source": item["filename"],
                "asignatura": item["asignatura"],
                "tema": item["tema"]
            })

        except Exception as e:
            logger.error(f"Error {item['filename']}: {e}")

    if ids:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        logger.info(f"Guardado. Total: {len(ids)}. DB en: {DB_PATH}")


# =============================================================================
# 4. EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    main()
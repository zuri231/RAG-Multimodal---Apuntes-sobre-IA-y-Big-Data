"""
================================================================================
PIPELINE DE INGESTA DE DOCUMENTOS (PDFs)
================================================================================
   Sistema ETL (Extract, Transform, Load) para procesar documentos académicos:
    1. Extract: Escanea carpetas recursivamente para encontrar PDFs.
    2. Transform (Metadata): Deduce Asignatura/Tema basándose en la ruta del archivo.
    3. Transform (Chunking): Trocea el texto en fragmentos manejables con solapamiento.
    4. Load: Vectoriza (Embeddings) y guarda en ChromaDB.

FLUJO COMPLETO:
    PDF en disco (ej: /BDA/Hadoop/comandos_hdfs.pdf)
        ↓
    [1] Loader (PyPDF) → Extrae texto crudo y metadatos básicos
        ↓
    [2] Enriquecimiento → Añade metadata: {asignatura: "BDA", tema: "Hadoop"}
        ↓
    [3] Splitter → Divide en chunks de 1000 caracteres (contexto ideal)
        ↓
    [4] Embedding Model → Convierte texto a vectores (Qwen/Qwen3-Embedding-0.6B)
        ↓
    [5] ChromaDB → Indexa vectores y metadatos para búsqueda semántica
================================================================================
"""
import os
import logging
import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ==============================================================================
# CONFIGURACIÓN Y LOGS
# ==============================================================================
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("ingesta_pdfs")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DB_PATH = os.getenv("DB_PATH", os.path.join(PROJECT_ROOT, "chroma_db_multimodal(casa_llava-multilingual-e5)"))
PDF_DIR = os.path.join(PROJECT_ROOT, "data", "pdfs")

MODELO_EMBEDDING = "Qwen/Qwen3-Embedding-0.6B"


def main():
  
    logger.info("="*60)
    logger.info(f"INICIANDO INGESTA DE PDFs")
    logger.info(f"Directorio: {PDF_DIR}")
    logger.info(f"Modelo: {MODELO_EMBEDDING}")
    logger.info("="*60)
    
    if not os.path.exists(PDF_DIR):
        logger.error(f"Error crítico: No existe la carpeta {PDF_DIR}")
        return

    # ====================================================================
    # PASO 1: EXTRACCIÓN Y ENRIQUECIMIENTO (Metadata Extraction)
    # ====================================================================
    
    docs = []
    total_archivos = 0
    
    print("\n Escaneando biblioteca de documentos...")
    
    for root, _, files in os.walk(PDF_DIR):
        for filename in files:
            if filename.lower().endswith(".pdf"):
                ruta_completa = os.path.join(root, filename)
                
                # --- LÓGICA DE METADATOS INTELIGENTE ---
                relativa = os.path.relpath(ruta_completa, PDF_DIR)
                partes = relativa.split(os.sep)
                
                asignatura = partes[0] if len(partes) > 1 else "general"
                tema = partes[1] if len(partes) > 2 else "general"
                
                try:
                    loader = PyPDFLoader(ruta_completa)
                    pdf_docs = loader.load()
                    for doc in pdf_docs:
                        doc.metadata["source"] = filename
                        doc.metadata["type"] = "text"
                        doc.metadata["asignatura"] = asignatura
                        doc.metadata["tema"] = tema
                        doc.metadata["path"] = relativa 
                    
                    docs.extend(pdf_docs)
                    total_archivos += 1
                    logger.info(f"   Leído: {asignatura} | {filename}")
                    
                except Exception as e:
                    logger.error(f"   Error leyendo {filename}: {e}")

    if total_archivos == 0:
        logger.warning(" No se encontraron PDFs. Revisa la carpeta 'data/pdfs'.")
        return

    # ====================================================================
    # PASO 2: CHUNKING (División de Texto)
    # ====================================================================
    # QUE HACE? Divide el texto en bloques de 1000 caracteres con 200 de solapamiento.
    
    logger.info(f"\n Troceando {len(docs)} páginas de documentos...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # Tamaño del fragmento
        chunk_overlap=200,  # Solapamiento para no cortar frases a la mitad
        separators=["\n\n", "\n", ". ", " ", ""] # Prioridad de corte
    )
    splits = text_splitter.split_documents(docs)
    
    logger.info(f"   - Total fragmentos (chunks) generados: {len(splits)}")

    # ====================================================================
    # PASO 3: EMBEDDING Y ALMACENAMIENTO (Vector Store)
    # ====================================================================
    logger.info(f"\n Preparando ChromaDB en: {DB_PATH}")
    
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name="text_knowledge",
        metadata={"hnsw:space": "cosine"}
    )

    logger.info(f"Cargando modelo en memoria ({MODELO_EMBEDDING})...")
    model = SentenceTransformer(MODELO_EMBEDDING)
    batch_size = 50
    total_batches = (len(splits) // batch_size) + 1
    
    print(f"\n Iniciando vectorización ({total_batches} lotes)...")
    
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i+batch_size]
        texts = [d.page_content for d in batch]
        metadatas = [d.metadata for d in batch]

        ids = [f"pdf_{i+j}" for j in range(len(batch))]

        embeddings = model.encode(texts).tolist()
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"   Lote {i//batch_size + 1}/{total_batches} procesado ({len(batch)} chunks)")

    logger.info("="*60)
    logger.info(" INGESTA DE PDFs COMPLETADA CORRECTAMENTE")
    logger.info("="*60)

if __name__ == "__main__":
    main()
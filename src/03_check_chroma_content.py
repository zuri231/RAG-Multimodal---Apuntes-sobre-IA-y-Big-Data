"""
================================================================================
HERRAMIENTA DE INSPECCION DE BASE DE DATOS (ChromaDB)
================================================================================
   Script de utilidad para auditar y verificar el contenido de la base de datos
   vectorial. Permite visualizar qué datos se han ingestado realmente.

FLUJO COMPLETO:
    1. Conexión: Se conecta a la base de datos ChromaDB persistente.
    2. Listado: Identifica todas las colecciones disponibles.
    3. Inspección: Para cada colección crítica ('multimodal' y 'text'):
       a. Cuenta el total de elementos.
       b. Recupera una muestra de documentos (IDs, Metadatos y Contenido).
       c. Muestra la información formateada en consola.

UTILIDAD:
    - Verificar si la ingesta (PDFs o Imágenes) funcionó correctamente.
    - Depurar problemas de metadatos faltantes.
    - Comprobar que los textos se han troceado (chunking) bien.
================================================================================
"""

import os
import chromadb
from dotenv import load_dotenv
import logging

# ==============================================================================
# CONFIGURACION Y LOGS
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# Rutas Dinámicas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.getenv("DB_PATH", os.path.join(PROJECT_ROOT, "chroma_db_multimodal"))

def inspeccionar_coleccion(client, nombre_col):
    print(f"\n{'='*60}")
    print(f"INSPECCIONANDO COLECCION: '{nombre_col}'")
    print(f"{'='*60}")
    
    try:
        collection = client.get_collection(nombre_col)
        
        count = collection.count()
        print(f"Total de documentos/imagenes: {count}")
        
        if count == 0:
            print("[AVISO] La coleccion esta vacia.")
            return
        data = collection.get(limit=30)
        
        ids = data['ids']
        metadatas = data['metadatas']
        documents = data['documents']
        
        print(f"\nMUESTRA DE DATOS ({len(ids)} ejemplos mostrados):")
        print("-" * 60)
        
        for i, doc_id in enumerate(ids):
            meta = metadatas[i] if metadatas else {}
            doc_text = documents[i] if documents else ""
            
            print(f"\n[REGISTRO {i+1}]")
            print(f"   ID: {doc_id}")
            print(f"   Metadatos: {meta}")
            
            preview_text = doc_text[:5000] + "..." if len(doc_text) > 5000 else doc_text
            preview_text = preview_text.replace("\n", " ") 
            
            print(f"   Contenido: \"{preview_text}\"")

    except Exception as e:
        print(f"[ERROR] No se pudo leer la coleccion '{nombre_col}'. Razon: {e}")


def main():
    print(f"Conectando a la base de datos en: {DB_PATH}")
    
    if not os.path.exists(DB_PATH):
        print("[ERROR CRITICO] No encuentro la carpeta de la base de datos.")
        print("   Por favor, ejecuta primero los scripts de ingesta:")
        print("   - 01_multimodal_ingest_smart.py (para imagenes)")
        print("   - 02_ingest_pdfs.py (para documentos)")
        return

    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collections = client.list_collections()
        
        nombres = [c.name for c in collections]
        print(f"Colecciones encontradas en el sistema: {nombres}")
        
        if "multimodal_knowledge" in nombres:
            inspeccionar_coleccion(client, "multimodal_knowledge")
        else:
            print("\n[AVISO] No encontre la coleccion 'multimodal_knowledge' (Imagenes).")

        if "text_knowledge" in nombres:
            inspeccionar_coleccion(client, "text_knowledge")
        else:
            print("\n[AVISO] No encontre la coleccion 'text_knowledge' (PDFs).")
            
    except Exception as e:
        print(f"[ERROR CRITICO] Fallo al conectar con ChromaDB: {e}")

if __name__ == "__main__":
    main()
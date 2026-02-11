"""
================================================================================
HERRAMIENTA DE DIAGNOSTICO: INSPECTOR DE IMAGENES
================================================================================
   Script de utilidad para consultar directamente la base de datos vectorial
   y recuperar la "visión" que tuvo la IA sobre una imagen específica.

FLUJO COMPLETO:
    1. Conexión: Accede a la base de datos ChromaDB local.
    2. Interacción: Solicita al usuario el nombre exacto de un archivo (ej: 'foto.png').
    3. Consulta (Filtering): Realiza una búsqueda exacta en los metadatos ('source').
    4. Reporte: Muestra la descripción textual generada por el modelo VLM (Vision Language Model)
       durante la fase de ingesta.

UTILIDAD:
    - Depuración: Verificar si una imagen concreta se ingestó correctamente.
    - Auditoría: Leer exactamente cómo describió la IA una imagen para entender
      por qué aparece (o no) en las búsquedas.
================================================================================
"""

import os
import sys
import chromadb
from dotenv import load_dotenv

# ==============================================================================
# CONFIGURACION
# ==============================================================================
load_dotenv()
DB_PATH = os.getenv("DB_PATH", "./chroma_db_multimodal(casa_llava_qwen)buena_spanish")
COLECCION_IMAGENES = "multimodal_knowledge"

def main():
    print("="*60)
    print("      INSPECTOR DE METADATOS DE IMAGEN (CHROMA DB)      ")
    print("="*60)

    # ====================================================================
    # PASO 1: CONEXION Y VALIDACION
    # ====================================================================
    if not DB_PATH or not os.path.exists(DB_PATH):
        print(f"[ERROR] No encuentro la base de datos en: {DB_PATH}")
        print("   Por favor, verifica tu archivo .env o la ruta configurada.")
        return

    print(f"[INFO] Conectando a: {os.path.basename(DB_PATH)}...")
    
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(COLECCION_IMAGENES)
        print(f"[INFO] Coleccion '{COLECCION_IMAGENES}' cargada correctamente.")
    except Exception as e:
        print(f"[ERROR] No existe la coleccion '{COLECCION_IMAGENES}'. Razón: {e}")
        return

    # ====================================================================
    # PASO 2: BUCLE DE CONSULTA INTERACTIVA
    # ====================================================================
    while True:
        print("\n" + "-"*60)
        print("Introduce el NOMBRE COMPLETO del archivo (ej: 'diagrama_hive.png')")
        nombre_archivo = input("Nombre del archivo (o escribe 'salir'): ").strip()

        # Condición de salida
        if nombre_archivo.lower() in ['salir', 'exit', 'quit']:
            print("[INFO] Cerrando inspector.")
            break
        
        if not nombre_archivo:
            continue

        # ====================================================================
        # PASO 3: CONSULTA POR METADATOS (Filtering)
        # ====================================================================
        try:
            resultados = collection.get(
                where={"source": nombre_archivo},
                include=["documents", "metadatas", "embeddings"]
            )
        except Exception as query_error:
            print(f"[ERROR] Fallo al consultar la base de datos: {query_error}")
            continue

        # ====================================================================
        # PASO 4: VISUALIZACION DE RESULTADOS
        # ====================================================================
        if resultados['ids']:
            descripcion = resultados['documents'][0]
            metadata = resultados['metadatas'][0]
            id_interno = resultados['ids'][0]
            
            print(f"\n[EXITO] IMAGEN ENCONTRADA")
            print(f"   Ruta original: {metadata.get('path', 'Desconocida')}")
            print(f"   ID Interno:    {id_interno}")
            print(f"   Asignatura:    {metadata.get('asignatura', 'General')}")
            print(f"   Tema:          {metadata.get('tema', 'General')}")
            
            print("-" * 30)
            print("DESCRIPCION GENERADA POR LA IA (Vision Model):")
            print("-" * 30)
            print(descripcion)
            print("-" * 30)
        else:
            print(f"\n[AVISO] No se encontro ninguna imagen con el nombre '{nombre_archivo}'.")
            print("   Nota: Asegurate de escribir la extension (.png, .jpg).")

if __name__ == "__main__":
    main()
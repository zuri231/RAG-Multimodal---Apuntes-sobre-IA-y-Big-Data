"""
================================================================================
VISUALIZACION DEL ESPACIO VECTORIAL (PROYECCION 2D)
================================================================================
   Herramienta de análisis de datos que proyecta los embeddings (vectores) de la 
   base de datos en un mapa 2D interactivo.

FLUJO COMPLETO:
    1. Conexión: Accede a ChromaDB y carga la colección de texto.
    2. Extracción: Obtiene embeddings (coordenadas matemáticas) y metadatos.
    3. Inspección: Muestra una muestra de texto por cada asignatura.
    4. Reducción (t-SNE): Algoritmo matemático que reduce 768 dimensiones a 2.
    5. Visualización: Genera un Scatter Plot interactivo con Plotly.

UTILIDAD:
    - Entender cómo agrupa la IA la información por asignaturas.
    - Detectar documentos aislados (outliers) o mal clasificados.
    - Verificar visualmente la calidad de la base de datos vectorial.
================================================================================
"""

import os
import logging
import chromadb
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from dotenv import load_dotenv

# ==============================================================================
# CONFIGURACIÓN Y LOGS
# ==============================================================================
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("visualizacion_datos")
DB_PATH = os.getenv("DB_PATH", "./chroma_db_multimodal(casa_llava_qwen)buena")
COLECCION_OBJETIVO = "text_knowledge"  # Colección a visualizar (PDFs)


def visualizar_mapa():
  
    logger.info("="*60)
    logger.info("GENERADOR DE MAPA DE CONOCIMIENTO (t-SNE)")
    logger.info("="*60)
    
    # ====================================================================
    # PASO 1: CONEXIÓN Y VALIDACIÓN
    # ====================================================================
    logger.info(f"🔌 Conectando a la base de datos en: {DB_PATH}")
    
    if not os.path.exists(DB_PATH):
        logger.error(f"Error: No se encuentra la carpeta en '{DB_PATH}'")
        return

    client = chromadb.PersistentClient(path=DB_PATH)
    
    try:
        collection = client.get_collection(name=COLECCION_OBJETIVO)
    except Exception as e:
        logger.error(f"La colección '{COLECCION_OBJETIVO}' no existe: {e}")
        logger.info("💡 Sugerencia: Cambia COLECCION_OBJETIVO a 'multimodal_knowledge' para ver imágenes.")
        return

    # ====================================================================
    # PASO 2: EXTRACCIÓN DE DATOS
    # ====================================================================
    logger.info("📥 Descargando embeddings y metadatos (esto puede tardar)...")
    datos = collection.get(include=['embeddings', 'metadatas', 'documents'])
    
    if datos['embeddings'] is None or len(datos['embeddings']) == 0:
        logger.warning("La base de datos está vacía o no devolvió embeddings.")
        return

    documents = datos['documents']
    metadatas = datos['metadatas']
    embeddings = np.array(datos['embeddings'])
    total_docs = len(embeddings)
    
    logger.info(f" Procesando {total_docs} puntos de datos...")

    # ====================================================================
    # PASO 3: INSPECCIÓN DE DATOS (Muestreo)
    # ====================================================================
    ejemplos_por_asignatura = {}

    metadatas_seguros = metadatas if metadatas else [{}] * total_docs
    asignaturas = []

    for i, meta in enumerate(metadatas_seguros):
        asig = meta.get("asignatura", "General") if meta else "General"
        asignaturas.append(asig)
        if asig not in ejemplos_por_asignatura and documents:
            ejemplos_por_asignatura[asig] = documents[i]

    print("\n" + "-"*50)
    print("MUESTRA DE CONTENIDO POR ASIGNATURA:")
    print("-"*50)
    for asig, texto in ejemplos_por_asignatura.items():
        print(f" {asig}: {texto[:100]}...")
    print("-"*50 + "\n")

    # ====================================================================
    # PASO 4: REDUCCIÓN DE DIMENSIONALIDAD (t-SNE)
    # ====================================================================
    logger.info(" Ejecutando algoritmo t-SNE (reduciendo dimensiones)...")
    perplejidad = min(30, total_docs - 1)
    if perplejidad < 1: perplejidad = 1
    
    tsne = TSNE(
        n_components=2,          
        perplexity=perplejidad,  
        random_state=42,       
        init='pca',
        learning_rate='auto'
    )
    
    vis_dims = tsne.fit_transform(embeddings)

    # ====================================================================
    # PASO 5: PREPARACIÓN Y VISUALIZACIÓN
    # ====================================================================
    df = pd.DataFrame({
        'x': vis_dims[:, 0],
        'y': vis_dims[:, 1],
        'Asignatura': asignaturas,
        'texto': documents if documents else [""] * total_docs
    })
    df["texto_corto"] = df["texto"].apply(lambda x: x[:150] + "..." if len(x) > 150 else x)

    logger.info("Generando gráfico interactivo con Plotly...")
    
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="Asignatura",
        custom_data=["Asignatura", "texto_corto"],
        title=f"Mapa de Conocimiento RAG ({total_docs} documentos)",
        labels={'x': 'Dimensión Latente 1', 'y': 'Dimensión Latente 2'},
        opacity=0.8,
        size_max=10
    )
    fig.update_traces(
        hovertemplate=
            "<b>Asignatura:</b> %{customdata[0]}<br>" +
            "<b>Contenido:</b> %{customdata[1]}<extra></extra>"
    )

    fig.update_layout(
        legend_title_text='Asignaturas',
        plot_bgcolor='white'
    )

    fig.show()
    logger.info("Gráfico generado correctamente.")

# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================
if __name__ == "__main__":
    visualizar_mapa()
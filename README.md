# 🤖 Asistente Virtual RAG Multimodal

**Asistente inteligente para la consulta de apuntes técnicos en PDF e imágenes sobre Inteligencia Artificial y Big Data.**

Este proyecto implementa un sistema **RAG (Retrieval-Augmented Generation)** multimodal capaz de responder preguntas técnicas complejas basándose en una base de conocimiento propia. Combina la capacidad de entender texto e imágenes para ofrecer respuestas precisas y fundamentadas.

---

## 🎯 Objetivo

El objetivo principal es crear un asistente capaz de responder dudas técnicas utilizando **documentos propios (PDFs) y diagramas/imágenes**, superando las limitaciones de los LLMs genéricos al inyectar contexto específico actualizado.

---

## 🏗️ Arquitectura

El sistema utiliza una arquitectura modular compuesta por:

* **Base de Datos Vectorial:** [ChromaDB](https://www.trychroma.com/) para el almacenamiento y recuperación eficiente de vectores.  
* **Embeddings:** `SentenceTransformers` para la vectorización de texto e imágenes.  
* **LLM (Generación):** Integración flexible con **Groq**, **OpenAI** o **Ollama**.  
* **Backend:** [FastAPI](https://fastapi.tiangolo.com/) para la gestión de la lógica y endpoints.  
* **Frontend:** [Streamlit](https://streamlit.io/) para una interfaz de usuario interactiva y amigable.

### ✨ Características Principales

* ✅ **Búsqueda Semántica:** Recuperación inteligente en documentos PDF.  
* ✅ **Soporte Multimodal:** Capacidad para entender y recuperar imágenes/diagramas.  
* ✅ **Reranking:** Reordenamiento de resultados para mayor precisión (Cross-Encoder).  
* ✅ **Query Rewriting:** Reformulación automática de preguntas con historial de chat.  
* ✅ **Transparencia:** Visualización de las fuentes y documentos utilizados en cada respuesta.

---

## 📂 Estructura del Proyecto

├── chroma\_db\_multimodal(casa\_llava\_qwen)buena\_spanish/  \# Base de datos vectorial persistente

├── data/

│   ├── imagenes/          \# Dataset de imágenes

│   └── pdfs/              \# Dataset de documentos PDF

├── img/                   \# Assets del proyecto (logos, avatares)

├── src/

│   ├── config.py          \# Configuración global

│   ├── app/

│   │   └── app.py         \# Frontend (Streamlit)

│   ├── api/

│   │   └── api.py         \# Backend (FastAPI)

│   ├── 01\_multimodal\_ingest\_smart.py  \# Script de ingesta de imágenes

│   ├── 02\_ingest\_pdfs.py              \# Script de ingesta de PDFs

│   ├── 03\_check\_chroma\_content.py     \# Utilidad de verificación

│   ├── 04\_resultados.py               \# Visualización de resultados

│   ├── 05\_comprobar.py                \# Tests A/B de texto

│   ├── 05\_comprobar\_imagenes.py       \# Tests A/B de imágenes

│   ├── 06\_buscar\_imagen.py            \# Buscador específico de imágenes

│   ├── 07\_eval\_retrieval.py           \# Evaluación de recuperación

│   ├── 08\_ragas.py                    \# Evaluación con RAGAS

│   └── 09\_evaluar\_metricas.py         \# Benchmark de configuraciones

├── .env                   \# Variables de entorno (API Keys)

├── requirements.txt       \# Dependencias

└── README.md              \# Documentación

## Instalación

1. Crear entorno con conda: conda create \--name rag\_multimodal python=3.10 \-y conda activate rag\_multimodal  
     
2. Instalar dependencias: pip install \-r requirements.txt

## Ejecución

1. Ejecutar la ingesta de imágenes (multimodal): python src/01\_multimodal\_ingest\_smart.py  
     
2. Ejecutar la ingesta de documentos PDF: python src/02\_ingest\_pdfs.py  
     
3. Lanzar la API: uvicorn api.api:app \--reload  
     
4. Lanzar la aplicación Streamlit: streamlit run app/app.py

## Evaluación del Sistema

Se han realizado distintas pruebas para evaluar el rendimiento del sistema RAG multimodal en términos de recuperación de información (retrieval) y calidad semántica de las respuestas.

### Comparación de Modelos de Embeddings

Se compararon distintos modelos para analizar cuál ofrece mejores resultados en español:

**Resultados:**

- Modelo `multilingual-e5`: **80.00%**  
- Modelo `qwen`: **90.00%**

Además, se evaluó el impacto del idioma de las descripciones de las imágenes (generado con CLIP):

- Modelo `qwen (no spanish)`: **85.00%**  
- Modelo `qwen (spanish)`: **95.00%**

**Conclusión:**  
El modelo **Qwen con las descripciones de las imágenes en español** obtiene el mejor rendimiento, confirmando la importancia de utilizar embeddings adaptados al idioma.

### Evaluación General de Retrieval (07\_eval\_retrieval.py)

Se ejecutó el script `07_eval_retrieval.py`, obteniendo:

- Hit Rate: 83.3%

Este resultado indica que el sistema recupera correctamente documentos relevantes en más del 80% de las consultas realizadas.

### Evaluación por Configuraciones (Chunk \+ Reranker)

Se evaluaron distintas configuraciones variando el tamaño de los fragmentos (chunk size) y el uso de reranker, midiendo Hit Rate@3, MRR@3 y latencia.

| Configuración | Hit Rate@3 | MRR@3 | Latencia (s) |
| :---- | :---- | :---- | :---- |
| db\_800 (Base) | 76.9% | 0.73 | 0.335 |
| db\_800 (+Reranker) | 84.6% | 0.77 | 5.083 |
| db\_1000 (Base) | 76.9% | 0.68 | 0.328 |
| db\_1000 (+Reranker) | 84.6% | 0.78 | 5.861 |

**Conclusiones:**

- El uso de **reranker mejora significativamente la precisión** (hasta un 84.6% de Hit Rate@3).  
- Aumenta la latencia, por lo que existe un compromiso entre calidad y velocidad.  
- La configuración `db_1000 + reranker` obtiene el mejor MRR@3 (0.78).

### Evaluación Semántica con RAGAS

Para evaluar la calidad de las respuestas generadas se utilizó la librería **RAGAS**, con las métricas:

- Faithfulness (fidelidad al contexto)  
- Answer Relevancy (relevancia de la respuesta)  
- Context Precision (precisión del contexto recuperado)

| Pregunta | Faithfulness | Answer Relevancy | Context Precision |
| :---- | :---- | :---- | :---- |
| 1\. Kafka | 1.000 | 0.895 | 0.633 |
| 2\. Componentes | 0.714 | 1.000 | 0.633 |
| 3\. Supervisado | 1.000 | 0.921 | 0.853 |
| **PROMEDIO** | **0.905** | **0.939** | **0.706** |

**Conclusiones:**

- El sistema presenta una alta **faithfulness (0.905)**, indicando que las respuestas están basadas en los documentos recuperados.  
- La **answer relevancy (0.939)** demuestra que las respuestas son adecuadas y coherentes con las preguntas.  
- La **context precision (0.706)** muestra un buen nivel de selección de fragmentos relevantes.

### Conclusión Global

Los resultados obtenidos demuestran que el sistema RAG multimodal:

- Recupera información relevante de forma eficaz.  
- Genera respuestas coherentes y fundamentadas.  
- Mejora su rendimiento al utilizar embeddings optimizados para español y reranking.  
- Presenta un equilibrio razonable entre calidad semántica y latencia.

Este proceso de evaluación valida la robustez del sistema y justifica las decisiones tomadas en el diseño del pipeline.

## Autores

Proyecto realizado por Zuriñe Colino y Aritz Monje.

# **Asistente Virtual RAG Multimodal: Especialización en IA y Big Data**

**Sistema de Recuperación Aumentada por Generación (RAG) con capacidades multimodales (Texto \+ Imagen) para la gestión del conocimiento académico.**

## **1\. Descripción del Proyecto**

Este repositorio contiene la implementación completa de un asistente virtual técnico diseñado para resolver el problema de la fragmentación de la información en el entorno universitario. El sistema permite a los estudiantes interactuar en lenguaje natural con una base de conocimiento curada, compuesta por apuntes técnicos (PDFs), diagramas de arquitectura y diapositivas de clase (Imágenes).

A diferencia de los LLMs generalistas (como ChatGPT), este sistema opera bajo un esquema de **Dominio Cerrado**: las respuestas se generan exclusivamente a partir de la documentación indexada, eliminando las alucinaciones y garantizando la trazabilidad de la información mediante citas explícitas a las fuentes.

La solución integra un pipeline avanzado de **Búsqueda Híbrida** (Semántica \+ Palabras Clave) y un sistema de **Reordenamiento (Reranking)**, optimizado específicamente para el idioma español y terminología técnica de Ingeniería de Datos.

### **1.1. Motivación y Problema a Resolver**

En asignaturas técnicas como *Big Data* o *Inteligencia Artificial*, el material de estudio suele estar disperso en múltiples formatos:

* **Texto denso:** Manuales de referencia y papers en PDF.  
* **Información visual crítica:** Diagramas de flujo (ej. arquitectura Kafka), capturas de código y esquemas conceptuales que los LLMs de texto tradicionales ignoran.

**El problema:** Los estudiantes pierden tiempo buscando referencias cruzadas y los modelos estándar fallan al interpretar preguntas que requieren contexto visual específico (ej. "¿Qué representa el bloque azul en el diagrama de arquitectura de Hadoop?").

**Nuestra solución:** Un motor RAG Multimodal que vectoriza tanto el texto como las descripciones semánticas de las imágenes, permitiendo una recuperación de información holística.

### **1.2. Objetivos Principales**

* **Centralización del Conocimiento:** Unificar fuentes heterogéneas en una única base de datos vectorial consultable (ChromaDB).  
* **Precisión Técnica (Zero-Hallucination):** Implementar *Guardrails* estrictos en el prompt del sistema para restringir las respuestas únicamente al contexto recuperado.  
* **Soporte Multimodal Real:** Utilizar modelos de visión (VLM) para generar descripciones ricas de imágenes educativas, permitiendo su recuperación mediante consultas textuales.  
* **Adaptabilidad de Interfaz:** Proveer una experiencia de usuario diferenciada mediante dos arquetipos de asistente:  
  * *Perfil Técnico (ArIA):* Respuestas concisas, código y logs.  
  * *Perfil Docente (LexIA):* Explicaciones pedagógicas y didácticas.  
* **Evaluación Científica:** Medir el rendimiento del sistema mediante métricas objetivas (Hit Rate, MRR, RAGAS) para validar la elección de modelos de embeddings.  
  ---

  ## **2\. Arquitectura Técnica**

El sistema se basa en una arquitectura de microservicios desacoplada, donde el frontend (Streamlit) se comunica con el núcleo lógico (FastAPI) mediante peticiones REST. El pipeline RAG implementado sigue un enfoque **híbrido y multimodal**.

### 

### **2.1. Diagrama del Flujo de Datos**

### ![][image1]

### **2.2. Componentes del Pipeline**

#### **A. Fase de Ingesta (Offline)**

Antes de la ejecución, los datos no estructurados se procesan y almacenan:

1. **Procesamiento de Texto (PDFs):** Se extrae el contenido textual, se limpia y se fragmenta (*chunking*) en ventanas de contexto optimizadas (1000 tokens con solapamiento).  
2. **Procesamiento de Imágenes:** Se utiliza un **Modelo de Visión-Lenguaje (VLM)** (como *LLaVA* o *Phi-3-Vision*) para generar descripciones textuales ricas de cada diagrama o diapositiva.  
3. **Vectorización Dual:**  
   * **Texto:** Se generan embeddings densos utilizando el modelo `Qwen/Qwen3-Embedding-0.6B`.  
   * **Imágenes:** Se generan embeddings visuales alineados semánticamente utilizando `clip-ViT-B-32`.  
4. **Almacenamiento:** Todo se indexa en **ChromaDB**, manteniendo metadatos críticos (asignatura, página, ruta del archivo).

   #### **B. Fase de Inferencia (Online)**

Cuando el usuario realiza una pregunta:

1. **Reescritura de Consulta (Query Rewriting):** Un LLM ligero reformula la pregunta del usuario utilizando el historial del chat para resolver correferencias (ej. transformar "¿y sus ventajas?" en "¿Cuáles son las ventajas de Kafka?").  
2. **Recuperación Híbrida (Hybrid Search):** Se ejecutan dos búsquedas en paralelo:  
   * *Búsqueda Densa (Vectorial):* Recupera conceptos semánticamente similares.  
   * *Búsqueda Dispersa (BM25):* Recupera coincidencias exactas de palabras clave.  
3. **Fusión de Resultados:** Se combinan ambas listas utilizando el algoritmo **Reciprocal Rank Fusion (RRF)** para obtener los candidatos más robustos.  
4. **Reordenamiento (Reranking):** Un modelo **Cross-Encoder** (`BAAI/bge-reranker-v2-m3`) evalúa la relevancia real de cada par pregunta-documento, descartando falsos positivos.  
5. **Generación de Respuesta:** Se construye un prompt dinámico inyectando el contexto recuperado y se envía al LLM principal (configurado con roles de "ArIA" o "LexIA") para generar la respuesta final en *streaming*.

## 

## 3\. Tecnologías y Modelos

El desarrollo del proyecto se ha realizado utilizando un stack tecnológico moderno, priorizando el rendimiento (baja latencia) y la precisión en la recuperación de información.

### 3.1. Stack Tecnológico (Core)

| Componente | Tecnología | Descripción y Uso |
| :---- | :---- | :---- |
| **Lenguaje Base** | Python 3.10+ | Lenguaje principal por su ecosistema de IA. |
| **Frontend** | Streamlit | Interfaz gráfica rápida para prototipado de aplicaciones de datos. |
| **Backend API** | FastAPI | Framework ASGI de alto rendimiento para exponer los endpoints del modelo. |
| **Vector Database** | ChromaDB | Base de datos vectorial *open-source* y persistente para almacenar embeddings. |
| **Librerías RAG** | SentenceTransformers | Orquestación de modelos de embedding y Cross-Encoders. |
| **Búsqueda Léxica** | Rank\_BM25 | Algoritmo probabilístico para recuperación por palabras clave (Sparse Retrieval). |
| **Procesamiento** | PyMuPDF / Pillow | Extracción de texto de PDFs y manipulación de imágenes. |

### 3.2. Modelos de Inteligencia Artificial

Se han seleccionado modelos específicos tras realizar benchmarks de rendimiento (ver Sección 6), optimizando el balance entre precisión semántica y coste computacional.

| Tipo de Modelo | Modelo Seleccionado | Justificación Técnica |
| :---- | :---- | :---- |
| **Embedding de Texto** | `Qwen/Qwen3-Embedding-0.6B` | Modelo SOTA (State-of-the-Art) multilingüe. Supera a modelos de OpenAI en benchmarks MTEB para español. |
| **Embedding de Imagen** | `clip-ViT-B-32` | Modelo de OpenAI que alinea texto e imagen en el mismo espacio vectorial, crucial para la búsqueda multimodal. |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | Cross-Encoder que reevalúa la relevancia semántica de los candidatos recuperados. Mejora el Hit Rate significativamente. |
| **LLM (Inferencia)** | `llama-3.3-70b-versatile` | Ejecutado vía **Groq** (LPU). Seleccionado por su velocidad de inferencia extrema (\>300 tokens/s) y capacidad de razonamiento. |
| **VLM (Ingesta)** | `llava-phi3` / `moondream` | Modelos de Visión-Lenguaje ejecutados localmente con **Ollama** para generar descripciones densas de las imágenes durante la ingesta. |

### 3.3. Decisiones de Arquitectura

1. **Enfoque "Hybrid Search" (Denso \+ Disperso):**  
     
   * Se utiliza **Búsqueda Vectorial** para captar el significado semántico (ej. entender que "aprendizaje automático" es similar a "machine learning").  
   * Se utiliza **BM25** para captar coincidencias exactas de términos técnicos o acrónimos (ej. "ACID", "CAP", "YARN") que los modelos vectoriales a veces diluyen.  
   * Ambos resultados se normalizan y combinan mediante **Reciprocal Rank Fusion (RRF)**.

   

2. **Estrategia Multimodal "Image-to-Text":**  
     
   * En lugar de realizar una búsqueda pura de imagen-a-imagen, el sistema procesa las imágenes en la fase de ingesta generando descripciones textuales detalladas. Esto permite que una consulta de texto ("diagrama de arquitectura kafka") recupere la imagen correcta basándose en su contenido semántico descrito.

   

3. **Pipeline de Dos Etapas (Retrieval \+ Reranking):**  
     
   * *Etapa 1 (Retrieval):* Recuperación rápida de 50 candidatos combinando ChromaDB y BM25.  
   * *Etapa 2 (Reranking):* Análisis profundo de esos 50 candidatos con el Cross-Encoder para seleccionar los 4 mejores contextualmente. Esto maximiza la precisión sin sacrificar la latencia.

## 4\. Estructura del Proyecto

El proyecto sigue una estructura modular rigurosa, separando claramente la lógica de ingestión de datos (ETL), el backend de inferencia, la interfaz de usuario y los módulos de validación científica.

📁 RAG\_MULTIMODAL/  
├── 📂 chroma\_db\_multimodal(...)/   \# Persistencia de Vectores (Base de Datos Vectorial)  
├── 📂 data/                        \# Dataset Origen (Input)  
│   ├── 📂 imagenes/                \# Diapositivas, diagramas y esquemas (.png, .jpg)  
│   └── 📂 pdfs/                    \# Apuntes técnicos y documentación (.pdf)  
├── 📂 img/                         \# Assets estáticos de la UI (logos, avatares)  
├── 📂 src/                         \# Código Fuente Principal  
│   ├── 📜 config.py                \# Configuración global y gestión de variables de entorno  
│   │  
│   ├── 📂 api/  
│   │   └── 📜 api.py               \# Backend FastAPI: Núcleo lógico del RAG y Endpoints  
│   ├── 📂 app/  
│   │   └── 📜 app.py               \# Frontend Streamlit: Interfaz de Chat y Gestión de Estado  
│   │  
│   │   \# \--- PIPELINE DE INGESTA (ETL) \---  
│   ├── 📜 01\_multimodal\_ingest\_smart.py  \# Procesamiento de imágenes y embeddings  
│   ├── 📜 02\_ingest\_pdfs.py              \# Procesamiento: Limpieza, Chunking etc…rización  
│   ├── 📜 03\_check\_chroma\_content.py     \# Diagnóstico para inspeccionar la DB  
│   │  
│   │   \# \--- SUITE DE EVALUACIÓN Y BENCHMARKING \---  
│   ├── 📜 04\_resultados.py         \# Visualización del espacio latente (Proyección t-SNE)  
│   ├── 📜 05\_comprobar.py          \# A/B Testing: Comparativa de modelos de texto  
│   ├── 📜 05\_comprobar\_imagenes.py \# A/B: Impacto idioma en recuperación visual  
│   ├── 📜 06\_buscar\_imagen.py      \# Depuración para búsqueda visual inversa  
│   ├── 📜 07\_eval\_retrieval.py     \# Cálculo de métricas de recuperación (Hit Rate)  
│   ├── 📜 08\_ragas.py              \# Evaluación de respuestas con RAGAS  
│   └── 📜 09\_evaluar\_metricas.py   \# (Chunk Size vs Reranking)  
│  
├── 📜 .env                         \# Credenciales y claves API (No incluido en repo)  
├── 📜 requirements.txt             \# Lista de dependencias y versiones  
└── 📜 README.md                    \# Documentación técnica del proyecto

### **4.1. Descripción de Módulos Clave**

* **`src/api/api.py` (Backend):** Es el orquestador del sistema. Recibe la consulta del usuario, ejecuta la reescritura de la pregunta, lanza la búsqueda híbrida en ChromaDB y BM25, aplica el reranking con Cross-Encoders y gestiona el streaming de la respuesta generada por el LLM.  
* **`src/app/app.py` (Frontend):** Gestiona la experiencia de usuario. Controla la sesión, el historial de chat, la renderización de imágenes recuperadas y la lógica de personalidades (ArIA/LexIA) mediante inyección de CSS dinámico.  
* **`src/01_multimodal_ingest_smart.py`:** Componente crítico de la multimodalidad. Utiliza un modelo de visión local para "ver" y describir textualmente cada imagen del dataset antes de vectorizarla. Esto permite que las imágenes sean recuperables mediante búsquedas semánticas de texto.  
* **`src/09_evaluar_metricas.py`:** Script científico utilizado para validar la arquitectura. Ejecuta pruebas automatizadas variando parámetros (tamaño de chunk, uso de reranker) para generar las métricas de rendimiento (Hit Rate, MRR, Latencia) presentadas en este documento.

## 5\. Instalación y Uso

Sigue estos pasos para desplegar el entorno de desarrollo local y ejecutar el asistente.

### 5.1. Prerrequisitos

* Anaconda o Miniconda instalado.  
* Python 3.10 o superior.  
* Git.

### 5.2. Configuración del Entorno

1. **Clonar el repositorio:**  
     
   git clone \[https://github.com/tu-usuario/rag-multimodal.git\](https://github.com/tu-usuario/rag-multimodal.git)  
     
   cd rag-multimodal  
     
2. **Crear y activar el entorno virtual:**  
     
   conda create \--name rag\_multimodal python=3.10 \-y  
     
   conda activate rag\_multimodal  
     
3. **Instalar dependencias:**  
     
   pip install \-r requirements.txt  
     
4. **Variables de Entorno (.env):** Crea un archivo `.env` en la raíz del proyecto y configura tus claves API. Es fundamental para el acceso a los modelos LLM.  
     
   \# Configuración del LLM (Groq / OpenRouter / OpenAI)  
     
   LLM\_PROVIDER="groq"  
     
   GROQ\_API\_KEY="gsk\_..."  
     
   OPENROUTER\_API\_KEY="sk-or-..."  
     
   \# Configuración de Rutas y Red  
     
   API\_HOST="127.0.0.1"  
     
   API\_PORT="8000"  
     
   DATA\_PATH\_IMAGENES="./data/imagenes"  
     
   DATA\_PATH\_PDFS="./data/pdfs"  
     
   DB\_PATH="./chroma\_db\_multimodal"

### 5.3. Ejecución del Sistema

El sistema requiere una fase inicial de preparación de datos (Ingesta) antes de poder realizar consultas.

#### Fase 1: Ingesta de Datos (Ejecutar solo al inicio o al actualizar apuntes)

1. **Procesar Imágenes (Multimodal):** Genera descripciones textuales de las imágenes ubicadas en `./data/imagenes` utilizando un VLM local.  
     
   python src/01\_multimodal\_ingest\_smart.py  
     
2. **Procesar Documentos (PDFs):** Limpia, fragmenta y vectoriza los PDFs ubicados en `./data/pdfs`.  
     
   python src/02\_ingest\_pdfs.py

#### Fase 2: Lanzamiento de la Aplicación

Para utilizar el asistente, es necesario ejecutar el Backend y el Frontend en **dos terminales separadas**:

**Terminal 1: Backend (API)** Inicia el servidor lógico que gestiona la IA y la base de datos.

python src/api/api.py

*Esperar hasta ver el mensaje: `[LISTO] Sistema preparado para consultas.`*

**Terminal 2: Frontend (UI)** Inicia la interfaz gráfica de usuario.

streamlit run src/app/app.py

Una vez iniciados ambos servicios, la aplicación se abrirá automáticamente en tu navegador predeterminado en: [http://localhost:8501](http://localhost:8501).

## 6\. Evaluación y Métricas

Para garantizar la fiabilidad del asistente en un entorno académico, se ha sometido el sistema a una batería de pruebas rigurosas, evaluando tanto la capacidad de recuperación (Retrieval) como la calidad de la generación (Generation).

### 6.1. Comparativa de Modelos de Embeddings

Se evaluó el rendimiento de distintos modelos para determinar cuál capturaba mejor la semántica en español del dominio técnico.

| Modelo de Embedding | Rendimiento (Accuracy) | Observaciones |
| :---- | :---: | :---- |
| `intfloat/multilingual-e5-large` | 80.00% | Buen rendimiento general, pero falla en terminología específica. |
| **`Qwen/Qwen3-Embedding-0.6B`** | **90.00%** | **Seleccionado.** Superior en comprensión de instrucciones y contexto técnico en español. |

**Impacto del Idioma en Multimodalidad:** Para la recuperación de imágenes, se analizó cómo afecta el idioma de la descripción generada por el VLM (Vision-Language Model).

| Configuración de Imagen | Rendimiento | Conclusión |
| :---- | :---: | :---- |
| Descripciones en Inglés (Raw) | 85.00% | El modelo de embedding pierde matices al cruzar idiomas. |
| **Descripciones en Español** | **95.00%** | La alineación lingüística entre la consulta del usuario y la descripción de la imagen es crítica. |

### 6.2. Evaluación de Arquitectura (Chunking & Reranking)

Se realizaron pruebas A/B variando el tamaño de fragmentación del texto (Chunk Size) y activando/desactivando el reordenamiento neuronal (Reranker).

**Métricas utilizadas:**

* **Hit Rate@3:** Probabilidad de que el documento correcto esté en el Top 3\.  
* **MRR@3 (Mean Reciprocal Rank):** Calidad del ordenamiento (cuanto más cerca de 1, mejor).  
* **Latencia:** Tiempo promedio de procesamiento por consulta.

| Configuración | Hit Rate@3 | MRR@3 | Latencia (s) | Análisis |
| :---- | :---: | :---: | :---: | :---- |
| `db_800` (Base) | 76.9% | 0.73 | **0.335s** | Muy rápido, pero precisión mejorable. |
| `db_800` (+Reranker) | 84.6% | 0.77 | 5.083s | Mejora notable en recuperación. |
| `db_1000` (Base) | 76.9% | 0.68 | **0.328s** | Similar al base de 800 tokens. |
| **`db_1000` (+Reranker)** | **84.6%** | **0.78** | 5.861s | **Configuración Óptima.** Máxima precisión semántica (MRR), aceptando un *trade-off* en latencia. |

**Conclusión Técnica:** La incorporación del **Cross-Encoder (Reranker)** es fundamental. Aunque introduce una latencia de \~5 segundos, eleva la precisión del sistema del 76% al **84.6%**, lo cual es crítico para evitar alucinaciones en respuestas técnicas.

### 

### 6.3. Calidad Semántica (Framework RAGAS)

Para evaluar la respuesta final generada por el LLM, se utilizó el framework [RAGAS](https://docs.ragas.io/), que utiliza un "Juez IA" (GPT-4 / Llama-3) para puntuar la calidad.

| Métrica | Puntuación (0-1) | Interpretación |
| :---- | :---: | :---- |
| **Faithfulness** | **0.905** | **Alta.** El sistema apenas alucina; las respuestas se basan casi exclusivamente en el contexto recuperado (apuntes). |
| **Answer Relevancy** | **0.939** | **Excelente.** El asistente responde exactamente a la intención de la pregunta del usuario. |
| **Context Precision** | **0.706** | **Buena.** El sistema recupera mayoritariamente información útil, aunque a veces incluye algo de "ruido" (contexto irrelevante) que el LLM debe filtrar. |

**Validación:** Los resultados demuestran que el sistema es **robusto y confiable** para su uso como tutor académico, priorizando la veracidad de la información (Faithfulness) sobre la creatividad.

## 7\. Funcionalidades del Sistema

El asistente ha sido diseñado no solo como un motor de búsqueda, sino como una herramienta de estudio interactiva con capacidades avanzadas de adaptación al usuario.

### 7.1. Personalización de la Experiencia (Dual Persona)

El sistema implementa dos arquetipos de asistente distintos, seleccionables desde la barra lateral. Esta funcionalidad altera tanto el **Prompt del Sistema (Backend)** como la **Interfaz Gráfica (Frontend)** mediante inyección dinámica de CSS.

| Característica | 👨‍💻 Modo ArIA (Técnico) | 👩‍🏫 Modo LexIA (Docente) |
| :---- | :---- | :---- |
| **Rol** | Ingeniero de Sistemas Senior. | Catedrática Universitaria. |
| **Objetivo** | Eficiencia y precisión técnica. | Pedagogía y comprensión profunda. |
| **Estilo de Respuesta** | Conciso, uso intensivo de *bullet points*, bloques de código y terminología experta. | Explicativo, uso de analogías, tono amable y estructuración didáctica. |
| **Interfaz (UI)** | Tema "Hacker/Terminal" (Fuente Monospace, Acentos Azul Neón). | Tema "Académico/Paper" (Fuente Serif, Acentos Violeta/Lavanda). |
| **Gestión de Errores** | Reportes de error técnicos (Logs). | Mensajes de ayuda y reorientación. |

### 7.2. Recuperación Multimodal (Texto \+ Imagen)

El sistema rompe la barrera del texto plano al integrar recursos visuales en las respuestas:

* **Indexación Semántica de Imágenes:** Las imágenes no se recuperan por nombre de archivo, sino por su contenido visual (interpretado por modelos VLM durante la ingesta).  
* **Renderizado Contextual:** Si la respuesta a una pregunta (ej: "Arquitectura de Spark") se entiende mejor con un diagrama, el sistema recupera la imagen correspondiente y la muestra junto a la explicación textual.  
* **Depuración Visual:** En el frontend, se incluye un expansor "Debug/Kernel" que muestra qué imágenes fueron consideradas candidatas y su puntuación de similitud.

### 7.3. Seguridad y Control de Alucinaciones

Para garantizar la idoneidad académica, se han implementado estrictos *Guardrails* en el prompt del sistema:

1. **Protocolo "Zero-Hallucination":** Si la información no existe en la base de datos vectorial (apuntes), el modelo tiene prohibido inventar una respuesta o utilizar conocimiento externo generalista.  
2. **Filtrado de Dominio:** El asistente rechaza consultas fuera del ámbito académico (ej: recetas de cocina, opiniones deportivas), manteniendo el foco en la materia de estudio.  
3. **Gestión de Rate Limits:** El sistema captura y gestiona proactivamente los errores de cuota de la API (Error 429), informando al usuario con mensajes amigables en lugar de fallos técnicos.

## 

## 8\. Autores y Licencia

Este proyecto ha sido desarrollado como parte del Trabajo de Fin de Máster / Especialización en Inteligencia Artificial y Big Data.

### 👥 Autores

4. **Zuriñe Colino** \- *Ingeniera de Datos & IA*  
5. **Aritz Monje** \- *Ingeniero de Datos & IA*

### 📄 Licencia

Este proyecto está bajo la Licencia **MIT** \- mira el archivo [LICENSE.md](http://LICENSE.md) para más detalles.

---

**Nota:** Este repositorio es de carácter académico y demostrativo. Los documentos PDF e imágenes utilizados en el dataset `data/` son propiedad de sus respectivos autores y se utilizan aquí únicamente con fines educativos bajo el concepto de *Fair Use*.

[image1]: <data:image/png;base64,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>

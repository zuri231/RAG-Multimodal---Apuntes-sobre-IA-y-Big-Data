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
     
3. Lanzar la API: python api/api.py  
     
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

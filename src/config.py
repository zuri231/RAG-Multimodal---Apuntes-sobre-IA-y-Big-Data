import os
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq  

# Cargar variables
load_dotenv()

class Config:
    # RUTAS
    DB_PATH = os.getenv("DB_PATH", "./chroma_db_multimodal")
    IMAGENES_DIR = os.getenv("DATA_PATH_IMAGENES", "./data/imagenes")
    
    # MODELOS LOCALES
    MODEL_TEXT = os.getenv("MODEL_EMBEDDING_TEXT", "Qwen/Qwen3-Embedding-0.6B")
    MODEL_IMAGE = os.getenv("MODEL_EMBEDDING_IMAGE", "clip-ViT-B-32")
    MODEL_RERANKER = os.getenv("MODEL_RERANKER", "BAAI/bge-reranker-v2-m3")
    
    # LLM PROVIDER (Switch)
    PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()
    
    # UMBRALES
    UMBRAL_RERANKER = float(os.getenv("UMBRAL_RERANKER", "0.0"))

    @staticmethod
    def get_llm_client():
        """Devuelve el cliente y el modelo configurado según el .env"""
        
        if Config.PROVIDER == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key: raise ValueError("Faltan GROQ_API_KEY en .env")
            
            print("⚡ Usando proveedor: GROQ (Velocidad extrema)")
            return {
                "client": Groq(api_key=api_key),
                "model": os.getenv("LLM_MODEL_GROQ", "llama-3.3-70b-versatile"),
                "type": "groq"
            }
            
        else: # Default: OpenRouter
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key: raise ValueError("Faltan OPENROUTER_API_KEY en .env")
            
            print("Usando proveedor: OPENROUTER")
            return {
                "client": OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key),
                "model": os.getenv("LLM_MODEL_OPENROUTER", "tngtech/deepseek-r1t2-chimera:free"),
                "type": "openai"
            }

# Instancia global para importar
settings = Config()
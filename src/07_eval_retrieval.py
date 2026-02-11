"""
================================================================================
EVALUACION DEL MOTOR DE BUSQUEDA (HIT RATE / RECALL)
================================================================================
   Script de validación que mide la precisión del sistema de recuperación (Retrieval).
   Utiliza un "Golden Set" (preguntas con respuesta conocida) para verificar si
   el documento correcto aparece entre los resultados recuperados.

METODOLOGIA:
    1. Golden Set: Lista predefinida de pares {Pregunta -> Documento Esperado}.
    2. Consulta: Se envía cada pregunta a la API (endpoint /ask).
    3. Análisis: Se extraen los metadatos de la respuesta (fuentes de texto e imágenes).
    4. Veredicto (Hit/Miss): 
       - HIT: El documento esperado aparece en la lista de fuentes.
       - MISS: El documento esperado NO fue recuperado.

METRICA:
    - Hit Rate: Porcentaje de veces que el sistema encuentra el documento correcto.
================================================================================
"""

import requests
import pandas as pd
import json
import time

# ==============================================================================
# CONFIGURACION
# ==============================================================================
API_URL = "http://127.0.0.1:8000/ask"
RETRIEVAL_TEST_SET = [
    {"q": "¿Que es Kafka?", "expected_doc": "apuntes_kafka.pdf"},
    {"q": "¿Que es el aprendizaje supervisado?", "expected_doc": "Explicacion_Modelos-Supervisado.pdf"},
    {"q": "¿Como funciona kafka?", "expected_doc": "servicio_principal_kafka.png"},
    {"q": "¿Que es la regresion logistica?", "expected_doc": "Regresión Logística 2025-2026.pdf"},
    {"q": "¿Que es mongoDB?", "expected_doc": "MongoDB.pdf"},
    {"q": "¿Cuales son los tipos de redes neuronales?", "expected_doc": "tipos_redes_neuronales.png"}
]

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def consultar_fuentes_api(pregunta):
    try:
        payload = {
            "pregunta": pregunta, 
            "history": [],     # Sin historial para evaluación aislada
            "persona": "chico" # Modo técnico por defecto
        }
        
        # Importante: stream=True porque la API devuelve datos por trozos
        with requests.post(API_URL, json=payload, stream=True, timeout=30) as response:
            if response.status_code != 200:
                print(f"[ERROR] API retornó estado {response.status_code}")
                return []

            # Iteramos línea por línea buscando el tipo "metadata"
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        
                        # Buscamos el bloque que contiene las fuentes
                        if chunk.get("type") == "metadata":
                            fuentes_texto = chunk.get("fuentes_texto", [])
                            # Extraemos solo el filename de los objetos de imagen
                            imagenes = [img.get("filename", "") for img in chunk.get("imagenes", [])]
                            
                            # Normalizamos y limpiamos nombres
                            return fuentes_texto + imagenes
                            
                    except json.JSONDecodeError:
                        continue
            
            return [] # Si termina sin metadatos

    except Exception as e:
        print(f"[ERROR] Fallo de conexión: {e}")
        return []

# ==============================================================================
# FUNCION PRINCIPAL DE EVALUACION
# ==============================================================================

def evaluar_golden_set():
    print("="*60)
    print(" INICIANDO EVALUACION DE RETRIEVAL (GOLDEN SET)")
    print(f" Total de casos de prueba: {len(RETRIEVAL_TEST_SET)}")
    print("="*60)

    results = []
    hits = 0
    start_time = time.time()

    for i, item in enumerate(RETRIEVAL_TEST_SET):
        q = item["q"]
        expected = item["expected_doc"]
        
        print(f"\n[CASO {i+1}] Pregunta: '{q}'")
        print(f"         Esperado: '{expected}'")
        
        # 1. Consultar API
        fuentes_recuperadas = consultar_fuentes_api(q)
        
        # 2. Verificar coincidencia (Hit/Miss)
        # Usamos coincidencia parcial (in) por si la ruta devuelta es absoluta o relativa
        is_hit = any(expected.lower() in s.lower() for s in fuentes_recuperadas)
        
        if is_hit:
            hits += 1
            print("         RESULTADO: [HIT] Documento encontrado.")
        else:
            print("         RESULTADO: [MISS] Documento NO encontrado.")
            # print(f"         Recuperado: {fuentes_recuperadas[:3]}...") # Debug opcional

        # 3. Registrar métricas
        results.append({
            "Pregunta": q,
            "Documento_Esperado": expected,
            "Encontrado": is_hit,
            "Total_Fuentes": len(fuentes_recuperadas),
            "Top_3_Fuentes": ", ".join(fuentes_recuperadas[:3])
        })

    # ==========================================================================
    # REPORTE FINAL
    # ==========================================================================
    total_cases = len(RETRIEVAL_TEST_SET)
    hit_rate = hits / total_cases if total_cases > 0 else 0
    duration = time.time() - start_time

    print("\n" + "="*60)
    print(" RESUMEN DE RESULTADOS")
    print("="*60)
    print(f" Aciertos Totales: {hits}/{total_cases}")
    print(f" HIT RATE:         {hit_rate*100:.1f}%")
    print(f" Tiempo Total:     {duration:.2f} segundos")
    print("="*60)

    # Guardar resultados en CSV para análisis posterior
    try:
        df = pd.DataFrame(results)
        output_file = "metricas_retrieval.csv"
        df.to_csv(output_file, index=False)
        print(f"\n[INFO] Tabla detallada guardada en '{output_file}'")
    except Exception as e:
        print(f"[ERROR] No se pudo guardar el CSV: {e}")

if __name__ == "__main__":
    evaluar_golden_set()
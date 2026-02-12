"""
Módulo para interactuar con Gemini 2.5 Flash
"""
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()


class GeminiLLM:
    """Clase para gestionar Gemini 2.5 Flash"""
    
    def __init__(self, model_name='gemini-2.5-flash'):
        """
        Inicializa el modelo Gemini
        
        Args:
            model_name: Nombre del modelo
        """
        # Configurar API key
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY no encontrada en .env")
        
        genai.configure(api_key=api_key)
        
        # Configuración del modelo
        generation_config = {
            'temperature': 0.3,
            'top_p': 0.9,
            'max_output_tokens': 2048,
        }
        
        self.model = genai.GenerativeModel(
            model_name,
            generation_config=generation_config
        )
        
        print(f"✓ Gemini {model_name} configurado correctamente")
    
    def generate_response(self, prompt: str) -> str:
        """
        Genera una respuesta dado un prompt
        
        Args:
            prompt: Prompt completo (sistema + contexto + pregunta)
            
        Returns:
            Respuesta del modelo
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error al generar respuesta: {str(e)}"
    
    def create_rag_prompt(self, contexto: str, pregunta: str) -> str:
        """
        Crea un prompt estructurado para RAG
        
        Args:
            contexto: Documentos recuperados
            pregunta: Pregunta del usuario
            
        Returns:
            Prompt formateado
        """
        prompt = f"""Eres un asistente especializado en el Club Balonmano Zamora.

CONTEXTO (información verificada y reciente):
{contexto}

PREGUNTA DEL USUARIO:
{pregunta}

INSTRUCCIONES IMPORTANTES:
1. Basa tu respuesta EXCLUSIVAMENTE en el contexto proporcionado
2. Si la información solicitada no está en el contexto, responde: "No dispongo de esa información en las noticias recientes"
3. Cita la fecha de la noticia al mencionar información específica
4. Si hay información contradictoria, menciona ambas versiones
5. Sé preciso con datos numéricos (resultados, fechas, nombres)
6. Mantén un tono profesional pero cercano

RESPUESTA:"""
        
        return prompt


def test_gemini():
    """Prueba el modelo Gemini"""
    print("=" * 60)
    print("PRUEBA DE GEMINI 2.5 FLASH")
    print("=" * 60)
    
    # Inicializar
    llm = GeminiLLM()
    
    # Contexto de ejemplo
    contexto = """
    [2024-02-12 - convocatoria] Convocatoria para el partido contra Sinfín. El entrenador del BM Zamora ha hecho pública la lista de convocados para el partido del próximo sábado 15 de febrero contra el CD Sinfín. El encuentro se disputará a las 18:00 en el pabellón Ángel Nieto.
    
    [2024-02-10 - noticia] El BM Zamora entrena pensando en el Sinfín. La plantilla del Balonmano Zamora ha completado esta semana una intensa preparación de cara al próximo compromiso liguero ante el CD Sinfín.
    """
    
    pregunta = "¿Cuándo es el próximo partido del BM Zamora?"
    
    print(f"\nPregunta: {pregunta}\n")
    print("Generando respuesta...\n")
    
    # Crear prompt y generar respuesta
    prompt = llm.create_rag_prompt(contexto, pregunta)
    respuesta = llm.generate_response(prompt)
    
    print("=" * 60)
    print("RESPUESTA:")
    print("=" * 60)
    print(respuesta)


if __name__ == "__main__":
    test_gemini()
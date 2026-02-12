"""
Pipeline RAG completo: Búsqueda + Generación
"""
from embeddings import EmbeddingModel
from vectordb import VectorDatabase
from llm import GeminiLLM


class RAGSystem:
    """Sistema RAG completo"""
    
    def __init__(self):
        """Inicializa todos los componentes"""
        print("Inicializando sistema RAG...")
        
        self.embedding_model = EmbeddingModel()
        self.vectordb = VectorDatabase()
        self.llm = GeminiLLM()
        
        print("✓ Sistema RAG listo\n")
    
    def query(self, pregunta: str, n_results=5):
        """
        Procesa una pregunta y genera respuesta
        
        Args:
            pregunta: Pregunta del usuario
            n_results: Número de documentos a recuperar
            
        Returns:
            Respuesta generada
        """
        # 1. Generar embedding de la pregunta
        print(f"📝 Pregunta: {pregunta}")
        print("\n🧠 Generando embedding de la pregunta...")
        query_embedding = self.embedding_model.encode_single(pregunta)
        
        # 2. Buscar en base de datos vectorial
        print(f"🔍 Buscando en ChromaDB (top {n_results})...")
        resultados = self.vectordb.search(
            query_embedding,
            n_results=n_results,
            where_filter={'estado': 'vigente'}
        )
        
        # 3. Preparar contexto
        print("📦 Preparando contexto...")
        contexto_parts = []
        for doc, meta in zip(resultados['documents'][0], resultados['metadatas'][0]):
            contexto_parts.append(f"[{meta['fecha']} - {meta['categoria']}] {doc}")
        
        contexto = "\n\n".join(contexto_parts)
        
        # 4. Generar respuesta con LLM
        print("🤖 Generando respuesta con Gemini 2.5 Flash...")
        prompt = self.llm.create_rag_prompt(contexto, pregunta)
        respuesta = self.llm.generate_response(prompt)
        
        return respuesta, resultados


def test_rag():
    """Prueba el sistema RAG completo"""
    print("=" * 70)
    print("SISTEMA RAG - BALONMANO ZAMORA")
    print("=" * 70)
    print()
    
    # Inicializar sistema
    rag = RAGSystem()
    
    # Preguntas de prueba
    preguntas = [
        "¿Cuándo es el próximo partido del BM Zamora?",
        "¿Cuál fue el resultado del último partido?",
        "¿En qué posición está el equipo en la clasificación?",
    ]
    
    for i, pregunta in enumerate(preguntas, 1):
        print("\n" + "=" * 70)
        print(f"PREGUNTA {i}")
        print("=" * 70)
        
        respuesta, resultados = rag.query(pregunta)
        
        print("\n" + "─" * 70)
        print("RESPUESTA:")
        print("─" * 70)
        print(respuesta)
        print()


if __name__ == "__main__":
    test_rag()
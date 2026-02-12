"""
Módulo para generar embeddings de texto usando multilingual-e5-large
"""
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    """Clase para gestionar el modelo de embeddings"""
    
    def __init__(self, model_name='intfloat/multilingual-e5-large'):
        """
        Inicializa el modelo de embeddings
        
        Args:
            model_name: Nombre del modelo en HuggingFace
        """
        print(f"Cargando modelo de embeddings: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("✓ Modelo cargado correctamente")
    
    def encode(self, texts):
        """
        Genera embeddings para uno o varios textos
        
        Args:
            texts: String o lista de strings
            
        Returns:
            numpy array con los embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        return embeddings
    
    def encode_single(self, text):
        """
        Genera embedding para un solo texto
        
        Args:
            text: String con el texto
            
        Returns:
            numpy array con el embedding
        """
        return self.encode([text])[0]


# Función auxiliar para pruebas
def test_embeddings():
    """Prueba rápida del modelo de embeddings"""
    model = EmbeddingModel()
    
    # Probar con texto de ejemplo
    texto = "El Balonmano Zamora ganó 28-24"
    embedding = model.encode_single(texto)
    
    print(f"\nTexto: {texto}")
    print(f"Dimensiones del embedding: {embedding.shape}")
    print(f"Primeros 5 valores: {embedding[:5]}")


if __name__ == "__main__":
    test_embeddings()
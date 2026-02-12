"""
Módulo para gestionar la base de datos vectorial ChromaDB
"""
import chromadb
from chromadb.config import Settings
import json
from datetime import datetime
from typing import List, Dict, Optional


class VectorDatabase:
    """Clase para gestionar ChromaDB"""
    
    def __init__(self, persist_directory="./data/chroma_db", collection_name="balonmano_zamora"):
        """
        Inicializa la base de datos vectorial
        
        Args:
            persist_directory: Directorio donde se guarda la BD
            collection_name: Nombre de la colección
        """
        print(f"Inicializando ChromaDB en: {persist_directory}")
        
        # Cliente persistente
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Obtener o crear colección
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Noticias del Balonmano Zamora"}
        )
        
        print(f"✓ Colección '{collection_name}' lista")
        print(f"  Documentos actuales: {self.collection.count()}")
    
    def add_documents(self, documents: List[Dict], embeddings: List):
        """
        Añade documentos a la base de datos
        
        Args:
            documents: Lista de diccionarios con las noticias
            embeddings: Lista de embeddings correspondientes
        """
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            # ID único
            ids.append(doc['id'])
            
            # Texto completo (título + contenido)
            text = f"{doc['titulo']}. {doc['contenido']}"
            texts.append(text)
            
            # Metadatos
            metadata = {
                'fecha': doc['fecha'],
                'categoria': doc['categoria'],
                'fuente': doc['fuente'],
                'competicion': doc['competicion'],
                'estado': 'vigente'
            }
            metadatas.append(metadata)
        
        # Añadir a ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"✓ Añadidos {len(documents)} documentos")
    
    def search(self, query_embedding, n_results=5, where_filter=None):
        """
        Busca documentos similares
        
        Args:
            query_embedding: Embedding de la pregunta
            n_results: Número de resultados
            where_filter: Filtros por metadatos (opcional)
            
        Returns:
            Diccionario con resultados
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding],
            n_results=n_results,
            where=where_filter,
            include=['documents', 'metadatas', 'distances']
        )
        
        return results
    
    def delete_all(self):
        """Elimina todos los documentos (usar con cuidado)"""
        ids = self.collection.get()['ids']
        if ids:
            self.collection.delete(ids=ids)
            print(f"✓ Eliminados {len(ids)} documentos")
    
    def get_stats(self):
        """Obtiene estadísticas de la colección"""
        count = self.collection.count()
        return {
            'total_documentos': count,
            'nombre_coleccion': self.collection.name
        }


def cargar_noticias_ejemplo():
    """Carga las noticias de ejemplo desde el JSON"""
    with open('./data/noticias_ejemplo.json', 'r', encoding='utf-8') as f:
        noticias = json.load(f)
    return noticias


def test_vectordb():
    """Prueba la base de datos vectorial"""
    from embeddings import EmbeddingModel
    
    print("=" * 60)
    print("PRUEBA DE CHROMADB")
    print("=" * 60)
    
    # 1. Cargar modelo de embeddings
    embedding_model = EmbeddingModel()
    
    # 2. Inicializar base de datos
    db = VectorDatabase()
    
    # 3. Limpiar BD (por si había datos anteriores)
    db.delete_all()
    
    # 4. Cargar noticias de ejemplo
    print("\nCargando noticias de ejemplo...")
    noticias = cargar_noticias_ejemplo()
    print(f"✓ Cargadas {len(noticias)} noticias")
    
    # 5. Generar embeddings
    print("\nGenerando embeddings...")
    textos = [f"{n['titulo']}. {n['contenido']}" for n in noticias]
    embeddings = embedding_model.encode(textos)
    print(f"✓ Generados {len(embeddings)} embeddings")
    
    # 6. Insertar en ChromaDB
    print("\nInsertando en ChromaDB...")
    db.add_documents(noticias, embeddings)
    
    # 7. Estadísticas
    print("\n" + "=" * 60)
    stats = db.get_stats()
    print(f"Total documentos: {stats['total_documentos']}")
    
    # 8. Prueba de búsqueda
    print("\n" + "=" * 60)
    print("PRUEBA DE BÚSQUEDA")
    print("=" * 60)
    
    pregunta = "¿Cuándo es el próximo partido?"
    print(f"\nPregunta: {pregunta}")
    
    query_embedding = embedding_model.encode_single(pregunta)
    resultados = db.search(query_embedding, n_results=3)
    
    print(f"\nTop 3 resultados más relevantes:")
    for i, (doc, meta, dist) in enumerate(zip(
        resultados['documents'][0],
        resultados['metadatas'][0],
        resultados['distances'][0]
    )):
        print(f"\n{i+1}. [{meta['fecha']}] {meta['categoria']}")
        print(f"   {doc[:100]}...")
        print(f"   Distancia: {dist:.4f}")


if __name__ == "__main__":
    test_vectordb()
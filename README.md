# Sistema RAG - Noticias Balonmano Zamora en Tiempo Real

Sistema de Retrieval-Augmented Generation (RAG) para responder preguntas sobre el Club Balonmano Zamora utilizando información actualizada.

## 🎯 Objetivo

Diseñar un sistema RAG que responda preguntas de usuarios sobre noticias, resultados y actualidad del Balonmano Zamora, manteniendo la información actualizada automáticamente.

## 🏗️ Arquitectura

- **Embeddings**: multilingual-e5-large (1024 dimensiones)
- **Base de datos vectorial**: ChromaDB
- **LLM**: Google Gemini 2.5 Flash
- **Scraping**: BeautifulSoup + APIs

## 📦 Instalación
```bash
# Clonar repositorio
git clone https://github.com/iragonz/Noticias-de-Balonmano-Zamora-en-tiempo-real.git
cd Noticias-de-Balonmano-Zamora-en-tiempo-real

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env y añadir tu GEMINI_API_KEY
```

## 🚀 Uso
```bash
# Probar embeddings
python src/embeddings.py

# Ejecutar sistema completo
python app.py
```

## 👨‍💻 Autor

Iván Ramos González
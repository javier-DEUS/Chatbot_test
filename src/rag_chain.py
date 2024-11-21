import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Cargar la clave de API desde las variables de entorno
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") # Asegúrate de tener un archivo .env en tu directorio de trabajo con una clave válida

RAG_PROMPT_TEMPLATE = """
You are a helpful coding assistant that can answer questions about the provided context. 
The context is usually a PDF document or an image (screenshot) of a code file. 
Augment your answers with code snippets from the context if necessary. 
If you don't know the answer, say you don't know.

Context: {context}
Question: {question}
"""

PROMPT = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


def format_docs(docs):
    """
    Formatea una lista de documentos concatenando su contenido con saltos de línea dobles.

    Args:
        docs (list): Lista de documentos, donde cada documento tiene un atributo `page_content`.

    Returns:
        str: Cadena de texto que combina el contenido de todos los documentos separados por dos líneas en blanco.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(chunks):
    """
    Crea una cadena RAG (Retrieval-Augmented Generation) utilizando FAISS para la recuperación
    de documentos y un modelo de lenguaje para la generación de respuestas.

    Args:
        chunks (list): Lista de fragmentos de documentos que serán utilizados para construir el índice de búsqueda.

    Returns:
        Pipeline: Cadena RAG que conecta la recuperación de documentos con el modelo de lenguaje para responder preguntas.
    """
    # Crear embeddings para los fragmentos usando OpenAI
    embeddings = OpenAIEmbeddings(api_key=api_key)
    
    # Crear índice FAISS a partir de los documentos
    doc_search = FAISS.from_documents(chunks, embeddings)
    
    # Configurar el recuperador con FAISS
    retriever = doc_search.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Recuperar los 5 documentos más similares
    )
    
    # Configurar el modelo de lenguaje (LLM) para responder preguntas
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    # Definir la cadena RAG
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

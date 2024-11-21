import logging 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.text_splitter import Language 
from langchain_community.document_loaders import PyPDFLoader 
from langchain_community.document_loaders.parsers.pdf import ( extract_from_images_with_rapidocr, ) 
from langchain.schema import Document



def split_documents(documents):
    """
    Divide documentos en fragmentos más pequeños para facilitar el procesamiento.

    Args:
        documents (list): Lista de documentos a dividir.

    Returns:
        list: Lista de fragmentos divididos.
    """
    # Configurar el divisor de texto
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Dividir los documentos en fragmentos más pequeños
    return text_splitter.split_documents(documents)



def process_pdf(source):
    # Cargar el PDF
    loader = PyPDFLoader(source)
    documents = loader.load()
    
    # Filtrar páginas escaneadas
    unscanned_documents = [doc for doc in documents if doc.page_content.strip() != ""]
    scanned_pages = len(documents) - len(unscanned_documents)
    
    if scanned_pages > 0:
        logging.info(f"Omitted {scanned_pages} scanned page(s) from the PDF.")
    
    # Validar que haya contenido procesable
    if not unscanned_documents:
        raise ValueError(
            "All pages in the PDF appear to be scanned. Please use a PDF with text content."
        )
    
    # Procesar y dividir los documentos filtrados
    return split_documents(unscanned_documents)


def process_image(source):
    """
    Procesa una imagen para extraer texto mediante OCR y dividirlo en documentos.

    Args:
        source (str): Ruta al archivo de imagen.

    Returns:
        list: Lista de documentos divididos con el contenido extraído.
    """
    # Extraer texto de la imagen usando OCR
    with open(source, "rb") as image_file:
        image_bytes = image_file.read()
        extracted_text = extract_from_images_with_rapidocr([image_bytes])
    
    # Crear documento con el texto extraído
    documents = [Document(page_content=extracted_text, metadata={"source": source})]
    
    # Dividir el documento en fragmentos más pequeños
    return split_documents(documents)


def process_document(source):
    """
    Determina el tipo de archivo y lo procesa según su extensión.

    Args:
        source (str): Ruta al archivo a procesar.

    Returns:
        list: Lista de documentos procesados.

    Raises:
        ValueError: Si el tipo de archivo no es compatible.
    """
    # Procesar según la extensión del archivo
    if source.lower().endswith(".pdf"):
        return process_pdf(source)
    elif source.lower().endswith((".png", ".jpg", ".jpeg")):
        return process_image(source)
    else:
        raise ValueError(f"Unsupported file type: {source}")

"""
loader.py — Extrae texto de archivos usando LangChain document loaders.

Recibe los bytes del archivo subido por el usuario, los escribe en un
archivo temporal, usa el loader correcto de LangChain, y devuelve
una lista de LangChain Documents listos para el splitter.

Por qué archivos temporales?
    Los loaders de LangChain trabajan con rutas de archivo en disco,
    no con bytes en memoria. El archivo temporal se borra automáticamente
    al salir del bloque with.
"""

import tempfile
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)


# Mapa de MIME type a clase loader de LangChain
LOADERS = {
    "application/pdf":    PyPDFLoader,
    "text/plain":         TextLoader,
    "text/markdown":      TextLoader,
    "text/csv":           CSVLoader,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": Docx2txtLoader, #para documentos de word
}

# Extensiones necesarias para que los loaders identifiquen el formato
EXTENSIONS = {
    "application/pdf":    ".pdf",
    "text/plain":         ".txt",
    "text/markdown":      ".md",
    "text/csv":           ".csv",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}


def load_documents(file_bytes: bytes, mime_type: str, filename: str) -> list[Document]:
    """
    Carga un archivo desde bytes y retorna una lista de LangChain Documents.

    Cada Document tiene:
        page_content — el texto extraido
        metadata     — {"source": filename, "page": N} (pagina solo para PDFs)

    Devuelve una lista de Documents. PDFs producen un Document por pagina, los demas formatos producen un solo Document con todo el texto.
    """
    loader_class = LOADERS.get(mime_type)
    if loader_class is None:
        raise ValueError(
            f"Tipo de archivo no soportado: '{mime_type}' para '{filename}'. "
            f"Tipos permitidos: {list(LOADERS.keys())}"
        )

    extension = EXTENSIONS[mime_type]

    # Escribimos los bytes a un archivo temporal con la extension correcta. delete=False porque algunos loaders necesitan leer el archivo mas de una vez.
    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # TextLoader necesita encoding explicito para manejar tildes y caracteres especiales
        if loader_class is TextLoader:
            loader = TextLoader(tmp_path, encoding="utf-8", autodetect_encoding=True)
        else:
            loader = loader_class(tmp_path)

        documents = loader.load()

        if not documents:
            raise ValueError(
                f"El archivo '{filename}' no contiene texto extraible. "
                "Si es un PDF escaneado, solo se soportan PDFs con texto seleccionable." #no se pueden pdf de escaneos
            )

        # Normalizamos el metadata: LangChain pone la ruta temporal,
        # nosotros queremos el nombre real del archivo del usuario.
        for doc in documents:
            doc.metadata["source"] = filename

        return documents

    finally:
        # Siempre borramos el temporal, pase lo que pase
        os.unlink(tmp_path)
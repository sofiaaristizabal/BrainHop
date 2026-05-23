"""
Chunker.py se encarga de tomar el texto plano ya extraido de los documentos y dividirlo en chunks con overlap para que no se pierda el contexto. 
"""

from app.config import config

def chunk_text( text: str, chunk_size: int |None = None, chunk_overlap: int |None=None,) -> list[str]:
    """
    Con etse metodo dividimos el texto en chunks con overlap utilizando separadores jerarquicos

    Estrategia de separación:
    1. Párrafos (\n\n) -> intentamos respetar la estructura del documento
    2. Líneas (\n)  -> si un párrafo es muy largo
    3. Oraciones (". ") -> si una línea es muy larga
    4. Palabras (" ")  -> último recurso
    5. Caracteres ("") ->  solo si todo lo anterior falla

     Lista de strings, cada uno es un chunk listo para hacerle embedding
    """

    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    if not text or not text.strip():
        return []
    
    #Es posible que ingresen parametros entonces validamos 
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size}). "
        )
    
    # Separadores en orden jerárquico: de lo más semántico a lo más granular
    separators = ["\n\n", "\n", ". ", " ", ""]
 
    chunks = _split_recursive(text, separators, chunk_size)

    if chunk_overlap > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, chunk_overlap)
 
    # Limpieza final: eliminamos chunks vacíos o demasiado cortos para ser útiles (menos de 50 caracteres probablemente no aportan contexto)
    chunks = [c.strip() for c in chunks if len(c.strip()) >= 50]
 
    return chunks

def _split_recursive(text:str, separators: list[str], chunk_size:int) -> list[str]:
    """
    Divide el texto recursivamente usando una lista de separadores 
    Si el texto cabe en un solo chunk → devuelve [texto].
    Si no intenta dividir con el primer separador. Si algún fragmento
    resultante sigue siendo demasiado grande, lo divide recursivamente
    con el siguiente separador.
    Este enfoque "respeta" la estructura del documento: primero intentamos
    no cortar párrafos, si el párrafo es muy largo intentamos no cortar
    oraciones, etc.
    """

    if len(text) <= chunk_size:
        return [text]
    
    if not separators:
        return _force_split(text, chunk_size)
    
    separator = separators[0]
    remaining_separators = separators[1:]
 
    # Dividimos por el separador actual
    if separator == "":
        return _force_split(text, chunk_size)
 
    parts = text.split(separator)
 
    chunks = []
    current_chunk = ""
 
    for part in parts:
        # Si agregar esta parte al chunk actual lo haría demasiado grande...
        candidate = current_chunk + separator + part if current_chunk else part
 
        if len(candidate) <= chunk_size:
            # ...todavía cabe, lo agregamos
            current_chunk = candidate
        else:
            # ...no cabe, guardamos el chunk actual y empezamos uno nuevo
            if current_chunk:
                chunks.append(current_chunk)
 
            # Si la parte sola es más grande que chunk_size, la dividimos
            # recursivamente con el siguiente separador
            if len(part) > chunk_size:
                sub_chunks = _split_recursive(part, remaining_separators, chunk_size)
                chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                current_chunk = part
 
    # No olvidamos el último chunk que quedó pendiente
    if current_chunk:
        chunks.append(current_chunk)
 
    return chunks

def _force_split(text: str, chunk_size: int) -> list[str]:
    """
    División forzada por número de caracteres cuando no hay separadores.
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
 
 
def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """
    Agrega el final del chunk anterior al inicio del chunk siguiente.
 
    """
    overlapped = [chunks[0]]  # el primer chunk no necesita overlap del anterior
 
    for i in range(1, len(chunks)):
        previous_chunk = chunks[i - 1]
        current_chunk = chunks[i]
 
        # Tomamos los últimos overlap caracteres del chunk anterior
        overlap_text = previous_chunk[-overlap:]
 
        if not current_chunk.startswith(overlap_text.strip()):
            overlapped.append(overlap_text + " " + current_chunk)
        else:
            overlapped.append(current_chunk)
 
    return overlapped
 
 
def get_chunk_stats(chunks: list[str]) -> dict:
    """
    Devuelve estadísticas sobre los chunks generados.
    Útil para debugging y para logging durante la ingesta.
 
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_length": 0,
            "min_length": 0,
            "max_length": 0,
            "total_chars": 0,
        }
 
    lengths = [len(c) for c in chunks]
    return {
        "total_chunks": len(chunks),
        "avg_length": round(sum(lengths) / len(lengths), 1),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "total_chars": sum(lengths),
    }
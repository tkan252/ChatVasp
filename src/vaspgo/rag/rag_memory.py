import os
from pathlib import Path
from typing import List, Set

import chromadb

from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
)

from vaspgo.rag.rag_indexer import DocumentIndexer


def create_rag_memory(
    collection_name: str,
    persistence_path: str = None,
    k: int = 3,
    score_threshold: float = 0.4,
) -> ChromaDBVectorMemory:
    """
    Create and initialize RAG vector memory.
    
    Args:
        collection_name: ChromaDB collection name
        persistence_path: Persistence path, if None uses default path
        k: Number of most similar results to return
        score_threshold: Minimum similarity score threshold
    
    Returns:
        ChromaDBVectorMemory instance
    """
    if persistence_path is None:
        persistence_path = os.path.join(
            str(Path.home()), ".chromadb_vaspgo"
        )
    
    rag_memory = ChromaDBVectorMemory(
        config=PersistentChromaDBVectorMemoryConfig(
            collection_name=collection_name,
            persistence_path=persistence_path,
            k=k,
            score_threshold=score_threshold,
        )
    )
    
    return rag_memory


def _get_chromadb_collection(rag_memory: ChromaDBVectorMemory):
    """
    Get the underlying ChromaDB collection from ChromaDBVectorMemory.
    
    Args:
        rag_memory: ChromaDBVectorMemory instance
        
    Returns:
        ChromaDB collection object or None
    """
    try:
        # Try to access collection directly
        if hasattr(rag_memory, '_collection') and rag_memory._collection:
            return rag_memory._collection
        
        # Try to access via _client
        if hasattr(rag_memory, '_client') and rag_memory._client:
            client = rag_memory._client
            collection_name = getattr(rag_memory._config, 'collection_name', None) or "VASP_WIKI"
            return client.get_or_create_collection(name=collection_name)
        
        # Fallback: create a new client connection using the same persistence path
        if hasattr(rag_memory, '_config'):
            persistence_path = getattr(rag_memory._config, 'persistence_path', None)
            collection_name = getattr(rag_memory._config, 'collection_name', None) or "VASP_WIKI"
            
            if persistence_path:
                client = chromadb.PersistentClient(path=persistence_path)
                return client.get_or_create_collection(name=collection_name)
    except Exception as e:
        print(f"Warning: Could not access ChromaDB collection: {str(e)}")
    
    return None


async def get_indexed_sources(rag_memory: ChromaDBVectorMemory) -> Set[str]:
    """
    Get set of sources that have already been indexed by querying the database metadata.
    
    Accesses ChromaDB collection directly to retrieve all documents and extract source metadata.
    
    Args:
        rag_memory: ChromaDBVectorMemory instance
        
    Returns:
        Set of source URLs/paths that are already indexed
    """
    indexed_sources = set()
    
    try:
        collection = _get_chromadb_collection(rag_memory)
        if collection:
            # Get all documents from the collection
            results = collection.get(limit=10000)  # Get up to 10000 documents
            
            # Extract unique sources from metadata
            if results and 'metadatas' in results and results['metadatas']:
                for metadata in results['metadatas']:
                    if metadata and isinstance(metadata, dict):
                        source = metadata.get('source')
                        if source:
                            indexed_sources.add(source)
    except Exception as e:
        # If query fails, return empty set (assume nothing is indexed)
        print(f"Warning: Could not query indexed sources: {str(e)}")
        return set()
    
    return indexed_sources


async def is_source_indexed(rag_memory: ChromaDBVectorMemory, source: str) -> bool:
    """
    Check if a specific source is already indexed in the database.
    
    Args:
        rag_memory: ChromaDBVectorMemory instance
        source: Source URL or path to check
        
    Returns:
        True if source is already indexed, False otherwise
    """
    try:
        collection = _get_chromadb_collection(rag_memory)
        if collection:
            # Query for documents with matching source metadata
            results = collection.get(
                where={"source": source},
                limit=1
            )
            return results is not None and len(results.get('ids', [])) > 0
    except Exception:
        pass
    
    return False


async def initialize_vasp_rag_memory(
    rag_memory: ChromaDBVectorMemory,
    clear_existing: bool = False,
    skip_indexed: bool = True,
) -> int:
    """
    Initialize VASP-related RAG memory, indexing INCAR example documents.
    Skips sources that are already indexed in the database.
    
    Args:
        rag_memory: ChromaDBVectorMemory instance
        clear_existing: Whether to clear existing memory
        skip_indexed: Whether to skip sources that are already indexed
    
    Returns:
        Total number of indexed document chunks
    """
    if clear_existing:
        await rag_memory.clear()
        print("Cleared existing RAG memory.")
    
    indexer = DocumentIndexer(memory=rag_memory, chunk_size=1500)
    
    sources = [
        "https://www.vasp.at/wiki/INCAR",
        # Global Parameters
        "https://www.vasp.at/wiki/ISTART",
        "https://www.vasp.at/wiki/ICHARG",
        "https://www.vasp.at/wiki/LCHARG",
        "https://www.vasp.at/wiki/LWAVE",
        "https://www.vasp.at/wiki/LREAL",
        "https://www.vasp.at/wiki/ADDGRID",
        "https://www.vasp.at/wiki/LVHAR",
        "https://www.vasp.at/wiki/GGA",
        "https://www.vasp.at/wiki/METAGGA",
        # Electronic Structure Calculation
        "https://www.vasp.at/wiki/PREC",
        "https://www.vasp.at/wiki/ALGO",
        "https://www.vasp.at/wiki/EDIFF",
        "https://www.vasp.at/wiki/LORBIT",
        "https://www.vasp.at/wiki/ISMEAR",
        "https://www.vasp.at/wiki/SIGMA",
        # Ionic Relaxation Calculation
        "https://www.vasp.at/wiki/EDIFFG",
        "https://www.vasp.at/wiki/IBRION",
        "https://www.vasp.at/wiki/ISIF",
        "https://www.vasp.at/wiki/ISYM",
        "https://www.vasp.at/wiki/IVDW",
        # Spin Polarization Calculation
        "https://www.vasp.at/wiki/ISPIN",
        "https://www.vasp.at/wiki/MAGMOM",
        "https://www.vasp.at/wiki/LMAXMIX",
        # DFT+U Calculation
        "https://www.vasp.at/wiki/LDAU",
        "https://www.vasp.at/wiki/LDAUTYPE",
        "https://www.vasp.at/wiki/LDAUL",
        "https://www.vasp.at/wiki/LDAUU",
        "https://www.vasp.at/wiki/LDAUJ",
        # HSE06 Calculation
        "https://www.vasp.at/wiki/LHFCALC",
        "https://www.vasp.at/wiki/HFSCREEN",
        "https://www.vasp.at/wiki/TIME",
    ]
    
    # Filter out already indexed sources
    if skip_indexed and not clear_existing:
        print("Checking for already indexed sources...")
        indexed_sources = await get_indexed_sources(rag_memory)
        new_sources = [s for s in sources if s not in indexed_sources]
        skipped_count = len(sources) - len(new_sources)
        
        if skipped_count > 0:
            print(f"Skipping {skipped_count} already indexed source(s): {', '.join(s for s in sources if s in indexed_sources)}")
        if new_sources:
            print(f"Indexing {len(new_sources)} new source(s)")
        else:
            print("All sources are already indexed. Nothing to do.")
            return 0
        
        sources = new_sources
    
    total_chunks = await indexer.index_documents(sources)
    print(f"Indexed {total_chunks} chunks from {len(sources)} VASP documents")
    
    return total_chunks


__all__ = [
    "create_rag_memory",
    "initialize_vasp_rag_memory",
    "get_indexed_sources",
    "is_source_indexed",
]

async def init():
    m = create_rag_memory("VASP_WIKI", ".")
    await initialize_vasp_rag_memory(m, clear_existing=False)

if __name__ == "__main__":
    import asyncio
    asyncio.run(init())

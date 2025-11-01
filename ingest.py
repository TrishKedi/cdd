# app/ingest.py
from .db import AsyncSessionLocal
from .repos import *
import asyncio
from utils import *
from settings import *

async def ingest_chunk():
    async with AsyncSessionLocal.begin() as s:
        repo_id = await ensure_repo(s, "webapp", "main", True)
        payload = {
          "repo_id": repo_id, "language": "js", "path": "src/api/auth.js",
          "path_prefix": "src/api/", "symbol_id": None,
          "start_line": 180, "end_line": 230, "wstart": 256, "wend": 512,
          "text_raw": "...", "text_canon": "...",
          "view_mask": 3, "sz_raw": 480, "sz_canon": 420,
          "canon_hash": "sha256hex", "is_rep": True,
          "shard_key": shard_key_for("webapp", "js"),
        }
        chunk_id = await upsert_chunk_row(s, payload)
        return chunk_id

async def ingest_embedding(chunck_id: int, faiss_vector_id: int):
    async with AsyncSessionLocal.begin() as s:        

      await ensure_pca(s, {
          "pca_version": "pca_v1",
          "input_dim": 1536,
          "output_dim": 512,
          "mean_blob": None,            # store binary or URI later if you want
          "projection_blob": None,
          "notes": "placeholder: PCA 1536→512; see artifact store"
      })

      payload = {
        "chunk_id": chunck_id,
        "view": "RAW",
        "pca_version": "pca_v1",
        "dim": 512,
        "l2_normalized": True,
        "storage_tier": "HOT",
        "faiss_vector_id": faiss_vector_id,
        "present_in_faiss": True
      }
      await upsert_embedding_row(s, payload)
      
async def register_faiss_index(chunck_id: int, faiss_vector_id: int):
    async with AsyncSessionLocal.begin() as s:        
        payload = {
          "chunk_id": chunck_id,
          "view": "RAW",
          "pca_version": "pca_v1",
          "dim": 512,
          "l2_normalized": True,
          "storage_tier": "Hot",
          "faiss_vector_id": faiss_vector_id,
          "present_in_faiss": True
        }

# async def finalize():
#   chunck_id = await ingest_chunk()
#   print(f"Chunck: {chunck_id} ingested ")
#   faiss_vector_id = 1
#   await ingest_embedding(chunck_id, faiss_vector_id)
#   print(f"Embdding ingested ")

# asyncio.run(finalize())

def fragments_to_payloads(
    fragments: list[dict],
    repo_id: int,
    repo_name: str,
    language: str,
) -> list[dict]:
    sk = shard_key_for(repo_name, language)
    payloads = []
    for frag in fragments:
        
        faiss_vector_id = frag.get('id')
        path = frag["path"]
        raw = frag["code"] or ""
        canon = frag["processedCode"] or ""
        start = frag.get('start')
        end = frag.get('end')
        payloads.append({
            # natural key pieces (use 0s if unknown)
            "repo_id": repo_id,
            "language": language,
            "path": path,
            "path_prefix": path_prefix_of(path),
            "symbol_id": None,
            "start_line": start,
            "end_line": end,
            "wstart": 0,
            "wend": 0,
            # views & sizes
            "text_raw": raw,
            "text_canon": canon,
            "view_mask": 3,                           # RAW|CANON
            "sz_raw": len(raw.split()),               # or tokenizer count later
            "sz_canon": len(canon.split()),
            "canon_hash": sha256_hex(canon),
            "is_rep": True,                           # you’re not collapsing near-dups now
            "shard_key": sk,
            "faiss_vector_id": faiss_vector_id
        })
    return payloads

async def ingest_chunks(code_blocks, repo_path):
    async with AsyncSessionLocal.begin() as s:
        
        repo_name = filename_from_path(repo_path)
        repo_id = await ensure_repo(s, repo_name, "main", True)
        payloads = fragments_to_payloads(code_blocks, repo_id, repo_name, 'js' )
        # print(f'Payloads: \n{[p.get("faiss_vector_id") for p in payloads]}')
        chunks = await upsert_batch_chunks(s, payloads)
        # print(f'chuncks: \n{chunks}')
        return chunks
        
async def ingest_embeddings(chunks, index_registry_id):
    async with AsyncSessionLocal.begin() as s:        

      await ensure_pca(s, {
          "pca_version": "pca_v1",
          "input_dim": 1536,
          "output_dim": 512,
          "mean_blob": None,            # store binary or URI later if you want
          "projection_blob": None,
          "notes": "placeholder: PCA 1536→512; see artifact store"
      })

      payloads = [{
                "chunk_id": chunk[0],
                "index_id": index_registry_id,
                "view": "RAW",
                "pca_version": "pca_v1",
                "dim": 512,
                "l2_normalized": True,
                "storage_tier": "HOT",
                "faiss_vector_id": chunk[1],
                "present_in_faiss": True

            }
            for chunk in chunks
      ]
      await upsert_batch_embeddings(s, payloads)

async def ingest_index(repo_path):

    async with AsyncSessionLocal.begin() as s:
        repo_name = filename_from_path(repo_path)
        file_name = f"{repo_name}.faiss"
        path_uri = INDEX_DIR
        shard_key = shard_key_for(filename_from_path(repo_path), 'js')
        print(f"shard_key:{shard_key}")
        repo_id = await ensure_repo(s, repo_name, "main", True)
   
        payload = {
            "repo_id": repo_id,
            "shard_key": shard_key,
            "view": "RAW",
            "algo": "IFlatIP",
            "file_name": file_name,
            "path_uri": path_uri,
            "is_active": True
        }

        index_id = await upsert_index_row(s, payload)
        return index_id

from datetime import datetime, timedelta
from cachetools import TTLCache
from typing import Dict, Any, Tuple, List

# Cache configuration
CACHE_TTL = 3600  # 1 hour in seconds
CACHE_MAXSIZE = 100  # Maximum number of index_registry_ids to cache

# Initialize cache with size limit and TTL
# - maxsize: Maximum number of entries before evicting least recently used items
# - ttl: Time in seconds before entries expire
# - Each entry is a tuple of (timestamp, embeddings_data)
_embedding_cache: TTLCache = TTLCache(
    maxsize=CACHE_MAXSIZE,
    ttl=CACHE_TTL
)

async def get_embeddings(index_registry_id: int) -> List[Dict[str, Any]]:
    """Get embeddings for a given index registry ID with caching.
    
    This function implements a memory-efficient caching strategy:
    1. LRU (Least Recently Used) cache with a fixed size limit (CACHE_MAXSIZE)
    2. TTL (Time To Live) expiration to ensure data freshness
    3. Automatic eviction of old entries when size limit is reached
    
    The cache size limit prevents unbounded memory growth by:
    - Storing maximum CACHE_MAXSIZE different index_registry_ids
    - Automatically removing least recently used entries when full
    - Expiring entries after CACHE_TTL seconds
    
    Args:
        index_registry_id: ID of the index to retrieve embeddings for
        
    Returns:
        List of embeddings with their associated chunk data
        
    Note:
        Cache entries are automatically evicted when either:
        - The entry is older than CACHE_TTL seconds
        - The cache reaches CACHE_MAXSIZE entries and a new entry needs to be cached
    """
    cache_key = f"embeddings_{index_registry_id}"
    
    # Try to get from cache - TTLCache handles expiration automatically
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    # If not in cache or expired, fetch from database
    async with AsyncSessionLocal.begin() as s:
        embeddings = await retrieve_embeddings(s, index_registry_id)
        
        # Cache the new data - TTLCache handles expiration and size limits
        _embedding_cache[cache_key] = embeddings
        return embeddings

async def get_index(repo_path):
    async with AsyncSessionLocal.begin() as s:
        shard_key = shard_key_for(filename_from_path(repo_path), 'js')
        index = await get_active_index(s, shard_key, 'RAW')
        return index


async def finalize_index_build(raw_code_blocks, repo_path, index_registry_id):
    chunks = await ingest_chunks(raw_code_blocks, repo_path)
    await ingest_embeddings(chunks, index_registry_id)
    print("=====CHUNKS AND EMBEDDINGS INGESTED=======")
    # print(chuncks)

async def delete_single_repo(repo_url):
    async with AsyncSessionLocal.begin() as s:
        repo_name = filename_from_path(repo_url)
        print(f'Deleting repo:{repo_name}')
        await delete_repo(s, repo_name)

async def get_single_chunk(faiss_vector_id):
    async with AsyncSessionLocal.begin() as s:
        chunk = await get_chunk_by(s, faiss_vector_id)
        return chunk

# asyncio.run(finalize())


import utils
from settings import *
import json
import os
import typer
from pathlib import Path
from typing import Iterable, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
from services import SimilarityLookup, CodeBlock, EmbeddingIndex, Model
from database.ingest import finalize_index_build, get_embeddings, ingest_index, get_index, delete_single_repo
import asyncio
from utils import sha256_hex, path_prefix_of, shard_key_for
from functools import reduce
from concurrent.futures import ThreadPoolExecutor

app = typer.Typer()

INDEX_PATH = utils.get_index_path()
similarity_lookup = SimilarityLookup(SIMILARITY_THRESHOLD)

console = Console()

def _short(code: str, max_lines: int = 8) -> str:
    lines = code.splitlines()
    return "\n".join(lines[:max_lines] + (["…"] if len(lines) > max_lines else []))

def _score_style(score: float) -> str:
    if score >= 0.90: return "bold green"
    if score >= 0.80: return "yellow"
    return "red"

def _link_text(path: Path, label: str, line: int = None) -> Text:
    # Makes the label clickable in modern terminals (VS Code, iTerm2, Windows Terminal)
    t = Text(label)
    uri = f"vscode://file/{path.resolve().as_posix()}"
    if line is not None:
        uri += f":{line}"
    t.stylize(f"link {uri}")
    return t

def print_matches(matches, project_root, verbose=False):

    """
    matches: iterable of objects like:
      {
        "path": "<source file>",
        "code": "<source snippet>",
        "matches": [ {"score": float, "path": "<target file>", "code": "<target snippet>", "start": int} ... ]
      }
    """
    root = (project_root or Path(".")).resolve()

    # --- Statistics calculation ---
    files_with_matches = 0
    total_matches = 0
    all_scores = []
    for entry in matches:
        rows = entry.get("matches") or []
        print(f"rows:{rows}")
        if rows:
            files_with_matches += 1
            total_matches += len(rows)
            all_scores.extend([float(m.get("score", 0)) for m in rows if "score" in m])

    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

    # --- Statistics Table ---
    stats_table = Table(show_header=False, box=None)
    stats_table.add_row("Files with matches:", str(files_with_matches))
    stats_table.add_row("Total semantic clones:", str(total_matches))
    stats_table.add_row("Average match score:", f"{avg_score:.2f}" if all_scores else "N/A")

    console.print(Panel(stats_table, title="Summary", border_style="green"))

    # --- Existing match display ---
    for entry in matches:
        src_path = Path(entry.get("path", "")).resolve()
        src_label = src_path.relative_to(root) if src_path.is_absolute() else entry.get("path", "<?>")

        rows = entry.get("matches") or []

        if rows:
            start = entry.get('start', 1)
            console.rule(Text(f"File: {src_label}:{start}", style="bold cyan"))

            if verbose:
                src_code = entry.get("code") or ""
                if src_code:
                    syn = Syntax(_short(src_code), "javascript", theme="monokai", word_wrap=True)
                    console.print(Panel(syn, title="Code", border_style="cyan"))

            table = Table(show_header=True, header_style="bold magenta", expand=True)
            table.add_column("Score", justify="right", width=6)
            table.add_column("File", overflow="fold")
            if verbose:
                table.add_column("Snippet", overflow="fold")

            for m in sorted(rows, key=lambda x: x.get("score", 0), reverse=True):
                score = float(m.get("score", 0))
                score_text = Text(f"{score:.2f}", style=_score_style(score))

                tgt_path = Path(m.get("path", "")).resolve()
                tgt_label = tgt_path.relative_to(root) if tgt_path.is_absolute() else m.get("path", "<?>")
                start_line = m.get("start")
                label_with_line = f"{tgt_label}:{start_line}" if start_line is not None else str(tgt_label)
                clickable = _link_text(tgt_path, label_with_line, start_line)

                if verbose:
                    tgt_code = m.get("code") or ""
                    syn = Syntax(_short(tgt_code), "javascript", theme="monokai", word_wrap=True)
                    table.add_row(score_text, clickable, syn)
                else:
                    table.add_row(score_text, clickable)

            console.print(Panel(table, title="Matches", border_style="magenta"))

@app.command()
def run_pipeline(
    code_locations: List[str] = typer.Argument(..., help="Path to the code directory"),
    verbose: bool = typer.Option(False, help="Show code snippets in output"),
    rebuild: bool = typer.Option(False, help="Build index for given repo(s)"),
    export: bool = typer.Option(False, help="Export matches to json file"),

):
    async def run_all():

        index_path_id_map = {}
        faiss_indexes = []

        if rebuild:
            print("rebuild index")
            faiss_files = await delete_single_repo(code_locations)
            for faiss_file in faiss_files:
                file_path = os.path.join(INDEX_DIR,faiss_file)
                if os.path.exists(file_path):
                    os.remove(file_path)  # deletes the file
                    print(f"Repo Faiss index deleted: {file_path}")
            # return
                    
        for code_directory in code_locations:            
            index_registry = await get_shard_active_index(code_directory)
            print(f"index_registry: {index_registry}")
            if index_registry:
                faiss_file = index_registry.get('file_name')
                index_registry_id = index_registry.get('index_id')
                index_path_id_map[index_registry_id] = os.path.join(INDEX_DIR, faiss_file)
                faiss_indexes.append(os.path.join(INDEX_DIR, faiss_file))
                continue

            # Process code blocks in batches, building index and storing embeddings as we go
            index_registry_id, index_path = await process_code_blocks_in_batches(code_directory)
            index_path_id_map[index_registry_id] = index_path


    asyncio.run(run_all())

async def process_code_blocks_in_batches(code_directory):
    """
    Process code blocks in batches, building index and generating embeddings for each batch.
    This function implements a streaming pipeline where each batch is fully processed
    (indexed, embedded, and stored) before moving to the next batch.
    
    Args:
        code_directory: Path to the directory containing code files
        
    Returns:
        tuple: (index_registry_id, index_path) for the processed repository
    """
    BATCH_SIZE = 50  # Adjust based on your memory constraints
    total_blocks = 0
    code_blocks = []
    code_block_service = CodeBlock()
    model = Model()
    embedding_index = None
    
    # Track index info after first batch
    index_registry_id = None
    index_path = None

    for block in code_block_service.extract_raw_code(code_directory, batch_size=BATCH_SIZE):
        code_blocks.append(block)
        total_blocks += 1
        
        # Process in batches to manage memory
        if len(code_blocks) >= BATCH_SIZE:
            # For first batch, create new index registry and FAISS index
            if index_registry_id is None:
                index_registry_id, faiss_file = await ingest_index(code_directory)
                index_path = os.path.join(INDEX_DIR, faiss_file)
                embedding_index = EmbeddingIndex(index_path)
            
            # Generate embeddings for current batch
            batch_embeddings = model.generate_embeddings(code_blocks)
            
            # Store embeddings in FAISS index
            embedding_index.add_embeddings(batch_embeddings)
            
            # Update database with batch information
            await finalize_index_build(code_blocks, code_directory, index_registry_id)
            
            print(f'====PROCESSED AND INDEXED BATCH OF {len(code_blocks)} BLOCKS====')
            code_blocks = []  # Clear the batch for next iteration

    # Process any remaining blocks
    if code_blocks:
        # Handle first batch case if total blocks were less than batch size
        if index_registry_id is None:
            index_registry_id, faiss_file = await ingest_index(code_directory)
            index_path = os.path.join(INDEX_DIR, faiss_file)
            embedding_index = EmbeddingIndex(index_path)
        
        batch_embeddings = model.generate_embeddings(code_blocks)
        embedding_index.add_embeddings(batch_embeddings)
        await finalize_index_build(code_blocks, code_directory, index_registry_id)
        print(f'====PROCESSED AND INDEXED FINAL BATCH OF {len(code_blocks)} BLOCKS====')

    print(f'====TOTAL CODE BLOCKS PROCESSED: {total_blocks}====')
    return index_registry_id, index_path

def generate_embeddings(code_blocks):
    model = Model()
    embeddings = model.generate_embeddings(code_blocks)
    print('====EMBEDDIGS GENERATED====')

    return embeddings

def store_embeddings(embeddings, index_path):
    embedding_index = EmbeddingIndex(index_path)
    embedding_index.add_embeddings(embeddings)
    print('====EMBEDDIGS ADDED TO FAISS INDEX====')

async def semantic_search(index_path, index_registry_id, index_path_id_map, visited, combined_matches, candidate_embeddings):
    query_embeddings = candidate_embeddings[index_registry_id]
    
    # print(query_embeddings)
    similarity_lookup = SimilarityLookup(index_path, SIMILARITY_THRESHOLD)
    matches = similarity_lookup.run_exhaustive_similarity_lookup(query_embeddings, index_path_id_map, visited, combined_matches, candidate_embeddings)

    return matches

# async def get_index_embeddings(index_path_id_map):
#     index_embeddings = {}
#     for index_id in index_path_id_map:
#         index_embeddings[index_id] = await get_embeddings(index_id)
#     return index_embeddings



async def exhaustive_search(index_path_id_map):
    combined_matches = []
    visited = set()
    candidate_embeddings = await get_index_embeddings(index_path_id_map)
    for index_registry_id, index_path in index_path_id_map.items():
        matches = await semantic_search(index_path, index_registry_id, index_path_id_map, visited, combined_matches, candidate_embeddings)
        combined_matches += matches
        visited.add(index_registry_id)
    
    return combined_matches

# async def run_exhaustive_similarity_lookup(index_path_id_map):
#     similarity_lookup = SimilarityLookup(SIMILARITY_THRESHOLD)
#     embeddings = await get_index_embeddings(index_path_id_map)
#     matches = []
#     query_index_stack = list(index_path_id_map.keys())
#     print(f"======Initial query_index_stack====:\n{query_index_stack}\n")

#     while len(query_index_stack):
#         candidate_index_stack = query_index_stack.copy()
#         query_index = query_index_stack.pop()
#         query_embeddings = embeddings[query_index]
#         query_index_path = index_path_id_map[query_index]
            
#         def reduce_candidates(matches, candidate_index):
#             print(f"query_index: {query_index} , candidate_index: {candidate_index}")
#             # print(f"======candidate_index====:\n\n{candidate_index}\n\n")
#             candidate_embeddings = embeddings[candidate_index]
#             candidate_index_path = index_path_id_map[candidate_index]

#             new_matches = similarity_lookup.run_similarity_lookup(
#                 query_embeddings, 
#                 candidate_embeddings, 
#                 query_index_path, 
#                 candidate_index_path
#             )

#             matches+=new_matches
#             # print(f"======MATCHES====:\n\n\n{matches}\n\n\n")
            
#             return matches

#         # print(f"======MATCHES====:\n\n\n{matches}\n\n\n")
#         matches += reduce(reduce_candidates, candidate_index_stack, matches)

#     return matches


async def yield_index_blocks(index_path_id_map):

    for index_id, index_path in index_path_id_map.items():
        blocks = await get_embeddings(index_id)

        yield index_path, blocks


async def get_index_blocks(index_path_id_map):
    index_blocks = {}
    for index_id, index_path in index_path_id_map.items():
        index_blocks[index_path] = await get_embeddings(index_id)

    return index_blocks

# async def search_all_shards(faiss_indexes, query_index_path):
#     loop = asyncio.get_running_loop()
#     max_workers = min(len(faiss_indexes), os.cpu_count() or 4)
#     print(f"number of workers:{max_workers}")
#     with ThreadPoolExecutor(max_workers=max_workers) as ex:
#         tasks = [
#             loop.run_in_executor(ex, similarity_lookup.search_shard, cand_index_path, query_index_path)
#             for cand_index_path in faiss_indexes
#         ]
#         results = await asyncio.gather(*tasks)

async def search_all_shards(faiss_indexes, query_index_path, query_blocks, blocks):
  
    loop = asyncio.get_running_loop()
    max_workers = min(len(faiss_indexes), os.cpu_count() or 4)
    print(f"number of workers:{max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        tasks = [
            loop.run_in_executor(
                ex, 
                similarity_lookup.search_shard,
                cand_index_path, 
                query_index_path
            )
            for cand_index_path in faiss_indexes
        ]
        results = await asyncio.gather(*tasks)

    def sim_matches(block_tup, result):
        print(query_blocks)
        
        block_vector_id = block_tup[0]
        m = block_tup[1]
        neighbours = result[0]
        scores = result[1]
        neighbour_similarity_map = list(zip(neighbours, scores))

        print(f"VECTOR_ID: \n\n\n{block_vector_id}\n\n\n")
        

        block = filter_blocks(query_blocks, block_vector_id)
        block_id = block[0]
        vector_id = block[1]
        path = block[2]
        code = block[3]
        start = block[4]

        matches = {
            'code': code,
            'path': path,
            'start': start,
            'matches':[ 
                {
                    # "id": code_blocks[ns_map[0]].get('id'),
                    "path": filter_blocks(cand_blocks, ns_map[0])[2],
                    # query_blocks[ns_map[0]][2], #To do ----extract this from DB via queries
                    "score": ns_map[1],
                    "code": filter_blocks(cand_blocks, ns_map[0])[3], #To do ----extract this from DB via queries
                    "start": filter_blocks(cand_blocks, ns_map[0])[4],

                }
                for ns_map in neighbour_similarity_map 
                if ns_map[1] >= similarity_threshold
                # and self.is_unique_match(unique_matches, [vector_id, ns_map[0]])
                ] 
        }
        block_vector_id+=1
        m.append(matches)

        return block_vector_id, matches
        
    

    def reduce_matches(cand_tup, cand_result):
        all_matches = []
        matches = cand_tup[1]
        cand_pstn = cand_tup[0]
        cand_blocks = blocks[faiss_indexes[cand_pstn]]
        
        all_matches+=matches
        cand_pstn+=1
        all_matches+=matches
        return cand_pstn, all_matches

    # matches = reduce(reduce_matches, results, (0, []))
    # matches = reduce(sim_matches, cand_result, (0, matches))
        
            
            



        
    
    # return {
    #     index_file : results
        
    # }
    # print(len(results))
    # print(f"results: \n\n\n{matches}\n\n\n")
    return results

# async def generate_lookup_results(index_path_id_map):
#     faiss_indexes = list(index_path_id_map.values())
#     results = {}

#     # blocks, index_path = await get_index_blocks(index_path_id_map)
#     for index_path in faiss_indexes:
#         index_file = os.path.basename(index_path)
#         shard_results = await search_all_shards(faiss_indexes, index_path)
#         faiss_indexes.remove(index_path)
#         # shard_results['query_index_path'] = index_path
#         # results[index_file] = shard_results

#         yield shard_results
        
#     # return results

async def generate_lookup_results(index_path_id_map):
    faiss_indexes = list(index_path_id_map.values())
    blocks = await get_index_blocks(index_path_id_map)
    results = []
    async for index_path, q_blocks in yield_index_blocks(index_path_id_map):
        # print(f"index_path: \n\n\n {index_path} \n\n\n")
        shard_results = await search_all_shards(faiss_indexes, index_path, q_blocks, blocks)
        faiss_indexes.remove(index_path)
        results+=shard_results
    
    # print(results)
    return results

async def generate_query_results(results):
    for index_path, query_results in results.items():
        for result in query_results:
            yield result

def filter_blocks(blocks, faiss_vector_id):
    def filter_by_fvid(query_blocks):
        return query_blocks[1] == faiss_vector_id
    
    return list(filter(filter_by_fvid, query_blocks))[0]

async def process_results(index_path_id_map):
    all_matches = {}
    all_blocks = await get_index_blocks(index_path_id_map)
    


    async for query, result in generate_lookup_results(index_path_id_map):
        search_results = []
        
        search_results = result.get('search_results')
        


        print(f"search_results:\n\n\n{result}\n\n\n")

       
        
        
        if search_results:
            query_index_path = result.get('query_index_path')
            query_blocks = all_blocks[query_index_path]

            print(f"QUERY BLOCKS: \n\n\n{query_blocks}\n\n\n")

            cand_index_path = os.path.join(INDEX_DIR, result.get('candidate_index_path'))
            cand_blocks = all_blocks[cand_index_path]

            for block_vector_id, score_nb in enumerate(search_results):
                neighbours = score_nb[0]
                scores = score_nb[1]
                neighbour_similarity_map = list(zip(neighbours, scores))

                print(f"VECTOR_ID: \n\n\n{block_vector_id}\n\n\n")
                

                # block = filter_blocks(query_blocks, block_vector_id)
                # block_id = block[0]
                # vector_id = block[1]
                # path = block[2]
                # code = block[3]
                # start = block[4]

            
                # matches = {
                # 'code': code,
                # 'path': path,
                # 'start': start,
                # 'matches':[ 
                #     {
                #         # "id": code_blocks[ns_map[0]].get('id'),
                #         "path": filter_blocks(cand_blocks, ns_map[0])[2],
                #         # query_blocks[ns_map[0]][2], #To do ----extract this from DB via queries
                #         "score": ns_map[1],
                #         "code": filter_blocks(cand_blocks, ns_map[0])[3], #To do ----extract this from DB via queries
                #         "start": filter_blocks(cand_blocks, ns_map[0])[4],

                #     }
                #     for ns_map in neighbour_similarity_map 
                #     if ns_map[1] >= similarity_lookup.similarity_threshold
                #     # and self.is_unique_match(unique_matches, [vector_id, ns_map[0]])
                #     ] 
                # }

                # print(f"MATCHES:\n\n\n{matches}\n\n\n")

async def run_exhaustive_similarity_lookup(index_path_id_map):

    matches = await generate_lookup_results(index_path_id_map)
    print(matches)

    return matches
    
    # return matches

def display_matches(matches, verbose, export):
    print_matches(matches, project_root=Path("."), verbose=verbose)
    if export:
        with open("matches.json", "w") as f:
            json.dump(matches, f, indent=2)
        print("Matches exported to matches.json")

def get_vector_test(index):
    code_blocks = utils.get_code_blocks(with_embeddings=True)

    # f_cb = code_blocks[0]
    # f_cb_id = f_cb.get('id')
    # print(f"=====First vector id: {f_cb_id}=====")
    fv = index.reconstruct(1)
    print("=====First vector=====")
    print(fv)

async def build_code_index(code_blocks, code_directory):
    index_registry_id, faiss_file = await ingest_index(code_directory)
    await finalize_index_build(code_blocks, code_directory, index_registry_id)
    index_path = os.path.join(INDEX_DIR, faiss_file)
    return index_registry_id, index_path

async def get_shard_active_index(code_directory):
    index_registry = await get_index(code_directory)

    # path_uri = index_registry.get('path_uri')
    # faiss_index_file = index_registry.get('file_name')

    # full_path = os.path.join(path_uri, faiss_index)

    # if os.path.exists(full_path):
    #     return full_path 

    return index_registry

def create_index_path_from_registry(index_registry):
    print(f"index_registry: {index_registry}")
    path_uri = index_registry.get('path_uri')
    faiss_index_file = index_registry.get('file_name')

    return os.path.join(path_uri, faiss_index_file)


    print(f"index_registry: {index_registry}")

if __name__ == "__main__":
    app(no_exception_traceback=True)

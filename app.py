import utils
import settings
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

app = typer.Typer()

INDEX_PATH = utils.get_index_path()
SIMILARITY_THRESHOLD = settings.SIMILARITY_THRESHOLD

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
        for code_directory in code_locations:

            index_path = utils.get_index_path_from_repo(code_directory)
            if rebuild:
                print("rebuild index")
                if os.path.exists(index_path):
                    os.remove(index_path)  # deletes the file
                    print(f"Repo Faiss index deleted: {index_path}")
                await delete_single_repo(code_directory)
                # return

            index_registry = await get_shard_active_index(code_directory)
            index_registry_id = None
            print(f"index_registry: {index_registry}")
            if index_registry:
                index_path = create_index_path_from_registry(index_registry)

                index_path_id_map[index_registry] = index_path
            # print(f"index_registry: {index_registry}")
        

            code_block_service = CodeBlock()
            
            print(f"========THE INDEX PATH: {index_path}========")
            embedding_index = EmbeddingIndex(index_path)
            model = Model()
            
            index = embedding_index.load_faiss_index()

            # Process code blocks in batches using lazy loading
            BATCH_SIZE = 50  # Adjust based on your memory constraints
            total_blocks = 0
            code_blocks = []

            print('====STARTING CODE BLOCK PROCESSING====')
            
            # Process code blocks as they're yielded
            for block in code_block_service.extract_raw_code(code_directory, batch_size=BATCH_SIZE):
                code_blocks.append(block)
                total_blocks += 1
                
                # Process in batches to manage memory
                if len(code_blocks) >= BATCH_SIZE:
                    code_block_service.load_code_blocks(code_blocks)
                    print(f'====PROCESSED BATCH OF {len(code_blocks)} BLOCKS====')
                    code_blocks = []  # Clear the batch

            # Process any remaining blocks
            if code_blocks:
                code_block_service.load_code_blocks(code_blocks)
                print(f'====PROCESSED FINAL BATCH OF {len(code_blocks)} BLOCKS====')

            print(f'====TOTAL CODE BLOCKS PROCESSED: {total_blocks}====')
            
            code_blocks = code_block_service.get_code_blocks()
            print('====ALL CODEBLOCKS RETRIEVED====')
            print(len(code_blocks))


            if index:
                return
                
                # get_vector_test(index)
                index_registry_id = index_registry.get('index_id')  
                await semantic_search(index_path, index_registry_id, verbose, export)

            if not index:
                # return
            
                embeddings = model.generate_embeddings(code_blocks)
                # code_block_ids = [code_block.get('id') for code_block in code_blocks]
                # zip(embeddings, code_blocks)
                print('====EMBEDDIGS RETRIEVED====')
                # print(embeddings)
                # print(f'Shape:{embeddings.shape}')
                embedding_index.add_embeddings(embeddings)
                print('====EMBEDDIGS ADDED TO FAISS INDEX====')

                # code_block_service.set_all_embeddings(embeddings)
                print('====CODE BLOCK EMBEDDINGS SET====')

                # with open(os.path.join(os.path.dirname(INDEX_PATH), 'codeBlocks-w-embeddings.json'), 'w') as out:
                #     json.dump(code_block_service.get_code_blocks(), out, indent=4)

                # get_vector_test(embedding_index.index)

            
                await execute_search(code_blocks, index_path, code_directory, verbose, export)

                # asyncio.run(semantic_search())

                # matches = similarity_lookup.find_code_duplicates(code_block_service.get_code_blocks())
            
                # print_matches(matches, project_root=Path("."), verbose=True)
    asyncio.run(run_all())


async def semantic_search(index_path, index_registry_id, index_path_id_map):
    query_embeddings = await get_embeddings(index_registry_id)
    # print(query_embeddings)
    similarity_lookup = SimilarityLookup(index_path, SIMILARITY_THRESHOLD)
    matches = similarity_lookup.find_code_duplicates(query_embeddings, index_path_id_map)

async def exhaustive_search(index_path_id_map):
    combined_matches = []
    for index_registry_id, index_path in index_path_id_map.items():
        matches = semantic_search(index_path, index_registry_id, index_path_id_map)
        combined_matches += matches
    return combined_matches

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

async def execute_search(code_blocks, index_path, code_directory, verbose, export):
    index_registry_id = await ingest_index(code_directory)
    await finalize_index_build(code_blocks, code_directory, index_registry_id)
    # await semantic_search(index_path, index_registry_id, verbose, export)

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
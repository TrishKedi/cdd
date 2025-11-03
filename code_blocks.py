from pathlib import Path
import json, subprocess, tempfile
from typing import List, Dict, Any, Optional, Set
import typer
import numpy as np
from tree_sitter import Language, Parser, Query
import tree_sitter_javascript as tsjs
import hashlib, uuid
import re
import os
import time
from multiprocessing import Pool, cpu_count
from functools import partial
from dataclasses import dataclass
from datetime import datetime

@dataclass
class FileCache:
    """Cache entry for a processed file."""
    mtime: float           # File's modification time
    size: int             # File size
    fragments: List[Dict]  # Extracted code fragments
    hash: str             # Content hash

class FileProcessingCache:
    """Manages caching of processed files to avoid reprocessing unchanged files."""
    
    def __init__(self, cache_file: str = ".code_block_cache.json"):
        self.cache_file = cache_file
        self.cache: Dict[str, FileCache] = {}
        self.load_cache()
    
    def load_cache(self) -> None:
        """Load cache from disk if it exists."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.cache = {
                        path: FileCache(**entry)
                        for path, entry in data.items()
                    }
        except Exception as e:
            typer.echo(f"⚠️ Cache loading failed: {e}")
            self.cache = {}
    
    def save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                cache_dict = {
                    path: {
                        'mtime': entry.mtime,
                        'size': entry.size,
                        'fragments': entry.fragments,
                        'hash': entry.hash
                    }
                    for path, entry in self.cache.items()
                }
                json.dump(cache_dict, f)

# Default patterns for files to exclude
DEFAULT_EXCLUDE_PATTERNS = {
    # Build and output directories
    '**/dist/**',
    '**/build/**',
    '**/out/**',
    
    # Dependencies
    '**/node_modules/**',
    '**/bower_components/**',
    '**/vendor/**',
    
    # Test files
    '**/test/**',
    '**/tests/**',
    '**/*.test.js',
    '**/*.spec.js',
    
    # Minified files
    '**/*.min.js',
    '**/*-min.js',
    '**/*bundle*.js',
    
    # Generated files
    '**/*.generated.js',
    '**/*.g.js',
    
    # Configuration files
    '**/webpack.config.js',
    '**/rollup.config.js',
    '**/jest.config.js',
    '**/babel.config.js',
    
    # Source maps
    '**/*.js.map'
}

# Patterns for detecting minified JavaScript
MINIFIED_JS_INDICATORS = [
    r'sourceMappingURL',      # Source map reference
    r'\.min\.js$',           # .min.js extension
    r'^[^\\n]{500,}$',       # Long lines (typical in minified code)
    r'\]{3,}',              # Multiple closing brackets in sequence
    r'\}{3,}'               # Multiple closing braces in sequence
]

class CodeBlock:
    """A class to handle code block parsing and analysis using tree-sitter.
    
    This class provides functionality for parsing JavaScript code blocks,
    extracting methods and identifying controllers using tree-sitter queries.
    """

    def __init__(self) -> None:
        """Initialize a new CodeBlock instance.
        
        Sets up the tree-sitter parser with JavaScript grammar and initializes
        queries for code analysis.
        """
        self.JAVASCRIPT: Language = Language(tsjs.language())
        self.code_blocks: List[Dict[str, Any]] = []
        self.parser: Parser = Parser(self.JAVASCRIPT)
        self.exclude_patterns = DEFAULT_EXCLUDE_PATTERNS
        self.minified_patterns = [re.compile(pattern) for pattern in MINIFIED_JS_INDICATORS]
        self.cache = FileProcessingCache()  # Initialize the cache handler
        self.cache_hits = 0
        self.cache_misses = 0
        self.CONTROLLER_QUERY: str = r"""
        (call_expression
        function: (member_expression
            property: (property_identifier) @callee_name
        )
        arguments: (arguments
            (_)
            (object) @controller_obj
            .
        )
        (#eq? @callee_name "extend")
        )"""
        
        METHODS_QUERY = r"""
        ; 1) Method shorthand: onInit() { ... }
        (object
        (method_definition
            name: (property_identifier) @method_name
        )
        ) @m1

        ; 2) Pair with function value: onInit: function (...) { ... }
        (object
        (pair
            key: (property_identifier) @method_name
            value: (function) @fn
        )
        ) @m2

        ; 3) Pair with arrow function value: onSomething: (...) => { ... }
        (object
        (pair
            key: (property_identifier) @method_name
            value: (arrow_function) @afn
        )
        ) @m3

        ; (optional) quoted keys: "onInit": function () {}
        (object
        (pair
            key: (string) @method_name_str
            value: (function) @fn2
        )
        ) @m4

        (object
        (pair
            key: (string) @method_name_str
            value: (arrow_function) @afn2
        )
        ) @m5
        """
        

        # self.ARROW_FUNCS = r"""
        # (object 
        # (pair 
        #     key: (property_identifier) @method_name
        #     value: (arrow_function parameters: @afn
        #     (formal_parameters (identifier)) @m3
        # """

        self.ARROW_FUNCS = r"""
        (object 
        (pair 
            key: (property_identifier) @method_name
            value: (arrow_function) @afn
        )
        )
      
        (object
        (method_definition
            name: (property_identifier) @method_name
        )
        ) @afn

        (pair
            key: (property_identifier) @method_name
            value: (function_expression) @afn
        )

        (method_definition
            name: (property_identifier) @method_name
        ) @afn
        """

        # self.ARROW_FUNCS = r"""
        # (object
        # (pair
        #     key: (property_identifier) @method_name
        #     value: (arrow_function) @afn
        # )
        # ) @m3
        # """

        self.FUNCTION_QUERY = r"""
        (function_declaration) @func
        (lexical_declaration
        (variable_declarator
            name: (identifier) @var_name
            value: [(arrow_function) (function)] @func
        )
        )
        (expression_statement
        (assignment_expression
            left: (member_expression) @member
            right: [(arrow_function) (function)] @func
        )
        )
        (method_definition) @func
        (pair
        key: (property_identifier) @prop
        value: [(arrow_function) (function)] @func
        )
        """

        

        self.OBJECT_METHODS_AND_PAIRS = r"""
        ; method shorthand: onInit() { ... } / render() { ... }
        (method_definition
        name: (property_identifier) @prop_name
        ) @mdef

        ; key: function(...) { ... }  or  key: (...) => { ... }
        (pair
        key: (property_identifier) @prop_name
        value: (choice (function) (function_expression) (arrow_function))
        ) @mpair

        ; allow quoted keys: "onInit": function(){}, "render": ()=>{}
        (pair
        key: (string) @prop_name_str
        value: (choice (function) (function_expression) (arrow_function))
        ) @mpair_str
        """

    def load_code_blocks(self, code_blocks_list: List[Dict[str, Any]]) -> None:
        """Load a list of code blocks into the service.
        
        Args:
            code_blocks_list: List of dictionaries representing code blocks
        """
        self.code_blocks = code_blocks_list

    def set_all_embeddings(self, embeddings: np.ndarray) -> None:
        """Set embeddings for all code blocks.
        
        Args:
            embeddings: Array of embeddings, where embeddings[i] corresponds
                      to self.code_blocks[i]
        """
        for index, cb in enumerate(self.code_blocks):
            cb['embeddings'] = embeddings[index].tolist()
            
    def set_matches(self, block_id: str, matches: List[Dict[str, Any]]) -> None:
        """Set similar code blocks for a specific block.
        
        Args:
            block_id: The ID of the code block to update
            matches: List of matching code blocks with similarity scores
            
        Raises:
            IndexError: If no block with the given ID is found
        """
        def filter_block(block: Dict[str, Any]) -> bool:
            return block['id'] == block_id
            
        matching_blocks = list(filter(filter_block, self.code_blocks))
        if not matching_blocks:
            raise IndexError(f"No code block found with ID {block_id}")
            
        matching_blocks[0]['matches'] = matches

    def get_code_blocks(self) -> List[Dict[str, Any]]:
        """Get all code blocks currently loaded.
        
        Returns:
            The list of all code blocks with their metadata
        """
        return self.code_blocks

    def _process_single_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single JavaScript file to extract code blocks.
        
        This is a helper method designed to run in a separate process.
        
        Args:
            file_path: Path to the JavaScript file to process
            
        Returns:
            List of extracted code blocks with their metadata
        """
        try:
            fragments = self.fragments_from_file(file_path)
            typer.echo(f"📄 Loaded {file_path} ({len(fragments)} fragments)")
            return fragments
        except Exception as e:
            typer.echo(f"⚠️ Skipping {file_path}: {e}")
            return []

    def extract_raw_code(self, code_dir: str, batch_size: int = 50) -> Any:
        """Extract code blocks from JavaScript files in a directory using parallel processing,
        lazy loading, and caching.
        
        This method combines parallel processing, lazy loading, and file caching to optimize
        both CPU and memory usage while avoiding reprocessing unchanged files. Instead of
        loading all files at once, it processes them in batches while yielding results
        incrementally.
        
        Args:
            code_dir: Path to the directory containing JavaScript files
            batch_size: Number of files to process in each batch
            
        Yields:
            Dict[str, Any]: Code blocks with their metadata, yielded as they're processed
            
        Raises:
            typer.Exit: If the directory does not exist or is not valid
        """
        directory = Path(code_dir)
        
        if not directory.exists() or not directory.is_dir():
            typer.echo(f"❌ {directory} is not a valid directory.")
            raise typer.Exit(code=1)

        # Get list of all JS files and filter them
        all_js_files = list(directory.rglob("*.js"))
        js_files = [f for f in all_js_files if self.should_process_file(f)]
        
        skipped = len(all_js_files) - len(js_files)
        if skipped > 0:
            typer.echo(f"🔍 Filtered out {skipped} files based on exclusion rules")
        
        if not js_files:
            typer.echo("No suitable JavaScript files found to process.")
            return

        # First pass: check cache and collect files needing processing
        files_to_process = []
        for file_path in js_files:
            cached_fragments = self.check_cache(file_path)
            if cached_fragments is not None:
                # Yield cached fragments immediately
                for fragment in cached_fragments:
                    yield fragment
            else:
                files_to_process.append(file_path)

        if not files_to_process:
            typer.echo("✨ All files loaded from cache!")
            return

        # Calculate optimal number of processes for remaining files
        num_processes = min(cpu_count(), batch_size, len(files_to_process))
        total_files = len(files_to_process)
        processed_files = 0
        
        typer.echo(f"Processing {total_files} uncached files using {num_processes} CPU cores")

        # Process remaining files in batches
        for i in range(0, len(files_to_process), batch_size):
            batch = files_to_process[i:i + batch_size]
            typer.echo(f"\nProcessing batch {(i//batch_size) + 1} ({len(batch)} files)")
            
            # Process batch in parallel
            with Pool(processes=num_processes) as pool:
                fragment_lists = pool.map(self._process_single_file, batch)
                
            # Update cache and yield fragments as they're processed
            for file_path, fragments in zip(batch, fragment_lists):
                if fragments:  # Only cache successful processing
                    self.update_cache(file_path, fragments)
                    for fragment in fragments:
                        yield fragment
                    
            processed_files += len(batch)
            typer.echo(f"Progress: {processed_files}/{total_files} files processed")
        
        # Print cache statistics
        total_files = self.cache_hits + self.cache_misses
        if total_files > 0:
            hit_rate = (self.cache_hits / total_files) * 100
            typer.echo(f"\n📊 Cache Statistics:")
            typer.echo(f"   Hits: {self.cache_hits}")
            typer.echo(f"   Misses: {self.cache_misses}")
            typer.echo(f"   Hit Rate: {hit_rate:.1f}%")

        # Assign IDs to all fragments after collecting them
        if all_fragments:
            self.assign_ids(all_fragments)
            typer.echo(f"🏷️  Assigned IDs to {len(all_fragments)} code fragments")
        
        return all_fragments

    def assign_ids(self, all_fragments: List[Dict[str, Any]]) -> None:
        """Assign sequential numeric IDs to code fragments.
        
        Args:
            all_fragments: List of code fragment dictionaries to assign IDs to
        """
        for frag_key, frag in enumerate(all_fragments):
            frag['id'] = frag_key
            
    def update_cache(self, file_path: Path, fragments: List[Dict[str, Any]]) -> None:
        """Update the cache with new file fragments.
        
        Args:
            file_path: Path to the processed file
            fragments: List of extracted code fragments
        """
        try:
            stat = file_path.stat()
            self.cache.cache[str(file_path)] = FileCache(
                mtime=stat.st_mtime,
                size=stat.st_size,
                fragments=fragments,
                hash=hashlib.sha256(file_path.read_bytes()).hexdigest()
            )
            self.cache.save_cache()  # Persist cache to disk
        except Exception as e:
            typer.echo(f"⚠️ Cache update failed for {file_path}: {e}")
            
    def check_cache(self, file_path: Path) -> Optional[List[Dict[str, Any]]]:
        """Check if a file is in the cache and return its fragments if valid.
        
        This method:
        1. Checks if the file exists in the cache
        2. Validates file metadata (size and mtime) hasn't changed
        3. Returns cached fragments if valid, None otherwise
        
        Args:
            file_path: Path to the file to check in cache
            
        Returns:
            List of cached fragments if valid cache exists, None otherwise
        """
        try:
            stat = file_path.stat()
            path_str = str(file_path)
            
            # Check if file is in cache and metadata matches
            if path_str in self.cache.cache:  # Use FileProcessingCache's cache dict
                cached = self.cache.cache[path_str]
                if (cached.mtime == stat.st_mtime and 
                    cached.size == stat.st_size):
                    self.cache_hits += 1
                    return cached.fragments
            
            self.cache_misses += 1
            return None
            
        except Exception as e:
            typer.echo(f"⚠️ Cache check failed for {file_path}: {e}")
            return None
        
    @staticmethod
    def babel_ast_from_code(code: str) -> Dict[str, Any]:
        """Parse JavaScript code into AST using Babel.
        
        Args:
            code: JavaScript source code as string
            
        Returns:
            Dict containing the parsed AST from Babel
            
        Raises:
            subprocess.CalledProcessError: If Babel parsing fails
        """
        with tempfile.NamedTemporaryFile("w+", suffix=".js", delete=False) as f:
            f.write(code); f.flush()
            out = subprocess.check_output(["node", "/Users/patriciaatim/Documents/personal-projects/code-duplication-detector/examples/jcdd-parser.js", f.name])
        return json.loads(out)

    def parse_js(self, code: str) -> tree_sitter.Tree:
        """Parse JavaScript code using tree-sitter.
        
        Args:
            code: JavaScript source code as string
            
        Returns:
            A tree-sitter Tree object representing the parsed code
        """
        return self.parser.parse(bytes(code, "utf-8"))

    def should_process_file(self, file_path: Path) -> bool:
        """Determine if a file should be processed based on filtering rules.
        
        This method applies several filters to determine if a file should be processed:
        1. Checks against exclude patterns (test files, minified files, etc.)
        2. Analyzes file content for minification
        3. Validates file size
        
        Args:
            file_path: Path to the JavaScript file
            
        Returns:
            bool: True if the file should be processed, False if it should be skipped
        """
        try:
            # Check against exclude patterns
            for pattern in self.exclude_patterns:
                if file_path.match(pattern):
                    typer.echo(f"📝 Skipping excluded file: {file_path}")
                    return False
            
            # Check file size (skip if too large)
            if file_path.stat().st_size > 1_000_000:  # Skip files larger than 1MB
                typer.echo(f"📝 Skipping large file: {file_path}")
                return False
            
            # Read first few KB to check for minification
            sample = file_path.read_text(encoding='utf-8', errors='ignore')[:5000]
            
            # Check for minification indicators
            for pattern in self.minified_patterns:
                if pattern.search(sample):
                    typer.echo(f"📝 Skipping minified file: {file_path}")
                    return False
            
            # Check for suspiciously long lines (typical in minified code)
            max_line_length = max(len(line) for line in sample.splitlines()[:10], default=0)
            if max_line_length > 500:
                typer.echo(f"📝 Skipping likely minified file (long lines): {file_path}")
                return False
            
            return True
            
        except Exception as e:
            typer.echo(f"⚠️ Error checking file {file_path}: {e}")
            return False

    def iter_captures(self, query: Query, root: tree_sitter.Node) -> Iterator[Tuple[tree_sitter.Node, str]]:
        """Normalize Tree-sitter captures to (node, capture_name) pairs.
        
        Handles differences in tree-sitter versions where captures may return
        either (node, name) or (node, index, pattern_index).
        
        Args:
            query: A compiled tree-sitter query
            root: Root node to search for captures
            
        Yields:
            Tuples of (node, capture_name)
        """
        for cap in query.captures(root):
            node = cap[0]
            second = cap[1]
            if isinstance(second, str):
                name = second
            else:
                name = query.capture_names[second]
            yield node, name

    def get_controller_nodes(self, tree: tree_sitter.Tree) -> List[tree_sitter.Node]:
        """Extract controller object nodes from a JavaScript AST.
        
        Looks for patterns matching controller objects in Angular-style code,
        specifically objects passed to .extend() calls.
        
        Args:
            tree: The parsed tree-sitter AST to search
            
        Returns:
            List of tree-sitter nodes representing controller objects
        """
        controller_objs = []
        controller_q = Query(self.JAVASCRIPT, self.CONTROLLER_QUERY)
        controller_q_result = controller_q.captures(tree.root_node)
        
        if controller_q_result.get('controller_obj'):
            controller_objs = controller_q_result.get('controller_obj')

        # print(f"========Controller Nodes=======: \n {controller_objs} \n")
        return controller_objs

    # def extract_function_nodes(self, tree, code_bytes: bytes):
    #     # print(f'TREE:\n{tree}\n')
    #     FUNCTION_TYPES = {
    #         "function_declaration",
    #         # "function",
    #         "arrow_function",
    #         "method_definition",
    #     }
    #     cursor = tree.walk()
    #     # print(f'CURSOR:\n{cursor}\n')

    #     test_node = cursor.node
    #     # children_funcs = test_node.children_by_field_name('name')
    #     # print(f'NODE:\n{test_node}\n')
    #     # print(children_funcs)
    #     walk_node = test_node.walk()
    #     # print(walk_node)
    #     # print(walk_node.goto_descendant(1))
    #     # print(f'NEXT NODE:\n{walk_node.node}\n')
    #     descendant_count = test_node.descendant_count
    #     # descendant = cursor.goto_descendant(0)
    #     # print(f"\n{descendant}\n")
    #     frags = []
    #     for i in range(1, descendant_count):
            
    #         descendant = walk_node.goto_descendant(i)
    #         current_node = walk_node.node
    #         # if current_node.type in FUNCTION_TYPES:

    #         print(f"NODE TYPE:\n{current_node.type}\n")
    #         print(f"CURRENT NODE:\n{current_node}\n")
            
    #         start, end = current_node.start_byte, current_node.end_byte
    #         fragment_code = code_bytes[start:end].decode("utf-8", errors="ignore")
    #         children_funcs = current_node.children_by_field_name('name')

    #         code_up_to_fragment = code_bytes[:start].decode("utf-8", errors="ignore")
    #         start_line = code_up_to_fragment.count('\n') + 1
    #         code_up_to_end = code_bytes[:end].decode("utf-8", errors="ignore")
    #         end_line = code_up_to_end.count('\n') + 1

    #         print(f"=====fragment_codee===:{fragment_code}")
    #         # print(f"=====end_line===:{end_line}")


    #         # frags.append(fragment)
    #         frags.append({
    #             "code": fragment_code,
    #             "start": start_line,
    #             "end": end_line
    #         })

    #             # print(f'====FRAGMENT:====\n{fragment}\n')
    #         # frags.append(fragment)

    #     # print(f'FRAGS:\n{frags}\n')
    #     return frags

    def extract_function_nodes(self, root_node: tree_sitter.Node, code_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract function-like nodes from a JavaScript AST.
        
        This method identifies and extracts various types of function definitions
        including function declarations, expressions, arrow functions, and
        method definitions.
        
        Args:
            root_node: The root node of the AST to search
            code_bytes: The raw bytes of the source code
            
        Returns:
            List of dictionaries containing extracted function information with
            keys: 'code', 'start', 'end', and optional metadata
        """
        query = Query(self.JAVASCRIPT, self.ARROW_FUNCS)
        
        # Valid function-like node types
        FUNCTION_TYPES = {
            "function_declaration",
            "function_expression",
            "arrow_function",
            "method_definition",
        }
        
        fragments = []
        func_captures = query.captures(root_node)
        if func_captures.get('afn'):
            nodes = func_captures.get('afn')
            # print(f"=====NODE COUNT====: {len(nodes)}")
            for i, node in enumerate(nodes):

                if node.type in FUNCTION_TYPES:
                    # print(f"=====NODE====: {node} ({capture_name})")
                    # print(f"=====NODE TYPE====: {node}")
                    # print(f"=====NODE TYPE====: {node.type}")

                    # children = node.children
                    # for child in children:

                    #     print(f"=====CHILD TYPE====: {child.type}")
                    #     child_code = code_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")

                    #     print(f"=====CHILD CODE====: {child_code}")

                    #     print(f"=====CHILD NODE====: {child}")
            
                    # if node.type in FUNCTION_TYPES:
                    start, end = node.start_byte, node.end_byte
                    fragment_code = code_bytes[start:end].decode("utf-8", errors="ignore")
                    code_up_to_fragment = code_bytes[:start].decode("utf-8", errors="ignore")
                    start_line = code_up_to_fragment.count('\n') + 1
                    code_up_to_end = code_bytes[:end].decode("utf-8", errors="ignore")
                    end_line = code_up_to_end.count('\n') + 1

                    # print(f"=====CODE#{i}====: {fragment_code}")

                    frags.append({
                        "code": fragment_code,
                        "start": start_line,
                        "end": end_line,
                    })

        # print(f'FRAGS:\n{frags}\n')
        return frags


    def normalize_js(self, snippet: str) -> str:
        """Normalize JavaScript code by cleaning up whitespace and comments.
        
        This is a lighter normalization than canonicalize_js, preserving most
        identifiers but cleaning up formatting differences.
        
        Args:
            snippet: JavaScript code snippet to normalize
            
        Returns:
            Normalized version of the code with consistent whitespace
        """
        # Compile regex patterns
        ident_pattern = re.compile(r"\b[A-Za-z_]\w*\b")
        number_pattern = re.compile(r"\b\d+(\.\d+)?\b")
        string_pattern = re.compile(r'("[^"]*"|\'[^\']*\'|`[^`]*`)', re.S)
        space_pattern = re.compile(r"\s+")
        
        # Remove comments
        snippet = re.sub(r"//.*$", "", snippet, flags=re.M)  # Line comments
        snippet = re.sub(r"/\*.*?\*/", "", snippet, flags=re.S)  # Block comments
        
        # Clean up whitespace
        snippet = snippet.replace(";", "")  # Remove semicolons
        snippet = space_pattern.sub(" ", snippet).strip()  # Normalize spaces
        
        return snippet

        return snippet

    def canonicalize_js(self, snippet: str) -> str:
        """Canonicalize JavaScript code by normalizing identifiers and literals.
        
        This performs a more aggressive normalization than normalize_js:
        1. Removes comments and normalizes whitespace
        2. Replaces string literals with <STR>
        3. Replaces number literals with <NUM>
        4. Replaces identifiers with canonical names (except keywords)
        
        Args:
            snippet: JavaScript code snippet to canonicalize
            
        Returns:
            Canonical form of the code suitable for similarity comparison
        """
        # Compile regex patterns
        ident_pattern = re.compile(r"\b[A-Za-z_]\w*\b")
        number_pattern = re.compile(r"\b\d+(\.\d+)?\b")
        string_pattern = re.compile(r'("[^"]*"|\'[^\']*\'|`[^`]*`)', re.S)
        space_pattern = re.compile(r"\s+")
        
        # Remove comments
        snippet = re.sub(r"//.*$", "", snippet, flags=re.M)  # Line comments
        snippet = re.sub(r"/\*.*?\*/", "", snippet, flags=re.S)  # Block comments
        
        # Normalize literals
        snippet = string_pattern.sub("<STR>", snippet)
        snippet = number_pattern.sub("<NUM>", snippet)
        
        # Define JavaScript keywords to preserve
        keywords = set("""
        break case catch class const continue debugger default delete do else export extends
        finally for function if import in instanceof let new return super switch this throw try
        typeof var void while with yield await enum implements interface package private protected public
        """.split())
        
        # Track identifier replacements
        next_id = 1
        mapping: Dict[str, str] = {}
        
        def replace_identifier(match: re.Match) -> str:
            """Replace identifiers while preserving keywords."""
            nonlocal next_id
            name = match.group(0)
            
            if name in keywords:
                return name
                
            if name not in mapping:
                mapping[name] = f"id_{next_id}"
                next_id += 1
            return mapping[name]
        
        # Apply identifier normalization
        snippet = ident_pattern.sub(replace_identifier, snippet)
        
        # Clean up whitespace
        snippet = snippet.replace(";", "")
        snippet = space_pattern.sub(" ", snippet).strip()
        
        return snippet

        return snippet

    def fragments_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract and process code fragments from a JavaScript file.
        
        This method:
        1. Parses the JavaScript file
        2. Detects if it contains an Angular-style controller
        3. Extracts function-like nodes
        4. Normalizes the extracted code
        
        Args:
            file_path: Path to the JavaScript file to process
            
        Returns:
            List of dictionaries containing code blocks with metadata:
            - code: Original code fragment
            - processedCode: Normalized version of the code
            - path: Source file path
            - start: Starting line number
            - end: Ending line number
            
        Raises:
            UnicodeDecodeError: If file cannot be read as UTF-8
            tree_sitter.LanguageError: If parsing fails
        """
        # Read and parse the file
        code = file_path.read_text(encoding="utf-8")
        tree = self.parse_js(code)
        root_node = tree.root_node
        
        # Check for Angular controller pattern
        ctr_node = self.get_controller_nodes(tree)
        if ctr_node:
            print("Controller Node Detected")
            root_node = ctr_node[0]
        else:
            print("No Controller Node Detected")
        
        # Extract function nodes
        fragments = self.extract_function_nodes(
            root_node, 
            code.encode("utf-8")
        )
        
        # Process and normalize fragments
        processed_fragments = [
            {
                **fragment,
                "processedCode": self.normalize_js(fragment.get('code')),
                'path': str(file_path)
            }
            for fragment in fragments
        ]
        
        return processed_fragments

        
    


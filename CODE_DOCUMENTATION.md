# Neural Text Corrector - Code Documentation

This document provides detailed code-level documentation for the Neural Text Corrector system, including all classes, methods, and functions.

## Table of Contents
1. [Core Module: spell_checker.py](#core-module-spell_checkerpy)
2. [Core Module: candidate_generator.py](#core-module-candidate_generatorpy)
3. [Web Application: app.py](#web-application-apppy)
4. [Utilities and Scripts](#utilities-and-scripts)
5. [Data Flow Examples](#data-flow-examples)

---

## Core Module: spell_checker.py

### Overview
The `spell_checker.py` module contains the main spell checking engine that combines GPT-2 language modeling with beam search for context-aware text correction.

### Functions

#### `initialize_language_model(model_identifier: str) -> GPT2LMHeadModel`
```python
def initialize_language_model(model_identifier: str) -> GPT2LMHeadModel:
    """
    Initialize and load a GPT-2 language model.
    
    Args:
        model_identifier (str): HuggingFace model identifier (e.g., "gpt2", "gpt2-medium")
    
    Returns:
        GPT2LMHeadModel: Loaded GPT-2 model in evaluation mode
    
    Example:
        model = initialize_language_model("gpt2")
    """
```

#### `initialize_text_tokenizer(tokenizer_identifier: str) -> GPT2Tokenizer`
```python
def initialize_text_tokenizer(tokenizer_identifier) -> GPT2Tokenizer:
    """
    Initialize the GPT-2 tokenizer.
    
    Args:
        tokenizer_identifier (str): HuggingFace tokenizer identifier
    
    Returns:
        GPT2Tokenizer: Initialized tokenizer
    """
```

#### `split_into_tokens(input_text: str) -> List[str]`
```python
def split_into_tokens(input_text):
    """
    Split text into tokens including words and punctuation.
    
    Args:
        input_text (str): Input text to tokenize
    
    Returns:
        List[str]: List of tokens (words and punctuation)
    
    Example:
        tokens = split_into_tokens("Hello, world!")
        # Returns: ["Hello", ",", " ", "world", "!"]
    """
```

### Class: NeuralTextCorrector

#### Constructor
```python
def __init__(self,
             model_name: str,
             tokenizer_name: str,
             vocabulary_size: int,
             max_edit_distance: int,
             min_length_by_distance: Dict[int, int],
             search_beam_size: int,
             error_penalties: Dict[int, float],
             fix_valid_words: bool,
             valid_word_cost: float,
             initial_char_cost: float,
             fix_whitespace: bool,
             max_split_distance: int,
             filter_candidates: bool,
             filter_paths: bool,
             filter_threshold: float):
    """
    Initialize the Neural Text Corrector.
    
    Args:
        model_name: GPT-2 model variant to use
        tokenizer_name: Tokenizer identifier
        vocabulary_size: Size of vocabulary for candidate generation
        max_edit_distance: Maximum edit distance for correction candidates
        min_length_by_distance: Minimum word length required for each edit distance
        search_beam_size: Number of paths to maintain in beam search
        error_penalties: Cost penalties for each edit distance level
        fix_valid_words: Whether to attempt corrections on dictionary words
        valid_word_cost: Additional cost for changing valid words
        initial_char_cost: Cost penalty for changing the first character
        fix_whitespace: Enable word split/merge corrections
        max_split_distance: Maximum edit distance for split operations
        filter_candidates: Enable candidate filtering for efficiency
        filter_paths: Enable path filtering during beam search
        filter_threshold: Cost threshold for filtering operations
    """
```

#### Method: `_configure_processing_device()`
```python
def _configure_processing_device(self):
    """
    Configure the processing device (CPU/GPU).
    
    Automatically detects CUDA availability and moves the model to GPU if available.
    Falls back to CPU if CUDA is not available or CUDA_VISIBLE_DEVICES is set to empty.
    
    Side Effects:
        - Sets self.processing_device to "cuda" or "cpu"
        - Moves self.language_model to the selected device
    """
```

#### Method: `_create_initial_path()`
```python
def _create_initial_path(self):
    """
    Create the initial path state for beam search.
    
    Returns:
        dict: Initial path state containing:
            - total_cost: 0 (initial cost)
            - probability_distribution: Initial token probabilities from GPT-2
            - cached_states: GPT-2 hidden states for efficiency
            - generated_text: Empty string
            - pending_steps: 0
    """
```

#### Method: `_refresh_path_states(path_collection)`
```python
def _refresh_path_states(self, path_collection):
    """
    Update probability distributions for paths that need refreshing.
    
    Args:
        path_collection (List[dict]): Collection of path states
    
    Returns:
        List[dict]: Updated path collection with refreshed probabilities
    
    Note:
        This method reuses cached GPT-2 states for efficiency, only computing
        new probabilities for the latest tokens.
    """
```

#### Method: `_convert_candidates_to_tokens(text_candidates)`
```python
def _convert_candidates_to_tokens(self, text_candidates):
    """
    Convert text candidates to token IDs.
    
    Args:
        text_candidates (List[str]): List of candidate text strings
    
    Returns:
        List[List[int]]: Token ID sequences for each candidate
    """
```

#### Method: `_generate_correction_options(current_token, following_token, has_preceding_space)`
```python
def _generate_correction_options(self, current_token, following_token, has_preceding_space):
    """
    Generate correction candidates for a token.
    
    Args:
        current_token (str): The token to correct
        following_token (str): Next token (for merge operations)
        has_preceding_space (bool): Whether there's a space before the token
    
    Returns:
        List[Tuple[str, int, bool]]: List of (candidate_text, edit_distance, is_split)
    
    Example:
        options = _generate_correction_options("helo", "world", False)
        # Returns: [("helo", 0, False), ("hello", 1, False), ("help", 2, False)]
    """
```

#### Method: `_find_optimal_single_token_cost(path_collection, original_token, correction_options, tokenized_options)`
```python
def _find_optimal_single_token_cost(self, path_collection, original_token, 
                                   correction_options, tokenized_options):
    """
    Find the minimum cost among single-token corrections.
    
    Args:
        path_collection: Current beam search paths
        original_token: Original token being corrected
        correction_options: List of correction candidates
        tokenized_options: Tokenized versions of candidates
    
    Returns:
        float: Minimum cost found for single-token corrections
    
    Note:
        Used for early pruning of multi-token candidates.
    """
```

#### Method: `_calculate_path_cost(current_cost, log_probability, original_token, corrected_token, edit_distance)`
```python
def _calculate_path_cost(self, current_cost, log_probability, 
                        original_token, corrected_token, edit_distance):
    """
    Calculate the total cost for a correction path.
    
    Args:
        current_cost (float): Current path cost
        log_probability (float): Log probability from language model
        original_token (str): Original token
        corrected_token (str): Corrected token
        edit_distance (int): Edit distance between tokens
    
    Returns:
        float: Updated total cost
    
    Cost Formula:
        cost = current_cost - log_probability + error_penalty
        if edit_distance > 0:
            if original is valid word: cost += valid_word_cost
            if first char changed: cost += initial_char_cost
    """
```

#### Method: `_choose_optimal_paths(path_collection)`
```python
def _choose_optimal_paths(self, path_collection):
    """
    Select the best paths to continue in beam search.
    
    Args:
        path_collection (List[dict]): All current paths
    
    Returns:
        List[dict]: Top paths based on beam size
    
    Note:
        Separates active and pending paths, maintaining beam_size of each.
    """
```

#### Method: `_filter_suboptimal_paths(path_collection)`
```python
def _filter_suboptimal_paths(self, path_collection):
    """
    Filter out paths that are unlikely to be optimal.
    
    Args:
        path_collection (List[dict]): Current paths
    
    Returns:
        List[dict]: Filtered paths within threshold of best path
    
    Note:
        Paths with cost > (best_cost + filter_threshold) are pruned.
    """
```

#### Method: `_execute_search_iteration(path_collection, current_token, following_token, has_preceding_space, show_progress)`
```python
def _execute_search_iteration(self, path_collection, current_token, 
                            following_token, has_preceding_space, show_progress):
    """
    Execute one iteration of beam search for a token.
    
    Args:
        path_collection: Current beam search paths
        current_token: Token being processed
        following_token: Next token (for merges)
        has_preceding_space: Whether token has preceding space
        show_progress: Print debug information
    
    Returns:
        List[dict]: Updated path collection
    
    Process:
        1. Generate correction candidates
        2. Evaluate each candidate with GPT-2
        3. Calculate costs and update paths
        4. Prune paths based on beam size and threshold
    """
```

#### Method: `process_text(input_text, show_progress=False)`
```python
def process_text(self, input_text, show_progress=False):
    """
    Main method to correct text.
    
    Args:
        input_text (str): Text to correct
        show_progress (bool): Print detailed progress information
    
    Returns:
        str: Corrected text
    
    Example:
        corrector = NeuralTextCorrector(...)
        corrected = corrector.process_text("helo wrld")
        # Returns: "hello world"
    """
```

---

## Core Module: candidate_generator.py

### Overview
The `candidate_generator.py` module handles the generation of correction candidates using edit distance, word splitting, and merging operations.

### Functions

#### `create_partial_text(term: str, positions_to_skip: Set[int]) -> str`
```python
def create_partial_text(term: str, positions_to_skip: Set[int]) -> str:
    """
    Create a partial text by removing characters at specified positions.
    
    Args:
        term: Original word
        positions_to_skip: Set of character positions to remove
    
    Returns:
        str: Partial text with characters removed
    
    Example:
        create_partial_text("hello", {1, 3})  # Returns: "hlo"
    """
```

#### `generate_partial_variations(term: str, characters_to_skip: int) -> Set[str]`
```python
def generate_partial_variations(term: str, characters_to_skip: int) -> Set[str]:
    """
    Generate all possible partial variations by removing n characters.
    
    Args:
        term: Original word
        characters_to_skip: Number of characters to remove
    
    Returns:
        Set[str]: All possible partial variations
    
    Example:
        generate_partial_variations("cat", 1)
        # Returns: {"at", "ct", "ca"}
    """
```

### Class: TextCandidateEngine

#### Constructor
```python
def __init__(self,
             vocabulary_size: int,
             max_distance: int,
             min_lengths: Dict[int, int],
             enable_space_fixes: bool,
             max_splits: int):
    """
    Initialize the candidate generation engine.
    
    Args:
        vocabulary_size: Number of words to load from frequency list
        max_distance: Maximum edit distance for candidates
        min_lengths: Minimum word length for each edit distance
        enable_space_fixes: Enable split/merge operations
        max_splits: Maximum edit distance for split operations
    """
```

#### Method: `_load_vocabulary()`
```python
def _load_vocabulary(self):
    """
    Load vocabulary from word frequency file.
    
    Reads data/word_frequencies.txt and loads the top N words
    based on vocabulary_size. Filters out words with non-alphabetic
    characters (except hyphens).
    """
```

#### Method: `_include_predefined_terms()`
```python
def _include_predefined_terms(self):
    """
    Add predefined contractions to vocabulary.
    
    Includes common contractions like "I'm", "can't", "won't", etc.
    that might not be in the frequency list.
    """
```

#### Method: `_build_partial_mapping()`
```python
def _build_partial_mapping(self):
    """
    Build an index of partial words to full words.
    
    Creates a mapping from partial words (with characters removed)
    to all possible full words that could generate them.
    This enables efficient candidate lookup.
    
    Example:
        "hllo" -> {"hello", "hallo"}
        "wrld" -> {"world"}
    """
```

#### Method: `_initialize_partial_mapping()`
```python
def _initialize_partial_mapping(self):
    """
    Initialize the partial mapping from cache or build it.
    
    Loads from data/word_stump_index.pkl if available,
    otherwise builds the index and caches it.
    """
```

#### Method: `_search_partial_mapping(term) -> Set[str]`
```python
def _search_partial_mapping(self, term) -> Set[str]:
    """
    Search for candidate words using the partial mapping.
    
    Args:
        term: Word to find candidates for
    
    Returns:
        Set[str]: Potential candidate words
    
    Note:
        Searches both exact matches and partial variations.
    """
```

#### Method: `_apply_match_filters(term, potential_matches, distance_limit, check_initial_character) -> List[Tuple[str, int, bool]]`
```python
def _apply_match_filters(self,
                       term: str,
                       potential_matches: Set[str],
                       distance_limit: int,
                       check_initial_character: bool) -> List[Tuple[str, int, bool]]:
    """
    Filter and score candidate matches.
    
    Args:
        term: Original word
        potential_matches: Set of candidate words
        distance_limit: Maximum allowed edit distance
        check_initial_character: Whether to require matching first character
    
    Returns:
        List of (candidate, distance, is_split) tuples
    
    Note:
        Calculates actual edit distance and filters based on constraints.
    """
```

#### Method: `_find_term_alternatives(term, distance_limit, check_initial_character=False) -> List[Tuple[str, int, bool]]`
```python
def _find_term_alternatives(self, term: str, distance_limit: int, 
                          check_initial_character=False) -> List[Tuple[str, int, bool]]:
    """
    Find alternative words within edit distance.
    
    Args:
        term: Word to find alternatives for
        distance_limit: Maximum edit distance
        check_initial_character: Require matching first character
    
    Returns:
        List of (alternative, distance, False) tuples
    """
```

#### Method: `_find_split_alternatives(input_token, distance_limit) -> List[Tuple[str, int, bool]]`
```python
def _find_split_alternatives(self, input_token: str, distance_limit: int) -> List[Tuple[str, int, bool]]:
    """
    Find alternatives by splitting a word.
    
    Args:
        input_token: Word to split
        distance_limit: Maximum total edit distance
    
    Returns:
        List of (split_words, total_distance, False) tuples
    
    Example:
        _find_split_alternatives("helloworld", 2)
        # Returns: [("hello world", 1, False)]
    """
```

#### Method: `_find_merge_alternatives(first_token, second_token) -> List[Tuple[str, int, bool]]`
```python
def _find_merge_alternatives(self, first_token: str, second_token: str) -> List[Tuple[str, int, bool]]:
    """
    Find alternatives by merging two words.
    
    Args:
        first_token: First word
        second_token: Second word
    
    Returns:
        List of (merged_word, distance, True) tuples
    
    Example:
        _find_merge_alternatives("hel", "lo")
        # Returns: [("hello", 1, True)]
    """
```

#### Method: `generate_alternatives(input_token, subsequent_token) -> List[Tuple[str, int, bool]]`
```python
def generate_alternatives(self, input_token: str, subsequent_token: Optional[str]) -> List[Tuple[str, int, bool]]:
    """
    Generate all correction alternatives for a token.
    
    Args:
        input_token: Token to correct
        subsequent_token: Following token (for merges)
    
    Returns:
        List of (candidate, distance, needs_delay) tuples
    
    Note:
        Combines regular alternatives, splits, merges, and handles capitalization.
    """
```

#### Method: `check_valid_word(term)`
```python
def check_valid_word(self, term):
    """
    Check if a word is in the vocabulary.
    
    Args:
        term (str): Word to check
    
    Returns:
        bool: True if word is valid
    """
```

---

## Web Application: app.py

### Overview
The Flask web application providing REST API and web interface for the spell checker.

### Global Variables
```python
app = Flask(__name__)  # Flask application instance
corrector = None       # Global NeuralTextCorrector instance
```

### Functions

#### `initialize_corrector()`
```python
def initialize_corrector():
    """
    Initialize the neural text corrector with default parameters.
    
    Creates a global corrector instance with optimized settings for
    web usage. Only initializes once to avoid reloading the model.
    
    Side Effects:
        Sets global 'corrector' variable
    """
```

### Routes

#### `@app.route('/')` - `index()`
```python
def index():
    """
    Render the main web interface.
    
    Returns:
        Rendered HTML template (templates/index.html)
    """
```

#### `@app.route('/correct', methods=['POST'])` - `correct_text()`
```python
def correct_text():
    """
    API endpoint for text correction.
    
    Request Body:
        {
            "text": "text to correct"
        }
    
    Response:
        {
            "original": "original text",
            "corrected": "corrected text",
            "processing_time": 0.42,
            "corrections_made": 2,
            "word_count": 5
        }
    
    Status Codes:
        200: Success
        400: No text provided
        500: Processing error
    """
```

#### `@app.route('/examples')` - `get_examples()`
```python
def get_examples():
    """
    Get example sentences for testing.
    
    Returns:
        JSON array of example sentences with spelling errors
    
    Example Response:
        [
            "helo wrld",
            "I hav a speling eror",
            "The qick brown fox jumps ovr the lzy dog"
        ]
    """
```

---

## Utilities and Scripts

### run_web.py
```python
#!/usr/bin/env python3
"""
Quick start script for the web interface.

Usage:
    python run_web.py          # Run with GPU
    python run_web.py --cpu    # Run with CPU only

Features:
    - Sets CUDA_VISIBLE_DEVICES for CPU mode
    - Displays startup information
    - Launches the Flask application
"""
```

### main.py
```python
"""
Command-line interface for the spell checker.

Usage:
    python main.py                          # Interactive mode
    python main.py -f input.txt -o output.txt  # File mode

Features:
    - Interactive text correction
    - Batch file processing
    - Progress reporting
"""
```

### scripts/cli.py
```python
"""
Command-line interface utilities.

Functions:
    - parse_arguments(): Parse command-line arguments
    - interactive_mode(): Run interactive correction loop
    - file_mode(): Process files in batch
"""
```

---

## Data Flow Examples

### Example 1: Simple Correction
```
Input: "helo"

1. Tokenization:
   ["helo"]

2. Candidate Generation:
   - Original: ("helo", 0, False)
   - Edit distance 1: ("hello", 1, False), ("help", 1, False), ("held", 1, False)

3. GPT-2 Evaluation:
   - P("hello"|context) = 0.8
   - P("help"|context) = 0.1
   - P("held"|context) = 0.05
   - P("helo"|context) = 0.001

4. Cost Calculation:
   - "hello": -log(0.8) + 3 (distance penalty) = 3.22
   - "help": -log(0.1) + 3 = 5.30
   - "helo": -log(0.001) + 0 = 6.91

5. Selection:
   "hello" (lowest cost)
```

### Example 2: Split Correction
```
Input: "helloworld"

1. Tokenization:
   ["helloworld"]

2. Candidate Generation:
   - Original: ("helloworld", 0, False)
   - Splits: ("hello world", 1, False)

3. GPT-2 Evaluation:
   - P("hello world"|context) = 0.9
   - P("helloworld"|context) = 0.001

4. Cost Calculation:
   - "hello world": -log(0.9) + 3 = 3.11
   - "helloworld": -log(0.001) + 0 = 6.91

5. Selection:
   "hello world"
```

### Example 3: Context-Aware Correction
```
Input: "I saw a bare in the forest"

1. Processing "bare":
   Candidates: ["bare", "bear", "bar"]

2. Context Evaluation:
   - P("bear"|"I saw a ... in the forest") = 0.95
   - P("bare"|"I saw a ... in the forest") = 0.03
   - P("bar"|"I saw a ... in the forest") = 0.01

3. Selection:
   "bear" (context makes it more likely than "bare")
```

---

## Performance Optimization Techniques

1. **Cached States**: GPT-2 hidden states are cached and reused
2. **Early Pruning**: Candidates filtered before full evaluation
3. **Beam Search**: Limited paths explored simultaneously
4. **Partial Mapping**: Pre-computed index for fast candidate lookup
5. **GPU Acceleration**: Model operations on GPU when available

---

## Error Handling

The system handles various error cases:

1. **Invalid Input**: Returns 400 for empty text
2. **Model Loading**: Lazy loading on first request
3. **CUDA Errors**: Falls back to CPU automatically
4. **Memory Issues**: Beam size limits memory usage
5. **Tokenization**: Handles special characters and punctuation

---

## Configuration Best Practices

1. **Beam Size**: 5-10 for speed, 15-20 for accuracy
2. **Edit Distance**: 2-3 for most cases, 4+ for OCR
3. **Vocabulary**: 50k-100k words for general use
4. **Filter Threshold**: 5-10 for balanced performance
5. **GPU vs CPU**: GPU 10-20x faster for large texts
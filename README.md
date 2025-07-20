# Neural Text Corrector - GPT-2 Based Spell Checker

A sophisticated spell checker that uses GPT-2 language model to correct spelling errors in text. This system combines neural language modeling with traditional spell checking techniques to provide context-aware text corrections.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Technical Details](#technical-details)

## Overview

The Neural Text Corrector is a spell checking system that leverages the power of GPT-2 to understand context and provide accurate spelling corrections. Unlike traditional spell checkers that rely solely on dictionary lookups, this system uses a neural language model to evaluate the likelihood of different correction candidates in context.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Interface                             │
│                    (Flask Application)                           │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REST API Layer                                │
│                  (/correct endpoint)                             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│               NeuralTextCorrector                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  Main Components:                        │    │
│  │                                                         │    │
│  │  1. GPT-2 Language Model (Transformer)                 │    │
│  │  2. TextCandidateEngine (Candidate Generator)          │    │
│  │  3. Beam Search Algorithm                              │    │
│  │  4. Cost Calculation System                            │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
┌──────────────────┐     ┌──────────────────────┐
│ Candidate        │     │   GPT-2 Model        │
│ Generation       │     │   Evaluation         │
│                  │     │                      │
│ - Edit Distance  │     │ - Token Probability  │
│ - Word Splits    │     │ - Context Analysis   │
│ - Word Merges    │     │ - Sequence Scoring   │
└──────────────────┘     └──────────────────────┘
```

## How It Works

### 1. **Input Processing**
   - User enters text through the web interface
   - Text is tokenized into words and punctuation

### 2. **Candidate Generation**
   - For each token, the system generates correction candidates:
     - **Edit Distance Candidates**: Words within a specified edit distance
     - **Split Candidates**: Attempts to split concatenated words (e.g., "helloworld" → "hello world")
     - **Merge Candidates**: Attempts to merge split words (e.g., "hel lo" → "hello")

### 3. **Neural Evaluation**
   - Each candidate is evaluated using GPT-2:
     - Calculates token probabilities in context
     - Uses beam search to explore multiple correction paths
     - Maintains a cost function that combines:
       - Language model probability
       - Edit distance penalty
       - Valid word penalties
       - Initial character change penalties

### 4. **Path Selection**
   - The system maintains multiple correction paths simultaneously
   - Paths are pruned based on cumulative cost
   - The lowest-cost path is selected as the final correction

### Flow Diagram

```
Input Text: "helo wrld"
    │
    ▼
┌─────────────────┐
│   Tokenization  │ → ["helo", " ", "wrld"]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ For each token: │
└────────┬────────┘
         │
    ┌────┴────┐
    │ "helo"  │
    └────┬────┘
         │
         ▼
┌──────────────────────┐
│ Generate Candidates: │
│ - hello (distance 1) │
│ - help (distance 2)  │
│ - held (distance 2)  │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│  GPT-2 Evaluation:   │
│ P("hello"|context)   │ ← Highest probability
│ P("help"|context)    │
│ P("held"|context)    │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│   Cost Calculation:  │
│ Cost = -log(P) +     │
│        edit_penalty  │
└─────────┬────────────┘
          │
          ▼
    ┌─────────┐
    │ "hello" │ ← Selected
    └─────────┘
```

## Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (optional, can run on CPU)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd gpt2-spell-checker-master
```

2. Create a virtual environment:
```bash
conda create -n gpt2-spellchecker python=3.9
conda activate gpt2-spellchecker
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Interface

```bash
# Run with GPU (default)
python run_web.py

# Run with CPU only
python run_web.py --cpu

# Or use environment variable
CUDA_VISIBLE_DEVICES="" python run_web.py
```

The web interface will be available at `http://localhost:5001`

### Command Line Usage

```bash
# Interactive mode
python main.py

# File processing
python main.py -f input.txt -o output.txt

# CPU-only mode
CUDA_VISIBLE_DEVICES="" python main.py
```

### Python API

```python
from core.spell_checker import NeuralTextCorrector

# Initialize the corrector
corrector = NeuralTextCorrector(
    model_name="gpt2",
    tokenizer_name="gpt2",
    vocabulary_size=50000,
    max_edit_distance=3,
    search_beam_size=10,
    min_length_by_distance={1: 1, 2: 4, 3: 6},
    error_penalties={0: 0, 1: 3, 2: 9, 3: 15},
    fix_valid_words=True,
    valid_word_cost=0,
    initial_char_cost=3,
    fix_whitespace=True,
    max_split_distance=2,
    filter_candidates=True,
    filter_paths=True,
    filter_threshold=5
)

# Correct text
corrected_text = corrector.process_text("helo wrld")
print(corrected_text)  # Output: "hello world"
```

## Configuration

The main configuration parameters in `NeuralTextCorrector`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_name` | GPT-2 model variant | "gpt2" |
| `vocabulary_size` | Size of vocabulary for candidate generation | 50000 |
| `max_edit_distance` | Maximum edit distance for candidates | 3 |
| `search_beam_size` | Number of paths to maintain in beam search | 10 |
| `min_length_by_distance` | Minimum word length for each edit distance | {1: 1, 2: 4, 3: 6} |
| `error_penalties` | Cost penalties for each edit distance | {0: 0, 1: 3, 2: 9, 3: 15} |
| `fix_valid_words` | Whether to attempt corrections on valid words | True |
| `valid_word_cost` | Additional cost for changing valid words | 0 |
| `initial_char_cost` | Cost for changing the first character | 3 |
| `fix_whitespace` | Enable split/merge corrections | True |
| `max_split_distance` | Maximum edit distance for split operations | 2 |
| `filter_candidates` | Enable candidate filtering | True |
| `filter_paths` | Enable path filtering | True |
| `filter_threshold` | Threshold for filtering | 5 |

## API Documentation

### POST /correct

Corrects spelling errors in the provided text.

**Request:**
```json
{
    "text": "helo wrld"
}
```

**Response:**
```json
{
    "original": "helo wrld",
    "corrected": "hello world",
    "processing_time": 0.42,
    "corrections_made": 2,
    "word_count": 2
}
```

### GET /examples

Returns example sentences for testing.

**Response:**
```json
[
    "helo wrld",
    "I hav a speling eror",
    "The qick brown fox jumps ovr the lzy dog"
]
```

## Technical Details

### Core Components

1. **NeuralTextCorrector** (`core/spell_checker.py`)
   - Main spell checking engine
   - Implements beam search algorithm
   - Manages GPT-2 model interactions
   - Handles cost calculations and path selection

2. **TextCandidateEngine** (`core/candidate_generator.py`)
   - Generates correction candidates
   - Maintains word frequency dictionary
   - Implements edit distance calculations
   - Handles word splitting/merging logic

3. **Web Application** (`app.py`)
   - Flask-based REST API
   - Handles HTTP requests
   - Manages model initialization
   - Provides web interface

### Algorithm Details

The spell checker uses a **beam search algorithm** with the following key features:

1. **Multi-path Exploration**: Maintains multiple correction hypotheses simultaneously
2. **Context-Aware Scoring**: Uses GPT-2 to evaluate candidates in context
3. **Efficient Pruning**: Filters unlikely paths early to improve performance
4. **Cached States**: Reuses GPT-2 hidden states for efficiency

### Performance Considerations

- **GPU Acceleration**: Significantly faster with CUDA-enabled GPU
- **Beam Size Trade-off**: Larger beams increase accuracy but reduce speed
- **Vocabulary Size**: Larger vocabularies provide more candidates but increase computation
- **Caching**: Word stump index is cached to speed up candidate generation

## File Structure

```
gpt2-spell-checker-master/
├── app.py                  # Flask web application
├── run_web.py             # Quick start script
├── config.yml             # Configuration file
├── requirements.txt       # Python dependencies
├── core/
│   ├── spell_checker.py   # Main spell checking engine
│   └── candidate_generator.py  # Candidate generation logic
├── data/
│   ├── word_frequencies.txt    # Word frequency list
│   └── word_stump_index.pkl   # Cached word index
├── static/
│   ├── css/
│   │   └── style.css     # Web interface styles
│   └── js/
│       └── main.js       # Frontend JavaScript
├── templates/
│   └── index.html        # Web interface template
├── scripts/
│   └── cli.py           # Command line interface
└── utils/
    ├── add_space_errors.py     # Testing utilities
    ├── filter_word_list.py     # Dictionary processing
    └── split_benchmark.py      # Performance testing
```

## Examples

```python
# Basic usage
text = "I cant beleive its hapenning"
corrected = corrector.process_text(text)
# Output: "I can't believe it's happening"

# Complex errors
text = "The qick brown fox jumps ovr the lzy dog"
corrected = corrector.process_text(text)
# Output: "The quick brown fox jumps over the lazy dog"

# Split/merge corrections
text = "helloworld needsto be fixed"
corrected = corrector.process_text(text)
# Output: "hello world needs to be fixed"
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License.
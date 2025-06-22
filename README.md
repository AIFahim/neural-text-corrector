# Neural Text Corrector

A context-aware text correction system powered by GPT-2 language model that corrects spelling errors by understanding context and meaning. Available as both command-line tool and web interface.

## Features

- Context-aware spelling correction using neural language models
- Corrects complex errors including real-word errors, split/merged words
- Interactive CLI and batch file processing
- Configurable correction behavior
- GPU acceleration support (optional)

## Requirements

- Python 3.9+
- Dependencies in requirements.txt

## Installation

```bash
# Navigate to the project directory
cd /mnt/c/Users/MdAsifIqbalFahim/PycharmProjects/spel_checker/gpt2-spell-checker-master

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Web Interface (NEW!)

Start the web server:
```bash
# Install Flask if not already installed
pip install flask

# Run the web interface
python app.py

# Or use the run script
python run_web.py

# For CPU-only mode
python run_web.py --cpu
```

Then open your browser and go to: http://localhost:5001

The web interface features:
- Clean, modern UI
- Real-time text correction
- Side-by-side comparison with highlighted changes
- Statistics (words, corrections, processing time)
- Example sentences
- Copy to clipboard functionality

### Interactive Mode (Command Line)

```bash
python main.py
```

Example:
```
> helo wrld
help world

> I cant beleive its hapenning
I can't believe it's happening
```

### File Processing

```bash
# Process a file
python main.py -f input.txt -o output.txt

# CPU-only mode (if GPU issues)
CUDA_VISIBLE_DEVICES="" python main.py -f input.txt -o output.txt
```

### Python API

```python
import sys
sys.path.append('path/to/neural-text-corrector/gpt2_spell_checker')
from core.spell_checker import NeuralTextCorrector

# Initialize
corrector = NeuralTextCorrector(
    model_name="gpt2",
    tokenizer_name="gpt2",
    vocabulary_size=50000,
    max_edit_distance=3,
    search_beam_size=10
)

# Correct text
text = "I hav a speling eror"
corrected = corrector.process_text(text)
print(corrected)  # "I have a spelling error"
```

## Configuration

Edit `config.yml` to customize:

- `model`: GPT-2 model size (gpt2, gpt2-medium, gpt2-large)
- `beam_width`: Number of candidates to consider
- `maximum_edit_distance`: Maximum character changes allowed
- `correct_real_words`: Enable/disable real-word error correction
- `correct_spaces`: Enable/disable whitespace correction

## Project Structure

```
neural-text-corrector/
├── gpt2_spell_checker/
│   ├── core/
│   │   ├── spell_checker.py      # Main correction engine
│   │   └── candidate_generator.py # Candidate generation
│   ├── scripts/
│   │   └── cli.py                # Command-line interface
│   └── utils/                    # Utility scripts
├── static/                       # Web interface assets
│   ├── css/
│   │   └── style.css            # Styling
│   └── js/
│       └── main.js              # Frontend logic
├── templates/
│   └── index.html               # Web interface template
├── data/
│   └── word_frequencies.txt      # Word frequency data
├── app.py                        # Flask web application
├── config.yml                    # Configuration
├── requirements.txt              # Dependencies
├── main.py                       # CLI entry point
├── run_web.py                    # Web interface launcher
├── example.py                    # Usage examples
├── setup.py                      # Package setup
└── README.md                     # This file
```

## Examples

See `example.py` for detailed usage examples including:
- Basic corrections
- Advanced configuration
- Batch processing
- Performance measurement
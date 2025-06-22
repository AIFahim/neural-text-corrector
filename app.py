#!/usr/bin/env python3
"""
Flask Web Application for Neural Text Corrector
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'gpt2_spell_checker'))

from core.spell_checker import NeuralTextCorrector
import time

app = Flask(__name__)

# Global variable to store the corrector instance
corrector = None

def initialize_corrector():
    """Initialize the neural text corrector"""
    global corrector
    if corrector is None:
        print("Loading neural text corrector model...")
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
        print("Model loaded successfully!")

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/correct', methods=['POST'])
def correct_text():
    """API endpoint to correct text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Initialize corrector if not already done
        if corrector is None:
            initialize_corrector()
        
        # Measure correction time
        start_time = time.time()
        corrected_text = corrector.process_text(text)
        processing_time = time.time() - start_time
        
        # Calculate basic statistics
        original_words = text.split()
        corrected_words = corrected_text.split()
        corrections_made = sum(1 for o, c in zip(original_words, corrected_words) if o != c)
        
        return jsonify({
            'original': text,
            'corrected': corrected_text,
            'processing_time': round(processing_time, 2),
            'corrections_made': corrections_made,
            'word_count': len(original_words)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/examples')
def get_examples():
    """Get example sentences"""
    examples = [
        "helo wrld",
        "I hav a speling eror",
        "The qick brown fox jumps ovr the lzy dog",
        "Can yu plese corect these mistaks",
        "Ths is a test of the spel cheker",
        "I cant beleive its hapenning",
        "programing is fun",
        "artifical inteligence is the futur"
    ]
    return jsonify(examples)

if __name__ == '__main__':
    # Initialize the corrector on startup
    print("Starting Neural Text Corrector Web Application...")
    initialize_corrector()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5001)
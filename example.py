#!/usr/bin/env python3
"""
Simple examples of Neural Text Corrector usage
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'gpt2_spell_checker'))

from core.spell_checker import NeuralTextCorrector


def main():
    print("Neural Text Corrector - Examples\n")
    
    # Initialize the corrector
    print("Loading model...")
    corrector = NeuralTextCorrector(
        model_name="gpt2",
        tokenizer_name="gpt2",
        vocabulary_size=50000,
        max_edit_distance=3,
        search_beam_size=10
    )
    
    # Example sentences with errors
    test_sentences = [
        "helo wrld",
        "I hav a speling eror",
        "The qick brown fox",
        "Can yu plese help",
        "Ths is amazng",
        "I cant beleive it",
        "programing is fun",
        "artifical inteligence"
    ]
    
    print("\nCorrecting sentences:")
    print("-" * 50)
    
    for sentence in test_sentences:
        corrected = corrector.process_text(sentence)
        print(f"Original:  {sentence}")
        print(f"Corrected: {corrected}")
        print()


if __name__ == "__main__":
    main()
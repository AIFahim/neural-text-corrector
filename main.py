#!/usr/bin/env python3
"""
Neural Text Corrector - Main Entry Point
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'gpt2_spell_checker'))

from scripts.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
#!/usr/bin/env python3
"""
Quick start script for the web interface
"""

import os
import sys

print("Starting Neural Text Corrector Web Interface...")
print("The model will be loaded on first use.")
print("Open your browser and go to: http://localhost:5001")
print("Press Ctrl+C to stop the server\n")

# Set environment variable to avoid CUDA if needed
if len(sys.argv) > 1 and sys.argv[1] == '--cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    print("Running in CPU mode")

# Run the app
os.system('python app.py')
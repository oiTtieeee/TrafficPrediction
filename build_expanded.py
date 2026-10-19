import subprocess
import fitz

# Read the current main.tex preamble and structure
with open('paper/main.tex', 'r', encoding='utf-8') as f:
    current = f.read()

# Let's inspect the sections in main.tex
print("Current main.tex read.")

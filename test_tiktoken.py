import os
import tiktoken
from pathlib import Path

# Look for the tokenizer file in multiple locations
possible_paths = [
    Path("cl100k_base.tiktoken"),  # Current directory
    Path.home() / ".cache" / "tiktoken" / "cl100k_base.tiktoken",  # Default cache location
    Path("C:/tiktoken_cache/cl100k_base.tiktoken")  # Windows specific location
]

# Find the first existing tokenizer file
tokenizer_path = None
for path in possible_paths:
    if path.exists():
        tokenizer_path = path
        print(f"Found tokenizer file at: {path}")
        break

if tokenizer_path is None:
    print("Could not find tokenizer file in any of the expected locations!")
    exit(1)

try:
    # Try to use the tokenizer
    enc = tiktoken.get_encoding("cl100k_base")
    print("Successfully loaded tokenizer!")
    
    # Test the tokenizer
    test_text = "Hello, world!"
    tokens = enc.encode(test_text)
    print(f"Test encoding of '{test_text}': {tokens}")
    print(f"Decoded back: {enc.decode(tokens)}")
    
except Exception as e:
    print(f"Error loading tokenizer: {str(e)}")
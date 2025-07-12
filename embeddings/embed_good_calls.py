import os
import glob
from typing import Dict, List
import json
from pinecone_store import PineconeManager
import re
import tiktoken

# Initialize Pinecone manager
pinecone_manager = PineconeManager()

MODEL_NAME = "text-embedding-3-large"

MAX_CHUNK_TOKENS = 1500
CHUNK_OVERLAP_TOKENS = 200

# Use OpenAI's tiktoken for accurate token counting
encoding = tiktoken.encoding_for_model("gpt-4")

def chunk_text_by_tokens(text: str, max_tokens: int = MAX_CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS) -> List[str]:
    """
    Split text into overlapping chunks of max_tokens tokens, with specified overlap.
    Returns a list of chunk strings.
    """
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    text_length = len(tokens)
    while start < text_length:
        end = min(start + max_tokens, text_length)
        chunk_tokens = tokens[start:end]
        chunk = encoding.decode(chunk_tokens)
        chunks.append(chunk)
        if end == text_length:
            break
        start += max_tokens - overlap
    return chunks

def embed_all_good_calls():
    """Read all cleaned transcripts and store them in Pinecone, chunking by tokens if needed."""
    good_calls_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 'data', 'good_calls')
    txt_files = glob.glob(os.path.join(good_calls_dir, '*.txt'))
    print(f"Found {len(txt_files)} transcripts to process...")
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
            chunks = chunk_text_by_tokens(transcript)
            total_chunks = len(chunks)
            print(f"\nProcessing {os.path.basename(file_path)}")
            print(f"Split into {total_chunks} token-based chunks")
            for idx, chunk in enumerate(chunks):
                metadata = {
                    'transcript': chunk,
                    'chunk_number': idx + 1,
                    'total_chunks': total_chunks,
                    'chunk_length': len(chunk)
                }
                vector_id = f"chunk{idx+1}"
                pinecone_manager.store_transcript(chunk, metadata)
                print(f"✓ Stored chunk {idx+1}/{total_chunks} (length: {len(chunk)} chars)")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    # Print index statistics
    stats = pinecone_manager.get_index_stats()
    print("\nPinecone Index Statistics:")
    print(stats)

if __name__ == '__main__':
    print("Starting to embed good calls (token-based chunking)...")
    embed_all_good_calls()
    print("Done!") 
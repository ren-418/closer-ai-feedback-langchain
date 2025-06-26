import os
import openai
from typing import List, Dict

def analyze_transcript(transcript: str) -> dict:
    """
    Analyze a sales call transcript using OpenAI and return structured results.
    """
    # TODO: Implement with OpenAI API
    return {
        'scores': {},
        'feedback': {},
        'insights': {},
    } 

MAX_CHUNK_LENGTH = 3000
CHUNK_OVERLAP = 300

def chunk_text(text: str, max_length: int = MAX_CHUNK_LENGTH, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks of max_length chars, with specified overlap."""
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + max_length, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_length:
            break
        start += max_length - overlap
    return chunks


def embed_new_transcript(transcript_text: str) -> List[Dict]:
    """
    Splits the transcript into overlapping chunks, generates embeddings for each chunk,
    and returns a list of dicts with chunk_text, embedding, chunk_number, total_chunks, and chunk_length.
    """
    chunks = chunk_text(transcript_text)
    total_chunks = len(chunks)
    results = []
    for idx, chunk in enumerate(chunks):
        # Generate embedding using OpenAI
        response = openai.embeddings.create(
            input=chunk,
            model="text-embedding-3-large"
        )
        embedding = response.data[0].embedding
        results.append({
            'chunk_text': chunk,
            'embedding': embedding,
            'chunk_number': idx + 1,
            'total_chunks': total_chunks,
            'chunk_length': len(chunk)
        })
    return results 

if __name__ == "__main__":
    # Example usage for testing
    sample_text = (
        "Hello, this is a test transcript. " * 200  # Make a long enough sample
    )
    results = embed_new_transcript(sample_text)
    print(f"Total chunks: {len(results)}")
    for r in results:
        print(f"Chunk {r['chunk_number']}/{r['total_chunks']} (length: {r['chunk_length']}): {r['chunk_text'][:60]}...")
        print(f"Embedding (first 5 dims): {r['embedding'][:5]}") 
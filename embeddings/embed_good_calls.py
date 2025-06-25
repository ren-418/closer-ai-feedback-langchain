import os
import glob
from typing import Dict, List
import json
from pinecone_store import PineconeManager
import re

# Initialize Pinecone manager
pinecone_manager = PineconeManager()

MODEL_NAME = "text-embedding-3-large"

MAX_CHUNK_LENGTH = 3000
CHUNK_OVERLAP = 300

def extract_metadata_from_filename(filename: str) -> Dict:
    """Extract minimal metadata from filename (closer name, date, etc.)."""
    base = os.path.basename(filename)
    name = base.replace('.txt', '')
    # Try to extract date if present
    date_str = ''
    for month in ['May', 'Jun', 'June']:
        if month in base:
            try:
                day = int(''.join(filter(str.isdigit, base.split(month)[1].split()[0])))
                date_str = f"2025-{month}-{day:02d}"
            except:
                pass
    return {
        'filename': base,
        'closer_name': name.split(' - ')[0] if ' - ' in name else name.split(' X ')[0],
        'date': date_str
    }

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to use as file_id (remove extension, spaces, special chars)."""
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)

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

def embed_all_good_calls():
    """Read all cleaned transcripts and store them in Pinecone, chunking by characters if needed."""
    good_calls_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 'data', 'good_calls')
    # Get all txt files
    txt_files = glob.glob(os.path.join(good_calls_dir, '*.txt'))
    print(f"Found {len(txt_files)} transcripts to process...")
    
    for file_path in txt_files:
        try:
            # Read transcript
            with open(file_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
            
            # Get file_id for all chunks from this file
            file_id = sanitize_filename(file_path)
            
            # Get chunks using overlapping character-based chunking
            chunks = chunk_text(transcript)
            total_chunks = len(chunks)
            print(f"\nProcessing {os.path.basename(file_path)}")
            print(f"Split into {total_chunks} chunks")
            
            metadata_base = extract_metadata_from_filename(file_path)
            metadata_base['file_id'] = file_id  
            for idx, chunk in enumerate(chunks):
                metadata = metadata_base.copy()
                metadata['transcript'] = chunk
                metadata['chunk_number'] = idx + 1
                metadata['total_chunks'] = total_chunks
                metadata['chunk_length'] = len(chunk)
                metadata['file_id'] = file_id  
                
                # Make vector_id unique per chunk
                vector_id = f"{file_id}_chunk{idx+1}"
                pinecone_manager.store_transcript(chunk, {**metadata, 'filename': vector_id})
                print(f"✓ Stored chunk {idx+1}/{total_chunks} (length: {len(chunk)} chars)")
        
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Print index statistics
    stats = pinecone_manager.get_index_stats()
    print("\nPinecone Index Statistics:")
    print(json.dumps(stats, indent=2))

if __name__ == '__main__':
    print("Starting to embed good calls...")
    embed_all_good_calls()
    print("Done!") 
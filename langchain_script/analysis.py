import os
from typing import List, Dict, Any
from openai import OpenAI
import json
from embeddings.pinecone_store import PineconeManager
import tiktoken

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize Pinecone manager
pinecone_manager = PineconeManager()

MAX_CHUNK_TOKENS = 3000
CHUNK_OVERLAP_TOKENS = 300
CONTEXT_WINDOW_TOKENS = 300  # Reduced for safety
MAX_REF_CHUNKS = 2
MAX_REF_TOKENS = 500
MAX_TOTAL_PROMPT_TOKENS = 8000

# Use OpenAI's tiktoken for accurate token counting
encoding = tiktoken.encoding_for_model("gpt-4")

def chunk_text_by_tokens(text: str, max_tokens: int = MAX_CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS) -> List[List[int]]:
    """
    Split text into overlapping chunks of max_tokens tokens, with specified overlap.
    Returns a list of token lists (not decoded yet).
    """
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    text_length = len(tokens)
    while start < text_length:
        end = min(start + max_tokens, text_length)
        chunk_tokens = tokens[start:end]
        chunks.append(chunk_tokens)
        if end == text_length:
            break
        start += max_tokens - overlap
    print(f"[Chunking] Split transcript into {len(chunks)} chunks (max {max_tokens} tokens, overlap {overlap})")
    return chunks

def get_context_window(chunks: List[List[int]], idx: int, context_window: int = CONTEXT_WINDOW_TOKENS) -> Dict[str, str]:
    """
    For a given chunk index, return the decoded previous and next context windows (if available).
    """
    prev_context = []
    next_context = []
    if idx > 0:
        prev_tokens = chunks[idx-1][-context_window:]
        prev_context = encoding.decode(prev_tokens)
    if idx < len(chunks) - 1:
        next_tokens = chunks[idx+1][:context_window]
        next_context = encoding.decode(next_tokens)
    return {
        'prev': prev_context,
        'next': next_context
    }

def truncate_reference_chunk(text: str, max_tokens: int = MAX_REF_TOKENS) -> str:
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return encoding.decode(tokens)

def embed_new_transcript(transcript_text: str) -> List[Dict]:
    """
    Splits the transcript into overlapping token-based chunks, generates embeddings for each chunk,
    and returns a list of dicts with chunk_text, embedding, chunk_number, total_chunks, chunk_length, and context windows.
    """
    chunk_token_lists = chunk_text_by_tokens(transcript_text)
    total_chunks = len(chunk_token_lists)
    results = []
    print(f"[Embedding] Generating embeddings for {total_chunks} chunks...")
    for idx, chunk_tokens in enumerate(chunk_token_lists):
        chunk = encoding.decode(chunk_tokens)
        print("embed each chunk ::", chunk)
        context = get_context_window(chunk_token_lists, idx)
        print("context :::", context)
        embedding = pinecone_manager.generate_embedding(chunk)
        results.append({
            'chunk_text': chunk,
            'embedding': embedding,
            'chunk_number': idx + 1,
            'total_chunks': total_chunks,
            'chunk_length': len(chunk),
            'context_prev': context['prev'],
            'context_next': context['next']
        })
        if (idx + 1) % 10 == 0 or (idx + 1) == total_chunks:
            print(f"[Embedding] Processed {idx + 1}/{total_chunks} chunks")
    return results

def build_chunk_analysis_prompt(chunk_text: str, reference_texts: List[Dict], context_prev: str = '', context_next: str = '') -> str:
    """
    Build a prompt for analyzing a chunk with reference examples and context window.
    """
    prompt = (
        "You are an expert sales call evaluator. Analyze this sales call chunk compared to reference examples.\n\n"
        "PREVIOUS CONTEXT:\n"
        f"{context_prev}\n\n" if context_prev else ""
        "CURRENT CHUNK:\n"
        f"```\n{chunk_text}\n```\n\n"
        "NEXT CONTEXT:\n"
        f"{context_next}\n\n" if context_next else ""
        "REFERENCE EXAMPLES FROM GOOD CALLS:\n"
    )
    for i, ref in enumerate(reference_texts[:MAX_REF_CHUNKS], 1):
        ref_text = truncate_reference_chunk(ref['metadata']['transcript'], MAX_REF_TOKENS)
        prompt += f"\nExample {i}:\n```\n{ref_text}\n```\n"
    prompt += (
        "\nProvide a detailed analysis in JSON format with the following structure:\n"
        "{\n"
        '  "strengths": ["strength1", "strength2", ...],\n'
        '  "weaknesses": ["weakness1", "weakness2", ...],\n'
        '  "suggestions": ["suggestion1", "suggestion2", ...],\n'
        '  "score": 0-100,\n'
        '  "letter_grade": "A/B/C",\n'
        '  "key_metrics": {\n'
        '    "rapport_building": 1-10,\n'
        '    "discovery": 1-10,\n'
        '    "objection_handling": 1-10,\n'
        '    "pitch_delivery": 1-10,\n'
        '    "closing_effectiveness": 1-10\n'
        '  }\n'
        "}"
    )
    return prompt

def analyze_chunk_with_rag(chunk_text: str, reference_chunks: List[Dict], context_prev: str = '', context_next: str = '', temperature: float = 0.3) -> Dict:
    """
    Analyze a chunk using RAG with reference examples from good calls and context window.
    """
    prompt = build_chunk_analysis_prompt(chunk_text, reference_chunks, context_prev, context_next)
    # Ensure prompt is within model context window
    prompt_tokens = len(encoding.encode(prompt))
    if prompt_tokens > MAX_TOTAL_PROMPT_TOKENS:
        # Truncate context windows if needed
        context_prev = truncate_reference_chunk(context_prev, 100)
        context_next = truncate_reference_chunk(context_next, 100)
        prompt = build_chunk_analysis_prompt(chunk_text, reference_chunks, context_prev, context_next)
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1000
    )
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {
            "error": "Failed to parse LLM response as JSON",
            "raw_response": response.choices[0].message.content
        }

def aggregate_chunk_analyses(chunk_analyses: List[Dict]) -> Dict:
    """
    Aggregate all chunk-level analyses into a holistic summary.
    Returns a comprehensive evaluation report.
    """
    prompt = (
        "You are an expert sales call evaluator. Based on the following chunk-level analyses, "
        "provide a comprehensive evaluation of the entire sales call.\n\n"
        "CHUNK ANALYSES:\n"
        f"{json.dumps(chunk_analyses, indent=2)}\n\n"
        "Provide a final report in JSON format with:\n"
        "{\n"
        '  "overall_score": 0-100,\n'
        '  "letter_grade": "A/B/C",\n'
        '  "key_strengths": ["strength1", "strength2", ...],\n'
        '  "key_weaknesses": ["weakness1", "weakness2", ...],\n'
        '  "actionable_suggestions": ["suggestion1", "suggestion2", ...],\n'
        '  "key_metrics": {\n'
        '    "rapport_building": 1-10,\n'
        '    "discovery": 1-10,\n'
        '    "objection_handling": 1-10,\n'
        '    "pitch_delivery": 1-10,\n'
        '    "closing_effectiveness": 1-10\n'
        '  },\n'
        '  "summary": "Detailed summary of the overall performance..."\n'
        "}"
    )
    
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {
            "error": "Failed to parse LLM response as JSON",
            "raw_response": response.choices[0].message.content
        }

if __name__ == "__main__":
    # Example usage for testing
    sample_text = (
        "Sales Rep: Hi there! Thanks for taking the time to chat today. "
        "Could you tell me about your current challenges?\n\n"
        "Prospect: Well, we're struggling with our manual processes..."
    )
    
    # Test the full pipeline
    chunks_data = embed_new_transcript(sample_text)
    print(f"\nProcessed {len(chunks_data)} chunks")
    
    for chunk_data in chunks_data:
        # Find similar chunks from good calls
        similar_chunks = pinecone_manager.find_similar_calls(
            chunk_data['chunk_text'],
            top_k=3
        )
        print(f"\nFound {len(similar_chunks)} similar chunks for comparison")
        
        # Analyze the chunk
        analysis = analyze_chunk_with_rag(
            chunk_data['chunk_text'],
            similar_chunks,
            chunk_data['context_prev'],
            chunk_data['context_next']
        )
        print("\nChunk Analysis:")
        print(json.dumps(analysis, indent=2)) 
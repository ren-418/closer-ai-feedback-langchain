import os
from typing import List, Dict, Any
from openai import OpenAI
import json
from embeddings.pinecone_store import PineconeManager

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize Pinecone manager
pinecone_manager = PineconeManager()

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
        embedding = pinecone_manager.generate_embedding(chunk)
        results.append({
            'chunk_text': chunk,
            'embedding': embedding,
            'chunk_number': idx + 1,
            'total_chunks': total_chunks,
            'chunk_length': len(chunk)
        })
    return results

def build_chunk_analysis_prompt(chunk_text: str, reference_texts: List[Dict]) -> str:
    """Build a prompt for analyzing a chunk with reference examples."""
    prompt = (
        "You are an expert sales call evaluator. Analyze this sales call chunk compared to reference examples.\n\n"
        "CURRENT CHUNK:\n"
        f"```\n{chunk_text}\n```\n\n"
        "REFERENCE EXAMPLES FROM GOOD CALLS:\n"
    )
    
    for i, ref in enumerate(reference_texts, 1):
        prompt += f"\nExample {i}:\n```\n{ref['metadata']['transcript']}\n```\n"
    
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

def analyze_chunk_with_rag(chunk_text: str, reference_chunks: List[Dict], temperature: float = 0.3) -> Dict:
    """Analyze a chunk using RAG with reference examples from good calls."""
    prompt = build_chunk_analysis_prompt(chunk_text, reference_chunks)
    
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
            similar_chunks
        )
        print("\nChunk Analysis:")
        print(json.dumps(analysis, indent=2)) 
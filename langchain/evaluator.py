#this is evaluator script

from typing import Dict, Optional
import json
from openai import OpenAI
import os
from .analysis import embed_new_transcript, analyze_chunk_with_rag, aggregate_chunk_analyses
from .transcript_parser import parse_transcript
from embeddings.pinecone_store import PineconeManager

class SalesCallEvaluator:
    def __init__(self):
        """Initialize the sales call evaluator with necessary clients."""
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.pinecone_manager = PineconeManager()
    
    def evaluate_transcript(self, transcript: str, top_k: int = 3) -> Dict:
        """
        Evaluate a sales call transcript using RAG and structured analysis.
        Returns a comprehensive evaluation report.
        """
        try:
            # 1. Parse transcript into sections
            parsed = parse_transcript(transcript)
            
            # 2. Chunk and embed the transcript
            chunks_data = embed_new_transcript(transcript)
            
            # 3. Analyze each chunk with RAG
            chunk_analyses = []
            for chunk_data in chunks_data:
                # Find similar chunks from good calls
                similar_chunks = self.pinecone_manager.find_similar_calls(
                    chunk_data['chunk_text'],
                    top_k=top_k
                )
                
                # Analyze this chunk
                analysis = analyze_chunk_with_rag(
                    chunk_data['chunk_text'],
                    similar_chunks
                )
                
                chunk_analyses.append({
                    'chunk_number': chunk_data['chunk_number'],
                    'total_chunks': chunk_data['total_chunks'],
                    'analysis': analysis
                })
            
            # 4. Aggregate analyses into final report
            final_analysis = aggregate_chunk_analyses([c['analysis'] for c in chunk_analyses])
            
            # 5. Combine everything into a structured report
            return {
                'transcript_analysis': {
                    'sections': parsed,
                    'speaker_stats': parsed.get('metadata', {})
                },
                'chunk_analyses': chunk_analyses,
                'final_analysis': final_analysis,
                'metadata': {
                    'total_chunks': len(chunks_data),
                    'references_per_chunk': top_k
                }
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed'
            }
    
    def evaluate_transcript_file(self, file_path: str, encoding: str = 'utf-8') -> Dict:
        """
        Evaluate a sales call transcript from a file.
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                transcript = f.read()
            return self.evaluate_transcript(transcript)
        except Exception as e:
            return {
                'error': f"Failed to process file {file_path}: {str(e)}",
                'status': 'failed'
            }

if __name__ == "__main__":
    # Test the evaluator
    evaluator = SalesCallEvaluator()
    
    sample_transcript = """
    Sales Rep: Hi there! Thanks for taking the time to chat today about our sales automation platform.
    
    Prospect: Thanks for having me. I've been looking into solutions like this.
    
    Sales Rep: Great to hear! Could you tell me about your current sales process and what challenges you're facing?
    
    Prospect: Well, we're a growing company and our sales team is struggling to keep up with leads.
    We're using a basic CRM but it's not really helping us automate anything.
    
    Sales Rep: I understand completely. Managing leads manually can be overwhelming, especially as you grow.
    Our platform actually helped similar companies increase their lead processing capacity by 3x while reducing manual work.
    Would you be interested in seeing a quick demo of how we do that?
    
    Prospect: Yes, that would be helpful. We definitely need to improve our efficiency.
    """
    
    result = evaluator.evaluate_transcript(sample_transcript)
    print("\nEvaluation Result:")
    print(json.dumps(result, indent=2))


#this is evaluator script

from typing import Dict, Optional
import json
from openai import OpenAI
import os
from datetime import datetime
from .analysis import embed_new_transcript, analyze_chunk_with_rag, aggregate_chunk_analyses
from embeddings.pinecone_store import PineconeManager

class SalesCallEvaluator:
    def __init__(self):
        """Initialize the sales call evaluator with necessary clients."""
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.pinecone_manager = PineconeManager()
    
    def evaluate_transcript(self, transcript: str, top_k: int = 3) -> Dict:
        """
        Evaluate a sales call transcript using RAG and structured analysis.
        Returns a comprehensive evaluation report with reference file tracking.
        Handles large transcripts by chunking and progress reporting.
        """
        try:
            print("[Evaluator] Starting transcript evaluation...")
            print(f"[Evaluator] Transcript length: {len(transcript)} characters")
            
            # Chunk and embed the transcript
            print("[Evaluator] Chunking and embedding transcript...")
            chunks_data = embed_new_transcript(transcript)
            total_chunks = len(chunks_data)
            print(f"[Evaluator] {total_chunks} chunks to analyze.")
            
            # Analyze each chunk
            chunk_analyses = []
            
            for idx, chunk_data in enumerate(chunks_data):
                print(f"[Evaluator] Analyzing chunk {idx+1}/{total_chunks}...")
                
                # Find similar chunks from good calls
                similar_chunks = self.pinecone_manager.find_similar_calls(
                    chunk_data['chunk_text'],
                    top_k=top_k
                )
                
                # Analyze the chunk with enhanced analysis
                analysis = analyze_chunk_with_rag(
                    chunk_data['chunk_text'],
                    similar_chunks,
                    chunk_data['context_prev'],
                    chunk_data['context_next'],
                    model="gpt-4o"
                )
                
                chunk_analyses.append({
                    'chunk_number': chunk_data['chunk_number'],
                    'total_chunks': chunk_data['total_chunks'],
                    'chunk_text_preview': chunk_data['chunk_text'][:200] + "..." if len(chunk_data['chunk_text']) > 200 else chunk_data['chunk_text'],
                    'analysis': analysis
                })
            
            # Aggregate all chunk analyses into final report
            print("[Evaluator] Aggregating chunk analyses into final report...")
            final_analysis = aggregate_chunk_analyses([c['analysis'] for c in chunk_analyses], model="gpt-4o")
            
            print("[Evaluator] Evaluation complete.")
            
            # Enhanced metadata with reference tracking
            metadata = {
                'total_chunks': total_chunks,
                'references_per_chunk': top_k,
                'total_reference_files_used': 0,
                'evaluation_timestamp': datetime.now().isoformat(),
                'transcript_length': len(transcript),
                'estimated_call_duration': f"{total_chunks * 2-3} minutes"  # Rough estimate
            }
            
            return {
                'transcript_analysis': {
                    'summary': final_analysis.get('executive_summary', {}),
                    'overall_score': final_analysis.get('executive_summary', {}).get('overall_score', 0),
                    'letter_grade': final_analysis.get('executive_summary', {}).get('letter_grade', 'C')
                },
                'chunk_analyses': chunk_analyses,
                'final_analysis': final_analysis,
                'metadata': metadata,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"[Evaluator] Error: {e}")
            return {
                'error': str(e),
                'status': 'failed',
                'metadata': {
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'error_type': type(e).__name__
                }
            }
    
    def evaluate_transcript_file(self, file_path: str, encoding: str = 'utf-8') -> Dict:
        """
        Evaluate a sales call transcript from a file.
        Handles large files and reports progress.
        """
        try:
            print(f"[Evaluator] Loading transcript from file: {file_path}")
            with open(file_path, 'r', encoding=encoding) as f:
                transcript = f.read()
            
            # Add file metadata
            result = self.evaluate_transcript(transcript)
            if 'metadata' in result:
                result['metadata']['source_file'] = file_path
                result['metadata']['file_size'] = len(transcript)
            
            return result
            
        except Exception as e:
            print(f"[Evaluator] File error: {e}")
            return {
                'error': f"Failed to process file {file_path}: {str(e)}",
                'status': 'failed',
                'metadata': {
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'source_file': file_path,
                    'error_type': type(e).__name__
                }
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


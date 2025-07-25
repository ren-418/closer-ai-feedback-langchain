#this is evaluator script

from typing import Dict, Optional
import json
from openai import OpenAI
import os
from datetime import datetime
import re
from .analysis import embed_new_transcript, analyze_chunk_with_rag, aggregate_chunk_analyses, clean_final_report_with_ai
from embeddings.pinecone_store import PineconeManager
from database.database_manager import DatabaseManager

class SalesCallEvaluator:
    def __init__(self):
        """Initialize the sales call evaluator with necessary clients."""
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.pinecone_manager = PineconeManager()
        self.db_manager = DatabaseManager()
    
    def evaluate_transcript(self, transcript: str, top_k: int = 3) -> Dict:
        """
        Evaluate a sales call transcript using RAG and structured analysis.
        Uses OpenAI Batch API for chunk analysis if possible.
        Returns a comprehensive evaluation report with reference file tracking.
        Handles large transcripts by chunking and progress reporting.
        """
        import tempfile
        import time
        try:
            print("[Evaluator] Starting transcript evaluation...")
            print(f"[Evaluator] Transcript length: {len(transcript)} characters")
            
            # At the start of evaluate_transcript, fetch business rules
            business_rules = self.db_manager.get_business_rules()
            
            # Chunk and embed the transcript
            print("[Evaluator] Chunking and embedding transcript...")
            chunks_data = embed_new_transcript(transcript)
            total_chunks = len(chunks_data)
            print(f"[Evaluator] {total_chunks} chunks to analyze.")

            # --- BATCH API WORKFLOW START ---
            print("[Evaluator] Preparing batch file for OpenAI Batch API...")
            batch_tasks = []
            chunk_id_map = {}  # Map custom_id to chunk_data for later matching
            for idx, chunk_data in enumerate(chunks_data):
                # Find similar chunks from good calls
                similar_chunks = self.pinecone_manager.find_similar_calls(
                    chunk_data['chunk_text'],
                    top_k=top_k
                )
                # Build prompt
                from .analysis import build_chunk_analysis_prompt, calculate_prompt_tokens
                prompt = build_chunk_analysis_prompt(
                    chunk_data['chunk_text'],
                    similar_chunks,
                    chunk_data['context_prev'],
                    chunk_data['context_next'],
                    business_rules=business_rules
                )
                # Dynamically calculate allowed_max_tokens as in non-batch mode
                prompt_tokens = calculate_prompt_tokens(prompt)
                allowed_max_tokens = min(4000, 8192 - prompt_tokens - 128)  # MAX_RESPONSE_TOKENS, CONTEXT_WINDOW, SAFETY_BUFFER
                allowed_max_tokens = max(256, allowed_max_tokens)
                if allowed_max_tokens < 256:
                    print(f"[Batch] Skipping chunk {idx+1}: prompt too long for model context window.")
                    continue
                custom_id = f"chunk-{idx+1}"
                chunk_id_map[custom_id] = {
                    'chunk_number': chunk_data['chunk_number'],
                    'total_chunks': chunk_data['total_chunks'],
                    'chunk_text_preview': chunk_data['chunk_text'][:200] + "..." if len(chunk_data['chunk_text']) > 200 else chunk_data['chunk_text'],
                }
                # Prepare batch task
                batch_tasks.append({
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4",
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": allowed_max_tokens
                    }
                })
            # Write batch file
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as batch_file:
                for task in batch_tasks:
                    batch_file.write(json.dumps(task) + "\n")
                batch_file_path = batch_file.name
            print(f"[Evaluator] Batch file written: {batch_file_path}")
            # Upload batch file
            from openai import OpenAI
            client = self.openai_client
            batch_file_obj = client.files.create(
                file=open(batch_file_path, "rb"),
                purpose="batch"
            )
            print(f"[Evaluator] Batch file uploaded: {batch_file_obj.id}")
            # Create batch job
            batch_job = client.batches.create(
                input_file_id=batch_file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            print(f"[Evaluator] Batch job created: {batch_job.id}")
            # Poll for completion
            max_wait_seconds = 2 * 60 * 60  # 2 hours
            poll_interval = 10  # seconds
            start_time = time.time()
            while True:
                job_status = client.batches.retrieve(batch_job.id)
                print(f"Batch job status: {job_status.status}")
                if job_status.status == "completed":
                    break
                elif job_status.status in ("failed", "expired", "cancelled"):
                    raise Exception(f"Batch job failed: {job_status.status}")
                if time.time() - start_time > max_wait_seconds:
                    raise Exception("Batch job polling timed out after 2 hours.")
                time.sleep(poll_interval)
            # Download results
            result_file_id = job_status.output_file_id
            retries = 0
            max_retries = 10
            while not result_file_id and retries < max_retries:
                print("[Evaluator] Waiting for output_file_id to become available...")
                time.sleep(10)
                job_status = client.batches.retrieve(batch_job.id)
                result_file_id = job_status.output_file_id
                retries += 1
            if not result_file_id:
                raise Exception("Batch job completed but no output_file_id found. Cannot download results.")
            result_content = client.files.content(result_file_id).content
            import tempfile
            with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".jsonl") as result_file:
                result_file.write(result_content)
                result_file_path = result_file.name
            print(f"[Evaluator] Batch results downloaded: {result_file_path}")
            # Parse results
            chunk_analyses = []
            with open(result_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    res = json.loads(line.strip())
                    custom_id = res["custom_id"]
                    response_body = res["response"]["body"]
                    # The LLM output is in response_body["choices"][0]["message"]["content"]
                    try:
                        analysis = json.loads(response_body["choices"][0]["message"]["content"])
                    except Exception as e:
                        analysis = {"error": f"Failed to parse LLM output: {e}", "raw": response_body}
                    chunk_analyses.append({
                        **chunk_id_map[custom_id],
                        'analysis': analysis
                    })
            # Aggregate all chunk analyses into final report
            print("[Evaluator] Aggregating chunk analyses into final report...")
            final_analysis = aggregate_chunk_analyses([c['analysis'] for c in chunk_analyses], business_rules=business_rules)

            # Clean the final report with AI
            final_analysis = clean_final_report_with_ai(final_analysis)

            # --- POST-PROCESSING FOR CLIENT CONCISENESS ---
            # (Removed: clean_bullet, is_redundant_negative, and related cleaning logic)
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
            print(f"[Evaluator] Batch API error: {e}")
            print("[Evaluator] Falling back to per-chunk analysis...")
            # --- FALLBACK: Old per-chunk method ---
            try:
                # Chunk and embed the transcript
                chunks_data = embed_new_transcript(transcript)
                total_chunks = len(chunks_data)
                business_rules = self.db_manager.get_business_rules()
                chunk_analyses = []
                for idx, chunk_data in enumerate(chunks_data):
                    similar_chunks = self.pinecone_manager.find_similar_calls(
                        chunk_data['chunk_text'],
                        top_k=top_k
                    )
                    analysis = analyze_chunk_with_rag(
                        chunk_data['chunk_text'],
                        similar_chunks,
                        chunk_data['context_prev'],
                        chunk_data['context_next'],
                        business_rules=business_rules
                    )
                    chunk_analyses.append({
                        'chunk_number': chunk_data['chunk_number'],
                        'total_chunks': chunk_data['total_chunks'],
                        'chunk_text_preview': chunk_data['chunk_text'][:200] + "..." if len(chunk_data['chunk_text']) > 200 else chunk_data['chunk_text'],
                        'analysis': analysis
                    })
                final_analysis = aggregate_chunk_analyses([c['analysis'] for c in chunk_analyses], business_rules=business_rules)
                # Clean the final report with AI
                final_analysis = clean_final_report_with_ai(final_analysis)
                metadata = {
                    'total_chunks': total_chunks,
                    'references_per_chunk': top_k,
                    'total_reference_files_used': 0,
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'transcript_length': len(transcript),
                    'estimated_call_duration': f"{total_chunks * 2-3} minutes"
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
            except Exception as e2:
                print(f"[Evaluator] Error: {e2}")
                return {
                    'error': str(e2),
                    'status': 'failed',
                    'metadata': {
                        'evaluation_timestamp': datetime.now().isoformat(),
                        'error_type': type(e2).__name__
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


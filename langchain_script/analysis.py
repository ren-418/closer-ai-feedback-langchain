import os
from typing import List, Dict, Any
from openai import OpenAI
import json
from embeddings.pinecone_store import PineconeManager
import tiktoken
from datetime import datetime

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize Pinecone manager
pinecone_manager = PineconeManager()

MAX_CHUNK_TOKENS = 3000
CHUNK_OVERLAP_TOKENS = 300
CONTEXT_WINDOW_TOKENS = 300  # Reduced for safety
MAX_REF_CHUNKS = 3  # Increased for better comparison
MAX_REF_TOKENS = 600  # Increased for more context
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
    
    # Safety check for very long texts
    if text_length > 50000:  # ~12,500 words
        print(f"[Warning] Very long transcript detected: {text_length} tokens. This may take a while to process.")
    
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
    """Truncate reference text to fit within token limits."""
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        print(f"[Token Management] Truncated reference from {len(encoding.encode(text))} to {len(tokens)} tokens")
    return encoding.decode(tokens)

def calculate_prompt_tokens(prompt: str) -> int:
    """Calculate token count for a prompt."""
    return len(encoding.encode(prompt))

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
        context = get_context_window(chunk_token_lists, idx)
        embedding = pinecone_manager.generate_embedding(chunk)
        results.append({
            'chunk_text': chunk,
            'embedding': embedding,
            'chunk_number': idx + 1,
            'total_chunks': total_chunks,
            'chunk_length': len(chunk),
            'chunk_tokens': len(chunk_tokens),
            'context_prev': context['prev'],
            'context_next': context['next']
        })
        if (idx + 1) % 10 == 0 or (idx + 1) == total_chunks:
            print(f"[Embedding] Processed {idx + 1}/{total_chunks} chunks")
    return results

def build_chunk_analysis_prompt(chunk_text: str, reference_texts: List[Dict], context_prev: str = '', context_next: str = '') -> str:
    """
    Build a professional prompt for analyzing a chunk with reference examples and context window.
    Includes token safety checks.
    """
    # Base prompt structure
    base_prompt = (
        "You are an expert sales call evaluator with 15+ years of experience in sales training and coaching. "
        "Analyze this sales call chunk by comparing it to reference examples from successful calls.\n\n"
    )
    
    # Add context sections
    context_sections = []
    if context_prev:
        context_sections.append(f"PREVIOUS CONTEXT:\n{context_prev}\n")
    if context_next:
        context_sections.append(f"NEXT CONTEXT:\n{context_next}\n")
    
    # Current chunk
    current_chunk = f"CURRENT CHUNK TO ANALYZE:\n```\n{chunk_text}\n```\n\n"
    
    # Reference examples with token management
    reference_section = "REFERENCE EXAMPLES FROM SUCCESSFUL CALLS:\n"
    ref_count = 0
    
    for i, ref in enumerate(reference_texts[:MAX_REF_CHUNKS], 1):
        ref_text = truncate_reference_chunk(ref['metadata']['transcript'], MAX_REF_TOKENS)
        ref_filename = ref['metadata'].get('filename', f'Reference {i}')
        ref_closer = ref['metadata'].get('closer_name', 'Unknown Closer')
        similarity_score = round(ref['score'], 3)
        
        ref_content = (
            f"\n--- REFERENCE {i} (Similarity: {similarity_score}) ---\n"
            f"File: {ref_filename}\n"
            f"Closer: {ref_closer}\n"
            f"Example:\n```\n{ref_text}\n```\n"
        )
        
        # Check if adding this reference would exceed limits
        test_prompt = base_prompt + "".join(context_sections) + current_chunk + reference_section + ref_content
        if calculate_prompt_tokens(test_prompt) < MAX_TOTAL_PROMPT_TOKENS - 2000:  # Leave room for instructions
            reference_section += ref_content
            ref_count += 1
        else:
            print(f"[Token Management] Skipping reference {i} to stay within token limits")
            break
    
    # Analysis instructions
    instructions = (
        "\nPROFESSIONAL ANALYSIS REQUIREMENTS:\n"
        "1. **Lead Questions**: Extract all questions asked by the lead (be specific)\n"
        "2. **Objections Identified**: List all objections that emerged during this chunk\n"
        "3. **Objection Handling Assessment**: \n"
        "   - How effectively did the closer address each objection?\n"
        "   - What techniques were used? (mirroring, reframing, etc.)\n"
        "   - Compare to reference examples - what worked well?\n"
        "4. **Engagement & Rapport**: Evaluate the closer's ability to build trust and maintain engagement\n"
        "5. **Discovery & Qualification**: Assess how well the closer gathered information and qualified the lead\n"
        "6. **Payment Discussion**: Evaluate if and how payment options were presented\n"
        "\n"
        "PROVIDE DETAILED FEEDBACK WITH SPECIFIC EXAMPLES:\n"
        "- Reference specific moments from the chunk\n"
        "- Compare to successful techniques from reference examples\n"
        "- Give concrete, actionable coaching advice\n"
        "- Focus on both strengths and areas for improvement\n"
        "\n"
        "Respond in this EXACT JSON format:\n"
        "{\n"
        '  "analysis_metadata": {\n'
        '    "chunk_number": 1,\n'
        '    "analysis_timestamp": "2024-01-01T12:00:00Z",\n'
        '    "reference_files_used": [\n'
        '      {"filename": "ref1.txt", "closer_name": "John Doe", "similarity_score": 0.85}\n'
        '    ]\n'
        '  },\n'
        '  "lead_interaction": {\n'
        '    "questions_asked": ["specific question 1", "specific question 2"],\n'
        '    "objections_raised": ["specific objection 1", "specific objection 2"],\n'
        '    "engagement_level": "high/medium/low",\n'
        '    "concerns_expressed": ["concern 1", "concern 2"]\n'
        '  },\n'
        '  "closer_performance": {\n'
        '    "strengths": [\n'
        '      {\n'
        '        "category": "objection_handling/rapport_building/discovery/closing",\n'
        '        "description": "Specific strength with example from transcript",\n'
        '        "reference_comparison": "How this compares to successful examples"\n'
        '      }\n'
        '    ],\n'
        '    "weaknesses": [\n'
        '      {\n'
        '        "category": "objection_handling/rapport_building/discovery/closing",\n'
        '        "description": "Specific weakness with example from transcript",\n'
        '        "reference_comparison": "How this differs from successful examples",\n'
        '        "improvement_suggestion": "Specific coaching advice"\n'
        '      }\n'
        '    ]\n'
        '  },\n'
        '  "coaching_recommendations": [\n'
        '    {\n'
        '      "priority": "high/medium/low",\n'
        '      "area": "objection_handling/rapport_building/discovery/closing",\n'
        '      "recommendation": "Specific, actionable coaching advice",\n'
        '      "example_from_reference": "How successful closers handle this"\n'
        '    }\n'
        '  ],\n'
        '  "scoring": {\n'
        '    "overall_score": 85,\n'
        '    "letter_grade": "A",\n'
        '    "detailed_metrics": {\n'
        '      "rapport_building": {"score": 8, "comment": "Specific feedback"},\n'
        '      "discovery": {"score": 7, "comment": "Specific feedback"},\n'
        '      "objection_handling": {"score": 9, "comment": "Specific feedback"},\n'
        '      "pitch_delivery": {"score": 8, "comment": "Specific feedback"},\n'
        '      "closing_effectiveness": {"score": 7, "comment": "Specific feedback"}\n'
        '    }\n'
        '  },\n'
        '  "key_insights": {\n'
        '    "best_practices_demonstrated": ["practice 1", "practice 2"],\n'
        '    "missed_opportunities": ["opportunity 1", "opportunity 2"],\n'
        '    "critical_moments": ["moment 1", "moment 2"]\n'
        '  }\n'
        "}"
    )
    
    # Build final prompt
    prompt = base_prompt + "".join(context_sections) + current_chunk + reference_section + instructions
    
    # Final token check and truncation if needed
    prompt_tokens = calculate_prompt_tokens(prompt)
    if prompt_tokens > MAX_TOTAL_PROMPT_TOKENS:
        print(f"[Token Management] Prompt too long ({prompt_tokens} tokens), truncating...")
        # Truncate chunk text if needed
        chunk_tokens = encoding.encode(chunk_text)
        max_chunk_tokens_for_prompt = MAX_TOTAL_PROMPT_TOKENS - 3000  # Leave room for other content
        if len(chunk_tokens) > max_chunk_tokens_for_prompt:
            chunk_tokens = chunk_tokens[:max_chunk_tokens_for_prompt]
            chunk_text = encoding.decode(chunk_tokens)
            print(f"[Token Management] Truncated chunk to {len(chunk_tokens)} tokens")
            # Rebuild prompt with truncated chunk
            prompt = build_chunk_analysis_prompt(chunk_text, reference_texts, context_prev, context_next)
    
    print(f"[Token Management] Final prompt: {calculate_prompt_tokens(prompt)} tokens")
    return prompt

def analyze_chunk_with_rag(chunk_text: str, reference_chunks: List[Dict], context_prev: str = '', context_next: str = '', temperature: float = 0.3) -> Dict:
    """
    Analyze a chunk using RAG with reference examples from good calls and context window.
    Returns enhanced analysis with reference file tracking and token safety.
    """
    try:
        prompt = build_chunk_analysis_prompt(chunk_text, reference_chunks, context_prev, context_next)
        
        # Final safety check
        prompt_tokens = calculate_prompt_tokens(prompt)
        if prompt_tokens > MAX_TOTAL_PROMPT_TOKENS:
            print(f"[Warning] Prompt still too long ({prompt_tokens} tokens), using fallback analysis")
            return {
                "error": f"Prompt too long ({prompt_tokens} tokens) for analysis",
                "analysis_metadata": {
                    "reference_files_used": [],
                    "analysis_timestamp": datetime.now().isoformat(),
                    "token_count": prompt_tokens
                }
            }
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=MAX_RESPONSE_TOKENS
        )
        
        analysis_result = json.loads(response.choices[0].message.content)
        
        # Add reference file tracking
        reference_files_used = []
        for ref in reference_chunks:
            reference_files_used.append({
                'filename': ref['metadata'].get('filename', 'Unknown'),
                'closer_name': ref['metadata'].get('closer_name', 'Unknown'),
                'similarity_score': round(ref['score'], 3),
                'date': ref['metadata'].get('date', 'Unknown')
            })
        
        # Ensure analysis_metadata exists and add reference files
        if 'analysis_metadata' not in analysis_result:
            analysis_result['analysis_metadata'] = {}
        
        analysis_result['analysis_metadata']['reference_files_used'] = reference_files_used
        analysis_result['analysis_metadata']['analysis_timestamp'] = datetime.now().isoformat()
        analysis_result['analysis_metadata']['token_count'] = prompt_tokens
        
        return analysis_result
        
    except json.JSONDecodeError as e:
        print(f"[Error] JSON decode error: {e}")
        return {
            "error": "Failed to parse LLM response as JSON",
            "raw_response": response.choices[0].message.content if 'response' in locals() else "No response",
            "analysis_metadata": {
                "reference_files_used": [],
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        print(f"[Error] Analysis failed: {e}")
        return {
            "error": f"Analysis failed: {str(e)}",
            "analysis_metadata": {
                "reference_files_used": [],
                "analysis_timestamp": datetime.now().isoformat()
            }
        }

def aggregate_chunk_analyses(chunk_analyses: List[Dict]) -> Dict:
    """
    Aggregate all chunk-level analyses into a comprehensive evaluation report.
    Returns a professional summary with reference file tracking and token safety.
    """
    # Collect all reference files used across chunks
    all_reference_files = set()
    for analysis in chunk_analyses:
        if 'analysis_metadata' in analysis and 'reference_files_used' in analysis['analysis_metadata']:
            for ref in analysis['analysis_metadata']['reference_files_used']:
                all_reference_files.add(f"{ref['filename']} ({ref['closer_name']})")
    
    # Prepare chunk summaries for aggregation
    chunk_summaries = []
    for i, analysis in enumerate(chunk_analyses):
        if 'error' not in analysis:
            summary = {
                'chunk_number': i + 1,
                'overall_score': analysis.get('scoring', {}).get('overall_score', 0),
                'letter_grade': analysis.get('scoring', {}).get('letter_grade', 'C'),
                'key_strengths': [s.get('description', '') for s in analysis.get('closer_performance', {}).get('strengths', [])],
                'key_weaknesses': [w.get('description', '') for w in analysis.get('closer_performance', {}).get('weaknesses', [])],
                'objections': analysis.get('lead_interaction', {}).get('objections_raised', []),
                'questions': analysis.get('lead_interaction', {}).get('questions_asked', [])
            }
            chunk_summaries.append(summary)
    
    # Build aggregation prompt with token safety
    prompt = (
        "You are an expert sales call evaluator creating a comprehensive final report. "
        "Based on the following chunk-level analyses, provide a professional evaluation summary.\n\n"
        "CHUNK ANALYSES SUMMARY:\n"
        f"{json.dumps(chunk_summaries, indent=2)}\n\n"
        "REFERENCE FILES USED:\n"
        f"{', '.join(all_reference_files)}\n\n"
        "Create a comprehensive final report that includes:\n"
        "1. **Executive Summary**: Overall performance assessment\n"
        "2. **Call Performance Analysis**: Detailed breakdown of strengths and weaknesses\n"
        "3. **Objection Handling Review**: How well objections were managed throughout the call\n"
        "4. **Engagement & Rapport Assessment**: Overall relationship building effectiveness\n"
        "5. **Discovery & Qualification**: How well the closer gathered information\n"
        "6. **Closing Effectiveness**: Assessment of closing techniques and results\n"
        "7. **Coaching Recommendations**: Priority-based improvement suggestions\n"
        "8. **Reference Comparisons**: How this call compares to successful examples\n"
        "\n"
        "Respond in this EXACT JSON format:\n"
        "{\n"
        '  "report_metadata": {\n'
        '    "total_chunks_analyzed": 5,\n'
        '    "reference_files_used": ["file1.txt (John Doe)", "file2.txt (Jane Smith)"],\n'
        '    "analysis_timestamp": "2024-01-01T12:00:00Z",\n'
        '    "call_duration_estimated": "15 minutes"\n'
        '  },\n'
        '  "executive_summary": {\n'
        '    "overall_assessment": "Professional summary of call performance",\n'
        '    "overall_score": 85,\n'
        '    "letter_grade": "A",\n'
        '    "key_highlights": ["highlight 1", "highlight 2"],\n'
        '    "critical_areas": ["area 1", "area 2"]\n'
        '  },\n'
        '  "detailed_analysis": {\n'
        '    "objection_handling": {\n'
        '      "score": 8,\n'
        '      "strengths": ["strength 1", "strength 2"],\n'
        '      "weaknesses": ["weakness 1", "weakness 2"],\n'
        '      "objections_encountered": ["objection 1", "objection 2"],\n'
        '      "handling_techniques_used": ["technique 1", "technique 2"]\n'
        '    },\n'
        '    "engagement_rapport": {\n'
        '      "score": 9,\n'
        '      "strengths": ["strength 1", "strength 2"],\n'
        '      "weaknesses": ["weakness 1", "weakness 2"],\n'
        '      "rapport_building_moments": ["moment 1", "moment 2"]\n'
        '    },\n'
        '    "discovery_qualification": {\n'
        '      "score": 7,\n'
        '      "strengths": ["strength 1", "strength 2"],\n'
        '      "weaknesses": ["weakness 1", "weakness 2"],\n'
        '      "information_gathered": ["info 1", "info 2"],\n'
        '      "qualification_questions": ["question 1", "question 2"]\n'
        '    },\n'
        '    "closing_effectiveness": {\n'
        '      "score": 8,\n'
        '      "strengths": ["strength 1", "strength 2"],\n'
        '      "weaknesses": ["weakness 1", "weakness 2"],\n'
        '      "closing_attempts": ["attempt 1", "attempt 2"],\n'
        '      "payment_discussion": "How payment was discussed"\n'
        '    }\n'
        '  },\n'
        '  "coaching_recommendations": [\n'
        '    {\n'
        '      "priority": "high/medium/low",\n'
        '      "category": "objection_handling/engagement/discovery/closing",\n'
        '      "recommendation": "Specific coaching advice",\n'
        '      "reference_example": "How successful closers handle this",\n'
        '      "expected_impact": "What improvement this will bring"\n'
        '    }\n'
        '  ],\n'
        '  "reference_comparisons": {\n'
        '    "similarities_to_successful_calls": ["similarity 1", "similarity 2"],\n'
        '    "differences_from_successful_calls": ["difference 1", "difference 2"],\n'
        '    "best_practices_demonstrated": ["practice 1", "practice 2"],\n'
        '    "missed_opportunities": ["opportunity 1", "opportunity 2"]\n'
        '  },\n'
        '  "lead_interaction_summary": {\n'
        '    "total_questions_asked": 5,\n'
        '    "total_objections_raised": 3,\n'
        '    "engagement_pattern": "high/medium/low",\n'
        '    "buying_signals": ["signal 1", "signal 2"],\n'
        '    "concerns_expressed": ["concern 1", "concern 2"]\n'
        '  },\n'
        '  "performance_metrics": {\n'
        '    "rapport_building": 8,\n'
        '    "discovery": 7,\n'
        '    "objection_handling": 8,\n'
        '    "pitch_delivery": 8,\n'
        '    "closing_effectiveness": 8,\n'
        '    "overall_performance": 8\n'
        '  }\n'
        "}"
    )
    
    # Check token count for aggregation prompt
    prompt_tokens = calculate_prompt_tokens(prompt)
    if prompt_tokens > MAX_TOTAL_PROMPT_TOKENS:
        print(f"[Warning] Aggregation prompt too long ({prompt_tokens} tokens), truncating chunk summaries")
        # Truncate chunk summaries if needed
        for summary in chunk_summaries:
            summary['key_strengths'] = summary['key_strengths'][:2]  # Keep only first 2
            summary['key_weaknesses'] = summary['key_weaknesses'][:2]  # Keep only first 2
        # Rebuild prompt with truncated summaries
        prompt = (
            "You are an expert sales call evaluator creating a comprehensive final report. "
            "Based on the following chunk-level analyses, provide a professional evaluation summary.\n\n"
            "CHUNK ANALYSES SUMMARY:\n"
            f"{json.dumps(chunk_summaries, indent=2)}\n\n"
            "REFERENCE FILES USED:\n"
            f"{', '.join(all_reference_files)}\n\n"
            "Create a comprehensive final report. Respond in JSON format as specified above."
        )
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=MAX_RESPONSE_TOKENS
        )
        
        final_report = json.loads(response.choices[0].message.content)
        
        # Ensure report_metadata exists and add reference files
        if 'report_metadata' not in final_report:
            final_report['report_metadata'] = {}
        
        final_report['report_metadata']['reference_files_used'] = list(all_reference_files)
        final_report['report_metadata']['analysis_timestamp'] = datetime.now().isoformat()
        final_report['report_metadata']['total_chunks_analyzed'] = len(chunk_analyses)
        
        return final_report
        
    except json.JSONDecodeError:
        return {
            "error": "Failed to parse LLM response as JSON",
            "raw_response": response.choices[0].message.content if 'response' in locals() else "No response",
            "report_metadata": {
                "reference_files_used": list(all_reference_files),
                "analysis_timestamp": datetime.now().isoformat(),
                "total_chunks_analyzed": len(chunk_analyses)
            }
        }
    except Exception as e:
        return {
            "error": f"Aggregation failed: {str(e)}",
            "report_metadata": {
                "reference_files_used": list(all_reference_files),
                "analysis_timestamp": datetime.now().isoformat(),
                "total_chunks_analyzed": len(chunk_analyses)
            }
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
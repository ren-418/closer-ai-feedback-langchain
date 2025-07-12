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

MAX_CHUNK_TOKENS = 1500
CHUNK_OVERLAP_TOKENS = 200
CONTEXT_WINDOW_TOKENS = 300  # Reduced for safety
MAX_REF_CHUNKS = 3
MAX_REF_TOKENS = 600
MAX_TOTAL_PROMPT_TOKENS = 8000
MAX_RESPONSE_TOKENS = 4000  # Increased from 2500 to 4000 for longer responses
  # Increased for longer aggregation responses

# Use OpenAI's tiktoken for accurate token counting
encoding = tiktoken.encoding_for_model("gpt-4")

CONTEXT_WINDOW = 8192
SAFETY_BUFFER = 128  # For OpenAI chat message overhead

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

def build_chunk_analysis_prompt(chunk_text: str, reference_texts: List[Dict], context_prev: str = '', context_next: str = '', business_rules: List[Dict] = None) -> str:
    """
    Build a professional prompt for analyzing a chunk with reference examples and context window.
    Includes token safety checks and explicit instructions for lead question extraction.
    """
   
    # Base prompt structure
    base_prompt = (
        "You are an expert sales call evaluator with 15+ years of experience in sales training and coaching. "
        "Analyze this sales call chunk by comparing it to reference examples from successful calls.\n\n"
        "IMPORTANT: In the transcript, the 'lead' is the prospect/customer, and the 'closer' is the sales representative. "
        "Only extract questions that are asked by the lead. Do NOT include any questions asked by the closer.\n"
        "If the transcript uses names, use the context to determine which speaker is the lead.\n\n"
        "Example:\n"
        "Transcript:\n"
        "Closer: How are you today?\n"
        "Lead: I'm good, thanks. Can you tell me more about your product?\n"
        "Closer: Sure! What are your main challenges?\n"
        "Lead: How much does it cost?\n\n"
        "Extracted lead questions:\n"
        "- Can you tell me more about your product?\n"
        "- How much does it cost?\n\n"
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
        ref_label = ref['metadata'].get('label', f'Reference {i}')
        ref_closer = ref['metadata'].get('closer_name', 'Unknown Closer')
        similarity_score = round(ref['score'], 3)
        
        ref_content = (
            f"\n--- REFERENCE {i} (Similarity: {similarity_score}) ---\n"
            f"File: {ref_label}\n"
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
    
    # Business rules section
    if business_rules and len(business_rules) > 0:
        rules_text = format_rules(business_rules)
        rules_section = (
            f"\nBUSINESS RULES TO CHECK (STRICTLY ENFORCE THESE ONLY):\n{rules_text}\n\n"
            "For each violation found (based ONLY on the above business rules):\n"
            "- Note the exact text and context where it appears\n"
            "- Suggest the correct term to use\n"
            "- Explain why it's a violation and its business impact\n"
            "- Indicate score penalty (typically -2 points per violation)\n"
        )
    else:
        rules_section = (
            "\nNO BUSINESS RULES are provided for this analysis. Do NOT invent or check for any business rule violations.\n"
        )
    
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
        "7. **Business Rules Violations**: Check for any violations in this chunk\n"
        "\n"
        "***CRITICAL INSTRUCTION:*** All feedback, especially for weaknesses and coaching recommendations, must be extremely specific, detailed, and actionable.\n"
        "- Reference the exact lines or phrases from the transcript for every point.\n"
        "- For every weakness, enumerate precisely what was missing, unclear, or insufficient, and explain why.\n"
        "- List the actual questions the closer asked that were weak or insufficient, and explain why they were not effective.\n"
        "- Suggest specific, context-aware questions the closer could have asked instead, tailored to the lead's situation.\n"
        "- If a solution or alternative was missing, describe in detail what solution(s) could have been offered, and how they should be presented to the lead.\n"
        "- For each coaching recommendation, provide a step-by-step, concrete example of how to implement it in a real conversation.\n"
        "- Identify any missed opportunities and specify exactly what should have been done differently, with sample dialogue.\n"
        "- ***Generic, vague, or high-level feedback is unacceptable and will be rejected.***\n"
        "- All feedback must be actionable and directly tied to the transcript content.\n"
        "\n"
        "EXAMPLES OF FEEDBACK (Unacceptable vs. Acceptable):\n"
        "Unacceptable: 'Could have probed deeper into the lead's needs.'\n"
        "Acceptable: 'The closer failed to ask about the lead's budget after the lead mentioned cost concerns (\"I'm not sure if this fits our budget\"). Instead, the closer should have asked: \"What budget range are you working with for this project?\" and \"Are there any financial constraints we should be aware of?\"'\n"
        "\n"
        "Unacceptable: 'Should provide clear solutions.'\n"
        "Acceptable: 'When the lead expressed concern about onboarding (\"I'm worried about how long it will take to get started\"), the closer did not address this. The closer should have responded: \"We offer a dedicated onboarding specialist who will guide you step-by-step, and most clients are fully set up within two weeks. Would you like to hear how this worked for a similar client?\"'\n"
        "\n"
        "Unacceptable: 'Could have handled objections better.'\n"
        "Acceptable: 'When the lead said, \"I'm not sure your solution integrates with our CRM,\" the closer only replied, \"We have many integrations.\" Instead, the closer should have asked, \"Which CRM are you using?\" and then provided a specific example: \"We recently helped a client with Salesforce integration—here's how we did it.\"'\n"
        "\n"
        "Unacceptable: 'Should ask more discovery questions.'\n"
        "Acceptable: 'The closer missed an opportunity to ask about the lead's current workflow after the lead described their manual process. The closer should have asked: \"Can you walk me through your current process step by step?\" and \"What are the biggest bottlenecks you face day-to-day?\"'\n"
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
        '  "custom_business_rules": {\n'
        '    "violations_found": [\n'
        '      {\n'
        '        "rule": "rule_name",\n'
        '        "violation_text": "problematic phrase",\n'
        '        "context": "The full sentence or context",\n'
        '        "correct_text": "preferred phrase",\n'
        '        "explanation": "Why this is a problem",\n'
        '        "score_impact": -2\n'
        '      }\n'
        '    ],\n'
        '    "total_violations": 1,\n'
        '    "total_score_penalty": -2\n'
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
        "}\n"
        "\nSCORING GUIDELINES:\n"
        "Only give a high score if there is a clear, strong reason. If there are significant issues or violations, do not hesitate to give a low score. Be strict and fair: reward excellence, penalize serious mistakes.\n"
        "\nGRADE RULES (for letter_grade):\n"
        "94-100  = A\n"
        "90-93.9 = A-\n"
        "87-89.9 = B+\n"
        "84-86.9 = B\n"
        "80-83.9 = B-\n"
        "77-79.9 = C+\n"
        "74-76.9 = C\n"
        "70-73.9 = C-\n"
        "67-69.9 = D+\n"
        "64-66.9 = D\n"
        "60-63.9 = D-\n"
        "0-59.9  = E."
    )
    
    # Build final prompt
    prompt = base_prompt + "".join(context_sections) + current_chunk + reference_section + rules_section + instructions
    # Final token check and summarization if needed
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
            prompt = build_chunk_analysis_prompt(chunk_text, reference_texts, context_prev, context_next, business_rules)
    
    print(f"[Token Management] Final prompt: {calculate_prompt_tokens(prompt)} tokens")
    return prompt

def analyze_chunk_with_rag(chunk_text: str, reference_chunks: List[Dict], context_prev: str = '', context_next: str = '', temperature: float = 0.3, business_rules: List[Dict] = None, chunk_number: int = None) -> Dict:
    """
    Analyze a chunk using RAG with reference examples from good calls and context window.
    Returns enhanced analysis with reference file tracking and token safety.
    Dynamically sets max_tokens to avoid context window errors, with a safety buffer.
    """
    try:
        prompt = build_chunk_analysis_prompt(chunk_text, reference_chunks, context_prev, context_next, business_rules)

        # Save the prompt as a JSON file for inspection
        prompt_save = {
            "chunk_number": chunk_number,
            "prompt": prompt,
            "context_prev": context_prev,
            "context_next": context_next,
            "reference_chunks_count": len(reference_chunks),
            "business_rules_count": len(business_rules) if business_rules else 0
        }
        fname = f"chunk_prompt_{chunk_number if chunk_number is not None else 'unknown'}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(prompt_save, f, ensure_ascii=False, indent=2)

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
        # Dynamically set max_tokens with buffer
        allowed_max_tokens = min(MAX_RESPONSE_TOKENS, CONTEXT_WINDOW - prompt_tokens - SAFETY_BUFFER)
        allowed_max_tokens = max(256, allowed_max_tokens)

        if allowed_max_tokens < MAX_RESPONSE_TOKENS:
            print(f"[Token Management] Reducing max_tokens from {MAX_RESPONSE_TOKENS} to {allowed_max_tokens} to fit context window (with buffer).")
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=allowed_max_tokens
        )
        analysis_result = json.loads(response.choices[0].message.content)
        # Add reference file tracking
        reference_files_used = []
        for ref in reference_chunks:
            reference_files_used.append({
                'filename': ref['metadata'].get('label', 'Unknown'),
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
        analysis_result['analysis_metadata']['max_tokens_used'] = allowed_max_tokens
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

def aggregate_chunk_analyses(chunk_analyses: List[Dict], business_rules: List[Dict] = None) -> Dict:
    """
    Aggregate all chunk-level analyses into a comprehensive evaluation report.
    Uses the detailed chunk analysis results directly to preserve specificity.
    """
    # Collect all reference files used across chunks
    all_reference_files = set()
    for analysis in chunk_analyses:
        if 'analysis_metadata' in analysis and 'reference_files_used' in analysis['analysis_metadata']:
            for ref in analysis['analysis_metadata']['reference_files_used']:
                all_reference_files.add(f"{ref['filename']} ({ref['closer_name']})")
    
    # Collect all violations from chunks
    all_violations = []
    total_violations = 0
    total_score_penalty = 0
    
    for i, analysis in enumerate(chunk_analyses):
        if 'error' not in analysis and 'custom_business_rules' in analysis:
            chunk_violations = analysis['custom_business_rules'].get('violations_found', [])
            for violation in chunk_violations:
                # Add chunk context to violation
                violation['chunk_number'] = i + 1
                all_violations.append(violation)
            total_violations += analysis['custom_business_rules'].get('total_violations', 0)
            total_score_penalty += analysis['custom_business_rules'].get('total_score_penalty', 0)
    
    # Extract all detailed information from chunk analyses
    all_strengths = []
    all_weaknesses = []
    all_coaching_recommendations = []
    all_lead_questions = []
    all_objections = []
    all_concerns = []
    all_scores = []
    
    for i, analysis in enumerate(chunk_analyses):
        if 'error' not in analysis:
            chunk_num = i + 1
            
            # Extract strengths with chunk context
            if 'closer_performance' in analysis and 'strengths' in analysis['closer_performance']:
                for strength in analysis['closer_performance']['strengths']:
                    strength['chunk_number'] = chunk_num
                    all_strengths.append(strength)
            
            # Extract weaknesses with chunk context
            if 'closer_performance' in analysis and 'weaknesses' in analysis['closer_performance']:
                for weakness in analysis['closer_performance']['weaknesses']:
                    weakness['chunk_number'] = chunk_num
                    all_weaknesses.append(weakness)
            
            # Extract coaching recommendations with chunk context
            if 'coaching_recommendations' in analysis:
                for rec in analysis['coaching_recommendations']:
                    rec['chunk_number'] = chunk_num
                    all_coaching_recommendations.append(rec)
            
            # Extract lead interactions
            if 'lead_interaction' in analysis:
                lead_int = analysis['lead_interaction']
                for question in lead_int.get('questions_asked', []):
                    all_lead_questions.append(f"Chunk {chunk_num}: {question}")
                for objection in lead_int.get('objections_raised', []):
                    all_objections.append(f"Chunk {chunk_num}: {objection}")
                for concern in lead_int.get('concerns_expressed', []):
                    all_concerns.append(f"Chunk {chunk_num}: {concern}")
            
            # Extract scores
            if 'scoring' in analysis:
                score_info = {
                    'chunk_number': chunk_num,
                    'overall_score': analysis['scoring'].get('overall_score', 0),
                    'letter_grade': analysis['scoring'].get('letter_grade', 'C'),
                    'detailed_metrics': analysis['scoring'].get('detailed_metrics', {})
                }
                all_scores.append(score_info)
    
    # Calculate overall scores
    if all_scores:
        overall_score = sum(score['overall_score'] for score in all_scores) / len(all_scores)
        # Determine letter grade based on overall score
        if overall_score >= 94:
            letter_grade = "A"
        elif overall_score >= 90:
            letter_grade = "A-"
        elif overall_score >= 87:
            letter_grade = "B+"
        elif overall_score >= 84:
            letter_grade = "B"
        elif overall_score >= 80:
            letter_grade = "B-"
        elif overall_score >= 77:
            letter_grade = "C+"
        elif overall_score >= 74:
            letter_grade = "C"
        elif overall_score >= 70:
            letter_grade = "C-"
        elif overall_score >= 67:
            letter_grade = "D+"
        elif overall_score >= 64:
            letter_grade = "D"
        elif overall_score >= 60:
            letter_grade = "D-"
        else:
            letter_grade = "E"
    else:
        overall_score = 0
        letter_grade = "C"
    
    # Build detailed analysis sections
    detailed_analysis = {
        "objection_handling": {
            "score": sum(score['detailed_metrics'].get('objection_handling', {}).get('score', 0) for score in all_scores) / max(len(all_scores), 1),
            "strengths": [s['description'] for s in all_strengths if s.get('category') == 'objection_handling'],
            "weaknesses": [w['description'] for w in all_weaknesses if w.get('category') == 'objection_handling'],
            "objections_encountered": all_objections,
            "handling_techniques_used": []
        },
        "engagement_rapport": {
            "score": sum(score['detailed_metrics'].get('rapport_building', {}).get('score', 0) for score in all_scores) / max(len(all_scores), 1),
            "strengths": [s['description'] for s in all_strengths if s.get('category') == 'rapport_building'],
            "weaknesses": [w['description'] for w in all_weaknesses if w.get('category') == 'rapport_building'],
            "rapport_building_moments": []
        },
        "discovery_qualification": {
            "score": sum(score['detailed_metrics'].get('discovery', {}).get('score', 0) for score in all_scores) / max(len(all_scores), 1),
            "strengths": [s['description'] for s in all_strengths if s.get('category') == 'discovery'],
            "weaknesses": [w['description'] for w in all_weaknesses if w.get('category') == 'discovery'],
            "information_gathered": [],
            "qualification_questions": []
        },
        "closing_effectiveness": {
            "score": sum(score['detailed_metrics'].get('closing_effectiveness', {}).get('score', 0) for score in all_scores) / max(len(all_scores), 1),
            "strengths": [s['description'] for s in all_strengths if s.get('category') == 'closing'],
            "weaknesses": [w['description'] for w in all_weaknesses if w.get('category') == 'closing'],
            "closing_attempts": [],
            "payment_discussion": ""
        }
    }
    
    # Build final report using detailed chunk data
    final_report = {
        "report_metadata": {
            "total_chunks_analyzed": len(chunk_analyses),
            "reference_files_used": list(all_reference_files),
            "analysis_timestamp": datetime.now().isoformat(),
            "call_duration_estimated": f"{len(chunk_analyses) * 5} minutes"  # Rough estimate
        },
        "executive_summary": {
            "overall_assessment": f"Call analyzed across {len(chunk_analyses)} chunks with detailed performance evaluation",
            "overall_score": round(overall_score, 1),
            "letter_grade": letter_grade,
            "key_highlights": [s['description'] for s in all_strengths[:3]],  # Top 3 strengths
            "critical_areas": [w['description'] for w in all_weaknesses[:3]]  # Top 3 weaknesses
        },
        "detailed_analysis": detailed_analysis,
        "custom_business_rules": {
            "violations_found": all_violations,
            "total_violations": total_violations,
            "total_score_penalty": total_score_penalty,
            "recommendations": []
        },
        "coaching_recommendations": all_coaching_recommendations,
        "reference_comparisons": {
            "similarities_to_successful_calls": [s.get('reference_comparison', '') for s in all_strengths if s.get('reference_comparison')],
            "differences_from_successful_calls": [w.get('reference_comparison', '') for w in all_weaknesses if w.get('reference_comparison')],
            "best_practices_demonstrated": [s['description'] for s in all_strengths],
            "missed_opportunities": [w['description'] for w in all_weaknesses]
        },
        "lead_interaction_summary": {
            "total_questions_asked": len(all_lead_questions),
            "total_objections_raised": len(all_objections),
            "questions_asked": all_lead_questions,
            "engagement_pattern": "high" if len(all_strengths) > len(all_weaknesses) else "medium" if len(all_strengths) == len(all_weaknesses) else "low",
            "buying_signals": [],
            "concerns_expressed": all_concerns
        },
        "performance_metrics": {
            "rapport_building": detailed_analysis["engagement_rapport"]["score"],
            "discovery": detailed_analysis["discovery_qualification"]["score"],
            "objection_handling": detailed_analysis["objection_handling"]["score"],
            "pitch_delivery": sum(score['detailed_metrics'].get('pitch_delivery', {}).get('score', 0) for score in all_scores) / max(len(all_scores), 1),
            "closing_effectiveness": detailed_analysis["closing_effectiveness"]["score"],
            "overall_performance": overall_score
        }
    }
    
    return final_report

def format_rules(business_rules: List[Dict]) -> str:
    """Format business rules as a compact numbered list, including violation_text and correct_text."""
    lines = []
    for i, rule in enumerate(business_rules, 1):
        lines.append(
            f"{i}. {rule.get('criteria_name', 'Rule')}: {rule.get('description', '')} "
            f"(Violation: '{rule.get('violation_text', '')}'"
            + (f", Correct: '{rule.get('correct_text', '')}'" if rule.get('correct_text') else "")
            + f", Penalty: {rule.get('score_penalty', 0)})"
        )
    return '\n'.join(lines)

def summarize_rules(business_rules: List[Dict]) -> str:
    """Summarize business rules into a compact summary if prompt is too long."""
    topics = set()
    for rule in business_rules:
        topics.add(rule.get('category', 'general'))
    return f"There are {len(business_rules)} business rules covering: {', '.join(sorted(topics))}. Pay special attention to main contents just critical criteria."

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
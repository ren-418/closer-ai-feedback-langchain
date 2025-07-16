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
    # STYLE REQUIREMENTS at the very top
    

    # BUSINESS RULES section highlighted
    if business_rules and len(business_rules) > 0:
        from .analysis import format_rules
        rules_text = format_rules(business_rules)
        rules_section = (
            "BUSINESS RULES TO CHECK (STRICT, CRITICAL):\n"
            "- Carefully review the following business rules.\n"
            "- For EACH violation found (based ONLY on the rules below):\n"
            "    - Note the exact text and context where it appears.\n"
            "    - Suggest the correct term to use.\n"
            "    - Explain why it's a violation and its business impact.\n"
            "    - Indicate score penalty (see below).\n"
            f"{rules_text}\n\n"
        )
    else:
        rules_section = (
            "NO BUSINESS RULES are provided for this analysis. Do NOT invent or check for any business rule violations.\n"
        )

    # Base prompt structure (examples updated to avoid 'the closer', 'the lead', etc.)
    base_prompt = (
        "You are an expert sales call evaluator with 15+ years of experience in sales training and coaching.\n"
        "Analyze this sales call chunk by comparing it to reference examples from successful calls.\n\n"
        "IMPORTANT: In the transcript, the 'lead' is the prospect/customer, and the 'closer' is the sales representative. Only extract questions that are asked by the lead. Do NOT include any questions asked by the closer.\n"
        "If the transcript uses names, use the context to determine which speaker is the lead.\n\n"
        "Example:\n"
        "Transcript:\n"
        "Speaker 1: How are you today?\n"
        "Speaker 2: I'm good, thanks. Can you tell me more about your product?\n"
        "Speaker 1: Sure! What are your main challenges?\n"
        "Speaker 2: How much does it cost?\n\n"
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
    current_chunk = f"CURRENT CHUNK TO ANALYZE:\n" + f"""\n{chunk_text}\n"""\

    # Reference examples with token management
    reference_section = "REFERENCE EXAMPLES FROM SUCCESSFUL CALLS:\n"
    ref_count = 0
    for i, ref in enumerate(reference_texts[:3], 1):
        ref_text = ref['metadata']['transcript'][:600]  # Truncate for brevity
        ref_label = ref['metadata'].get('label', f'Reference {i}')
        similarity_score = round(ref['score'], 3)
        ref_content = (
            f"\n--- REFERENCE {i} (Similarity: {similarity_score}) ---\n"
            f"File: {ref_label}\n"
            f"Example:\n" + f"""\n{ref_text}\n"""\
        )
        test_prompt = base_prompt + "".join(context_sections) + current_chunk + reference_section + ref_content
        if len(test_prompt) < 6000:  # crude token check for this edit
            reference_section += ref_content
            ref_count += 1
        else:
            break

    # Analysis instructions (unchanged, but will follow new style)
    instructions = (
        "\nPROFESSIONAL ANALYSIS REQUIREMENTS:\n"
        "1. **Lead Questions**: Extract all questions asked by the lead (be specific)\n"
        "2. **Objections Identified**: List all objections that emerged during this chunk\n"
        "3. **Objection Handling Assessment**: \n"
        "   - How effectively were objections addressed?\n"
        "   - What techniques were used? (mirroring, reframing, etc.)\n"
        "   - Compare to reference examples - what worked well?\n"
        "4. **Engagement & Rapport**: Evaluate ability to build trust and maintain engagement\n"
        "5. **Discovery & Qualification**: Assess how well information was gathered and the lead was qualified\n"
        "6. **Payment Discussion**: Evaluate if and how payment options were presented\n"
        "7. **Business Rules Violations**: Check for any violations in this chunk\n"
        "\n"
        "***CRITICAL INSTRUCTION:*** All feedback, especially for weaknesses and coaching recommendations, must be extremely specific, detailed, and actionable.\n"
        "- Reference the exact lines or phrases from the transcript for every point.\n"
        "- For every weakness, enumerate precisely what was missing, unclear, or insufficient, and explain why.\n"
        "- List the actual questions that were weak or insufficient, and explain why they were not effective.\n"
        "- Suggest specific, context-aware questions that could have been asked instead, tailored to the lead's situation.\n"
        "- If a solution or alternative was missing, describe in detail what solution(s) could have been offered, and how they should be presented.\n"
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
        "Respond in this EXACT JSON format (note: for each item in 'detailed_analysis', include 'strengths' and 'weaknesses' arrays, and omit the 'comment' field from 'detailed_metrics'):\n"
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
        '    "concerns_expressed": ["concern 1", "concern 2"],\n'
        '    "buying_signals": ["signal 1", "signal 2"]\n'
        '  },\n'
        '  "custom_business_rules": {\n'
        '    "violations_found": [\n'
        '      {\n'
        '        "rule": "rule_name",\n'
        '        "violation_text": "problematic phrase",\n'
        '        "context": "The full sentence or context",\n'
        '        "recommendation": "preferred phrase",\n'
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
        '      "rapport_building": {"score": 8},\n'
        '      "discovery_qualification": {"score": 7},\n'
        '      "objection_handling": {"score": 9},\n'
        '      "closing_effectiveness": {"score": 7}\n'
        '    }\n'
        '  },\n'
        '  "detailed_analysis": {\n'
        '    "objection_handling": {\n'
        '      "score": 9,\n'
        '      "strengths": ["strength 1", "strength 2"],\n'
        '      "weaknesses": ["weakness 1", "weakness 2"],\n'
        '      "handling_techniques_used": ["specific technique 1", "specific technique 2"],\n'
        '      "objections_encountered": ["specific objection 1", "specific objection 2"]\n'
        '    },\n'
        '    "rapport_building": {\n'
        '      "score": 8,\n'
        '      "strengths": ["strength 1", "strength 2"],\n'
        '      "weaknesses": ["weakness 1", "weakness 2"],\n'
        '      "rapport_building_moments": ["specific moment 1", "specific moment 2"]\n'
        '    },\n'
        '    "discovery_qualification": {\n'
        '      "score": 7,\n'
        '      "strengths": ["strength 1", "strength 2"],\n'
        '      "weaknesses": ["weakness 1", "weakness 2"],\n'
        '      "information_gathered": ["specific info 1", "specific info 2"],\n'
        '      "qualification_questions": ["specific question 1", "specific question 2"]\n'
        '    },\n'
        '    "closing_effectiveness": {\n'
        '      "score": 7,\n'
        '      "strengths": ["strength 1", "strength 2"],\n'
        '      "weaknesses": ["weakness 1", "weakness 2"],\n'
        '      "closing_attempts": ["specific attempt 1", "specific attempt 2"],\n'
        '      "payment_discussion": ["specific payment discussion1","specific payment discussion2"]\n'
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

    # Final reminder at the end
    final_reminder = (
        "\nREMINDER: Follow ALL instructions above, including PROFESSIONAL ANALYSIS REQUIREMENTS, CRITICAL INSTRUCTION, business rules, scoring guidelines, and required JSON format.\n"
    )

    # Build final prompt
    prompt = base_prompt + "".join(context_sections) + current_chunk + reference_section + rules_section + instructions + final_reminder
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

def generate_overall_assessment(overall_score: float, letter_grade: str, strengths_count: int, weaknesses_count: int, total_chunks: int) -> str:
    """
    Generate a meaningful overall assessment based on performance metrics.
    """
    # Base assessment based on score
    if overall_score >= 90:
        performance_level = "excellent"
        assessment = f"Excellent performance with a {letter_grade} grade ({overall_score:.1f}/100). "
    elif overall_score >= 80:
        performance_level = "good"
        assessment = f"Good performance with a {letter_grade} grade ({overall_score:.1f}/100). "
    elif overall_score >= 70:
        performance_level = "fair"
        assessment = f"Fair performance with a {letter_grade} grade ({overall_score:.1f}/100). "
    elif overall_score >= 60:
        performance_level = "poor"
        assessment = f"Poor performance with a {letter_grade} grade ({overall_score:.1f}/100). "
    else:
        performance_level = "very_poor"
        assessment = f"Very poor performance with a {letter_grade} grade ({overall_score:.1f}/100). "
    
    # Add balance assessment with more context
    if strengths_count > weaknesses_count:
        if strengths_count >= 5:
            balance = f"The call demonstrated {strengths_count} strong techniques with only {weaknesses_count} areas for improvement. "
        else:
            balance = f"The call showed more strengths ({strengths_count}) than weaknesses ({weaknesses_count}). "
    elif weaknesses_count > strengths_count:
        if weaknesses_count >= 5:
            balance = f"The call had {weaknesses_count} areas needing attention with only {strengths_count} strengths identified. "
        else:
            balance = f"The call had more areas for improvement ({weaknesses_count}) than strengths ({strengths_count}). "
    else:
        balance = f"The call showed a balanced mix with {strengths_count} strengths and {weaknesses_count} areas for improvement. "
    
    # Add call length context with more detail
    if total_chunks <= 3:
        length_context = f"This was a relatively short call ({total_chunks} segments analyzed). "
    elif total_chunks <= 6:
        length_context = f"This was a moderate-length call ({total_chunks} segments analyzed). "
    else:
        length_context = f"This was a lengthy call with extensive interaction ({total_chunks} segments analyzed). "
    
    # Add performance context based on score ranges
    if overall_score >= 85:
        score_context = "This performance level indicates strong sales skills and effective techniques. "
    elif overall_score >= 75:
        score_context = "This performance level shows solid fundamentals with room for enhancement. "
    elif overall_score >= 65:
        score_context = "This performance level suggests basic skills with significant improvement opportunities. "
    else:
        score_context = "This performance level indicates fundamental areas need attention. "
    
    # Add specific insights based on performance and patterns
    if performance_level in ["excellent", "good"]:
        if strengths_count >= 3:
            insight = "Multiple strong techniques were demonstrated throughout the call, indicating well-rounded sales skills."
        else:
            insight = "Key strengths were identified despite some areas for improvement, showing potential for growth."
    elif performance_level in ["fair", "poor"]:
        if weaknesses_count >= 3:
            insight = "Several areas need attention to improve future performance and conversion rates."
        else:
            insight = "Targeted improvements in specific areas could significantly enhance overall effectiveness."
    else:
        insight = "Significant improvements are needed across multiple areas to reach acceptable performance levels."
    
    # Add coaching priority
    if weaknesses_count >= 5:
        coaching_priority = "High priority coaching recommended to address multiple areas."
    elif weaknesses_count >= 3:
        coaching_priority = "Moderate coaching focus needed on key improvement areas."
    elif weaknesses_count >= 1:
        coaching_priority = "Light coaching recommended to refine specific techniques."
    else:
        coaching_priority = "Minimal coaching needed - focus on maintaining current strengths."
    
    return assessment + balance + length_context + score_context + insight + " " + coaching_priority

def aggregate_chunk_analyses(chunk_analyses: List[Dict], business_rules: List[Dict] = None) -> Dict:
    """
    Aggregate all chunk-level analyses into a comprehensive evaluation report.
    Uses direct data compilation to preserve all specific details from chunks.
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
    
    # Extract ALL specific details from chunks
    all_strengths = []
    all_weaknesses = []
    all_coaching_recommendations = []
    all_lead_questions = []
    all_objections = []
    all_concerns = []
    all_buying_signals = []
    
    # Initialize detailed analysis structure
    detailed_analysis = {
        'objection_handling': {'score': 0, 'strengths': [], 'weaknesses': [], 'objections_encountered': [], 'handling_techniques_used': []},
        'rapport_building': {'score': 0, 'strengths': [], 'weaknesses': [], 'rapport_building_moments': []},
        'discovery_qualification': {'score': 0, 'strengths': [], 'weaknesses': [], 'information_gathered': [], 'qualification_questions': []},
        'closing_effectiveness': {'score': 0, 'strengths': [], 'weaknesses': [], 'closing_attempts': [], 'payment_discussion': []},
    }
    
    total_score = 0
    total_chunks_with_scores = 0
    
    for i, analysis in enumerate(chunk_analyses):
        if 'error' not in analysis:
            chunk_number = i + 1
            
            # Extract strengths with chunk context
            for strength in analysis.get('closer_performance', {}).get('strengths', []):
                if isinstance(strength, dict) and 'description' in strength:
                    strength_with_context = strength.copy()
                    strength_with_context['chunk_number'] = chunk_number
                    all_strengths.append(strength_with_context)
            
            # Extract weaknesses with chunk context
            for weakness in analysis.get('closer_performance', {}).get('weaknesses', []):
                if isinstance(weakness, dict) and 'description' in weakness:
                    weakness_with_context = weakness.copy()
                    weakness_with_context['chunk_number'] = chunk_number
                    all_weaknesses.append(weakness_with_context)
            
            # Extract coaching recommendations with chunk context
            for rec in analysis.get('coaching_recommendations', []):
                if isinstance(rec, dict) and 'recommendation' in rec:
                    rec_with_context = rec.copy()
                    rec_with_context['chunk_number'] = chunk_number
                    all_coaching_recommendations.append(rec_with_context)
            
            # Extract lead interaction details
            lead_interaction = analysis.get('lead_interaction', {})
            for question in lead_interaction.get('questions_asked', []):
                all_lead_questions.append(f"{question}")
            for objection in lead_interaction.get('objections_raised', []):
                all_objections.append(f"{objection}")
            for concern in lead_interaction.get('concerns_expressed', []):
                all_concerns.append(f"{concern}")
            for signal in lead_interaction.get('buying_signals', []):
                all_buying_signals.append(f"{signal}")
            
            # Extract detailed analysis by category
            detailed = analysis.get('detailed_analysis', {})
            
            # Objection handling
            objection_handling = detailed.get('objection_handling', {})
            for technique in objection_handling.get('handling_techniques_used', []):
                detailed_analysis['objection_handling']['handling_techniques_used'].append(f"{technique}")
            for objection in objection_handling.get('objections_encountered', []):
                detailed_analysis['objection_handling']['objections_encountered'].append(f"{objection}")
            # Add strengths/weaknesses for objection_handling
            for strength in objection_handling.get('strengths', []):
                detailed_analysis['objection_handling']['strengths'].append(strength)
            for weakness in objection_handling.get('weaknesses', []):
                detailed_analysis['objection_handling']['weaknesses'].append(weakness)
            
            # Engagement & rapport
            rapport_building = detailed.get('rapport_building', {})
            for moment in rapport_building.get('rapport_building_moments', []):
                detailed_analysis['rapport_building']['rapport_building_moments'].append(f"{moment}")
            # Add strengths/weaknesses for rapport_building
            for strength in rapport_building.get('strengths', []):
                detailed_analysis['rapport_building']['strengths'].append(strength)
            for weakness in rapport_building.get('weaknesses', []):
                detailed_analysis['rapport_building']['weaknesses'].append(weakness)
            
            # Discovery & qualification
            discovery_qualification = detailed.get('discovery_qualification', {})
            for info in discovery_qualification.get('information_gathered', []):
                detailed_analysis['discovery_qualification']['information_gathered'].append(f"{info}")
            for question in discovery_qualification.get('qualification_questions', []):
                detailed_analysis['discovery_qualification']['qualification_questions'].append(f"{question}")
            # Add strengths/weaknesses for discovery_qualification
            for strength in discovery_qualification.get('strengths', []):
                detailed_analysis['discovery_qualification']['strengths'].append(strength)
            for weakness in discovery_qualification.get('weaknesses', []):
                detailed_analysis['discovery_qualification']['weaknesses'].append(weakness)
            
            # Closing effectiveness
            closing_effectiveness = detailed.get('closing_effectiveness', {})
            for attempt in closing_effectiveness.get('closing_attempts', []):
                detailed_analysis['closing_effectiveness']['closing_attempts'].append(f"{attempt}")
            # Handle payment_discussion exactly like other array fields
            for pd_item in closing_effectiveness.get('payment_discussion', []):
                detailed_analysis['closing_effectiveness']['payment_discussion'].append(f"{pd_item}")
            # Add strengths/weaknesses for closing_effectiveness
            for strength in closing_effectiveness.get('strengths', []):
                detailed_analysis['closing_effectiveness']['strengths'].append(strength)
            for weakness in closing_effectiveness.get('weaknesses', []):
                detailed_analysis['closing_effectiveness']['weaknesses'].append(weakness)
            
            # Collect scores for averaging
            scoring = analysis.get('scoring', {})
            if 'overall_score' in scoring:
                total_score += scoring['overall_score']
                total_chunks_with_scores += 1
            
            # Collect detailed metrics
            detailed_metrics = scoring.get('detailed_metrics', {})
            for category, metric in detailed_metrics.items():
                if isinstance(metric, dict) and 'score' in metric:
                    if category in detailed_analysis:
                        detailed_analysis[category]['score'] += metric['score']
    
    # Calculate average scores
    overall_score = total_score / total_chunks_with_scores if total_chunks_with_scores > 0 else 0
    for category in detailed_analysis:
        if total_chunks_with_scores > 0:
            detailed_analysis[category]['score'] = round(detailed_analysis[category]['score'] / total_chunks_with_scores, 1)

    # Subtract business rule violation penalty from overall_score
    overall_score_penalized = overall_score + total_score_penalty
    overall_score_penalized = max(0, overall_score_penalized)  # Clamp to 0 minimum

    # Use penalized score for letter grade
    if overall_score_penalized >= 94:
        letter_grade = "A"
    elif overall_score_penalized >= 90:
        letter_grade = "A-"
    elif overall_score_penalized >= 87:
        letter_grade = "B+"
    elif overall_score_penalized >= 84:
        letter_grade = "B"
    elif overall_score_penalized >= 80:
        letter_grade = "B-"
    elif overall_score_penalized >= 77:
        letter_grade = "C+"
    elif overall_score_penalized >= 74:
        letter_grade = "C"
    elif overall_score_penalized >= 70:
        letter_grade = "C-"
    elif overall_score_penalized >= 67:
        letter_grade = "D+"
    elif overall_score_penalized >= 64:
        letter_grade = "D"
    elif overall_score_penalized >= 60:
        letter_grade = "D-"
    else:
        letter_grade = "E"

    # Calculate overall performance from detailed analysis scores for consistency
    detailed_scores = [
        detailed_analysis["rapport_building"]["score"],
        detailed_analysis["discovery_qualification"]["score"],
        detailed_analysis["objection_handling"]["score"],
        detailed_analysis["closing_effectiveness"]["score"]
    ]
    overall_performance = sum(detailed_scores) / len(detailed_scores)
    
    # Aggregate business rule violations by rule (simple format)
    violations_by_rule = {}
    for violation in all_violations:
        # Use criteria_name or rule as the key
        rule_key = violation.get('criteria_name') or violation.get('rule')
        if not rule_key:
            continue
        if rule_key not in violations_by_rule:
            # Find the rule description and penalty from business_rules
            rule_obj = next((r for r in (business_rules or []) if r.get('criteria_name') == rule_key), {})
            violations_by_rule[rule_key] = {
                'rule': rule_obj.get('description', rule_key),
                'total_violations': 0,
                'total_score_penalty': 0,
                'items': []
            }
        # Add the violation instance (simple fields only)
        violations_by_rule[rule_key]['items'].append({
            'violation_text': violation.get('violation_text', ''),
            'explanation': violation.get('explanation', ''),
            'recommendation': violation.get('recommendation', '')
        })
        # Update counts
        penalty = violation.get('score_impact') or violation.get('score_penalty') or -2
        violations_by_rule[rule_key]['total_violations'] += 1
        violations_by_rule[rule_key]['total_score_penalty'] += penalty

    # Convert to list for report
    violations_by_rule_list = list(violations_by_rule.values())

    # Build a flat list of violation items
    violation_items = []
    total_score_penalty = 0
    for violation in all_violations:
        violation_items.append({
            "violation_text": violation.get("violation_text", ""),
            "explanation": violation.get("explanation", ""),
            "recommendation": violation.get("recommendation", "")
        })
        penalty = violation.get("score_impact") or violation.get("score_penalty") or -2
        total_score_penalty += penalty

    # Build final report with exact structure expected by frontend
    final_report = {
        "report_metadata": {
            "total_chunks_analyzed": len(chunk_analyses),
            "reference_files_used": list(all_reference_files),
            "analysis_timestamp": datetime.now().isoformat(),
            "call_duration_estimated": f"{len(chunk_analyses) * 5} minutes"
        },
        "executive_summary": {
            "overall_assessment": generate_overall_assessment(overall_score_penalized, letter_grade, len(all_strengths), len(all_weaknesses), len(chunk_analyses)),
            "overall_score": round(overall_score_penalized, 1),
            "letter_grade": letter_grade,
            "key_highlights": [s.get('description', '') for s in all_strengths[:3] if isinstance(s, dict)],
            "critical_areas": [w.get('description', '') for w in all_weaknesses[:3] if isinstance(w, dict)]
        },
        "detailed_analysis": detailed_analysis,
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
            "buying_signals": all_buying_signals,
            "concerns_expressed": all_concerns
        },
        "performance_metrics": {
            "rapport_building": int(round(detailed_analysis["rapport_building"]["score"])),
            "discovery": int(round(detailed_analysis["discovery_qualification"]["score"])),
            "objection_handling": int(round(detailed_analysis["objection_handling"]["score"])),
            "closing_effectiveness": int(round(detailed_analysis["closing_effectiveness"]["score"])),
            "overall_performance": int(round(overall_performance))
        }
    }
    
    # Add business rules violations if any
    if violation_items:
        final_report["custom_business_rules"] = {
            "violations": violation_items,
            "total_violations": len(violation_items),
            "total_score_penalty": total_score_penalty
        }
    
    return final_report

def clean_list_with_ai(section_name: str, items: list) -> list:
    """
    Use the LLM to deduplicate and resolve contradictions in a single list section, following client feedback.
    """
    if not items or not isinstance(items, list):
        return items
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    prompt = (
        f"You are a professional sales report editor. Here is a list of {section_name} from a sales call analysis report. "
        "Your task is to:\n"
        "- Remove duplicate or near-duplicate items.\n"
        "- If there are direct contradictions (e.g., two items say opposite things), clarify or merge them, but do not delete both unless one is clearly wrong.\n"
        "- Do NOT repeat or include phrases like: 'This is similar to the successful examples...', 'In the successful examples...', 'The closer...', 'The lead...', or 'No objections in this chunk/section/part/segment' or similar negative/filler statements.\n"
        "- Avoid repeating 'the closer' or 'the lead' in every bullet point. Assume the reader knows the roles.\n"
        "- Do NOT summarize, generalize, or omit any details or examples. Do NOT lose any substantive content from chunk analyses.\n"
        "- Return only the cleaned list as a JSON array, with no extra commentary.\n"
        f"Here is the list:\n{json.dumps(items, ensure_ascii=False, indent=2)}"
    )
    prompt_tokens = calculate_prompt_tokens(prompt)
    allowed_max_tokens = min(MAX_RESPONSE_TOKENS, CONTEXT_WINDOW - prompt_tokens - SAFETY_BUFFER)
    allowed_max_tokens = max(256, allowed_max_tokens)
    print(f"[AI CLEAN] Cleaning section '{section_name}' - prompt tokens: {prompt_tokens}, allowed max_tokens: {allowed_max_tokens}")
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=allowed_max_tokens
    )
    cleaned_list = json.loads(response.choices[0].message.content)
    return cleaned_list

def clean_final_report_with_ai(final_report: dict) -> dict:
    """
    Clean each large list section of the final report with the LLM, then reassemble.
    Output is identical in structure/detail to cleaning the whole report at once, but avoids context window errors.
    Now includes all substantive lists except 'violations'.
    """
    # Deep copy to avoid mutating input
    import copy
    cleaned_report = copy.deepcopy(final_report)
    # Define which sections/lists to clean (do NOT include 'violations')
    list_paths = [
        # Executive summary
        (['executive_summary', 'key_highlights'], 'executive_summary key_highlights'),
        (['executive_summary', 'critical_areas'], 'executive_summary critical_areas'),
        # Detailed analysis: objection_handling
        (['detailed_analysis', 'objection_handling', 'strengths'], 'objection_handling strengths'),
        (['detailed_analysis', 'objection_handling', 'weaknesses'], 'objection_handling weaknesses'),
        (['detailed_analysis', 'objection_handling', 'objections_encountered'], 'objection_handling objections_encountered'),
        (['detailed_analysis', 'objection_handling', 'handling_techniques_used'], 'objection_handling handling_techniques_used'),
        # Detailed analysis: rapport_building
        (['detailed_analysis', 'rapport_building', 'strengths'], 'rapport_building strengths'),
        (['detailed_analysis', 'rapport_building', 'weaknesses'], 'rapport_building weaknesses'),
        (['detailed_analysis', 'rapport_building', 'rapport_building_moments'], 'rapport_building moments'),
        # Detailed analysis: discovery_qualification
        (['detailed_analysis', 'discovery_qualification', 'strengths'], 'discovery_qualification strengths'),
        (['detailed_analysis', 'discovery_qualification', 'weaknesses'], 'discovery_qualification weaknesses'),
        (['detailed_analysis', 'discovery_qualification', 'information_gathered'], 'discovery_qualification information_gathered'),
        (['detailed_analysis', 'discovery_qualification', 'qualification_questions'], 'discovery_qualification qualification_questions'),
        # Detailed analysis: closing_effectiveness
        (['detailed_analysis', 'closing_effectiveness', 'strengths'], 'closing_effectiveness strengths'),
        (['detailed_analysis', 'closing_effectiveness', 'weaknesses'], 'closing_effectiveness weaknesses'),
        (['detailed_analysis', 'closing_effectiveness', 'closing_attempts'], 'closing_effectiveness closing_attempts'),
        (['detailed_analysis', 'closing_effectiveness', 'payment_discussion'], 'closing_effectiveness payment_discussion'),
        # Coaching recommendations
        (['coaching_recommendations'], 'coaching_recommendations'),
        # Reference comparisons
        (['reference_comparisons', 'similarities_to_successful_calls'], 'similarities_to_successful_calls'),
        (['reference_comparisons', 'differences_from_successful_calls'], 'differences_from_successful_calls'),
        (['reference_comparisons', 'best_practices_demonstrated'], 'best_practices_demonstrated'),
        (['reference_comparisons', 'missed_opportunities'], 'missed_opportunities'),
        # Lead interaction summary
        (['lead_interaction_summary', 'questions_asked'], 'lead_interaction_summary questions_asked'),
        (['lead_interaction_summary', 'buying_signals'], 'lead_interaction_summary buying_signals'),
        (['lead_interaction_summary', 'concerns_expressed'], 'lead_interaction_summary concerns_expressed'),
    ]
    for path, section_name in list_paths:
        # Traverse to the list
        d = cleaned_report
        for key in path[:-1]:
            d = d.get(key, {})
        last_key = path[-1]
        if last_key in d and isinstance(d[last_key], list) and d[last_key]:
            d[last_key] = clean_list_with_ai(section_name, d[last_key])
    return cleaned_report

def format_rules(business_rules: List[Dict]) -> str:
    """Format business rules as a compact numbered list, including violation_text and correct_text."""
    lines = []
    for i, rule in enumerate(business_rules, 1):
        lines.append(
            f"{i}. {rule.get('criteria_name', 'Rule')}: {rule.get('description', '')} "
            f"(Violation: '{rule.get('violation_text', '')}'"
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
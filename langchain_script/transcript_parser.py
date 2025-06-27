import re
from typing import Dict, List
from openai import OpenAI
import os
import json

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def extract_speakers(transcript: str) -> List[str]:
    """Extract unique speaker names from the transcript."""
    speaker_pattern = r'^([^:]+):'
    speakers = set()
    for line in transcript.split('\n'):
        match = re.match(speaker_pattern, line.strip())
        if match:
            speakers.add(match.group(1).strip())
    return list(speakers)

def identify_speaker_roles(speakers: List[str]) -> Dict[str, str]:
    """Identify which speaker is the sales rep and which is the prospect."""
    roles = {}
    sales_indicators = ['sales', 'rep', 'agent', 'representative']
    for speaker in speakers:
        is_sales = any(indicator.lower() in speaker.lower() for indicator in sales_indicators)
        roles[speaker] = 'sales_rep' if is_sales else 'prospect'
    return roles

def parse_transcript(transcript: str) -> dict:
    """
    Parse the transcript and identify key sections: opener, pitch, objections, payment, closing, etc.
    Uses GPT-4 to analyze and segment the conversation.
    """
    # Extract speakers and their roles
    speakers = extract_speakers(transcript)
    roles = identify_speaker_roles(speakers)
    
    # Build prompt for GPT-4 to analyze sections
    prompt = f"""
    Analyze this sales call transcript and segment it into key sections.
    Speakers: {json.dumps(roles)}
    
    Transcript:
    {transcript}
    
    Identify and extract the following sections (with exact quotes):
    1. Opener/Introduction
    2. Discovery/Needs Assessment
    3. Pitch/Value Proposition
    4. Objection Handling
    5. Payment/Pricing Discussion
    6. Closing/Next Steps
    
    For each section, include:
    - The exact text from the transcript
    - Start and end markers (speaker turns)
    - Key points discussed
    
    Respond in JSON format with these sections as keys.
    If a section is not present, use empty string for text and empty list for key points.
    """
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )
        
        sections = json.loads(response.choices[0].message.content)
    except Exception as e:
        sections = {
            "error": str(e),
            "raw_transcript": transcript
        }
    
    # Add metadata about speakers
    sections['metadata'] = {
        'speakers': speakers,
        'speaker_roles': roles,
        'total_turns': len(re.findall(r'^[^:]+:', transcript, re.MULTILINE))
    }
    
    return sections

if __name__ == "__main__":
    # Test the parser with a sample transcript
    sample = """
    Sales Rep: Hi there! Thanks for taking the time to chat today about our sales automation platform.
    
    Prospect: Thanks for having me. I've been looking into solutions like this.
    
    Sales Rep: Great to hear! Could you tell me about your current sales process and what challenges you're facing?
    
    Prospect: Well, we're a growing company and our sales team is struggling to keep up with leads.
    """
    
    result = parse_transcript(sample)
    print(json.dumps(result, indent=2)) 
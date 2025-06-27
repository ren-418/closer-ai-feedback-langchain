#this is main script

import os
from dotenv import load_dotenv
import json
from langchain.evaluator import evaluate_transcript_with_rag

# Load environment variables from .env file
load_dotenv()

def check_environment():
    """Check if all required environment variables are set."""
    required_vars = [
        'OPENAI_API_KEY',
        'PINECONE_API_KEY',
        'PINECONE_CLOUD',
        'PINECONE_REGION'
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Please set them in your .env file."
        )

def format_report(report):
    """Format the analysis report for console output."""
    print("\n=== Sales Call Analysis Report ===\n")
    
    if "raw_response" in report:
        print("Raw LLM Response:")
        print(report["raw_response"])
        return

    print("Overall Score:", report.get("score", "N/A"))
    print("Letter Grade:", report.get("letter_grade", "N/A"))
    print("\nStrengths:")
    for strength in report.get("strengths", []):
        print(f"- {strength}")
    
    print("\nWeaknesses:")
    for weakness in report.get("weaknesses", []):
        print(f"- {weakness}")
    
    print("\nSuggestions:")
    for suggestion in report.get("suggestions", []):
        print(f"- {suggestion}")
    
    if "summary" in report:
        print("\nSummary:")
        print(report["summary"])

def main():
    try:
        # Check environment variables
        check_environment()
        
        # Sample transcript for testing
        # In production, you would load this from a file or API
        sample_transcript = """
        Sales Rep: Hi there! Thanks for taking the time to chat with me today about our sales automation platform. Could you tell me a bit about your current sales process and what challenges you're facing?

        Prospect: Well, we're a growing company and our sales team is struggling to keep up with leads. We're using a basic CRM but it's not really helping us automate anything.

        Sales Rep: I understand completely. Managing leads manually can be overwhelming, especially as you grow. Our platform actually helped similar companies increase their lead processing capacity by 3x while reducing manual work. Would you be interested in seeing a quick demo of how we do that?

        Prospect: Yes, that would be helpful. We definitely need to improve our efficiency.
        """
        
        print("AI Sales Call Evaluator - Processing transcript...")
        
        # Run the RAG evaluation pipeline
        report = evaluate_transcript_with_rag(sample_transcript)
        
        # Format and display the results
        format_report(report)
        
        # Optionally save the report
        with open('latest_analysis.json', 'w') as f:
            json.dump(report, f, indent=2)
            print("\nFull report saved to latest_analysis.json")
            
    except EnvironmentError as e:
        print(f"Environment Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()


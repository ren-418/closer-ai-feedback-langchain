#this is main script

import os
from dotenv import load_dotenv
import json
import sys
from langchain_script.evaluator import SalesCallEvaluator

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
    """Format the enhanced analysis report for console output."""
    print("\n" + "="*60)
    print("           AI SALES CALL EVALUATION REPORT")
    print("="*60)
    
    if report.get('status') == 'failed':
        print(f"❌ Evaluation Failed: {report.get('error', 'Unknown error')}")
        return

    # Display metadata
    metadata = report.get('metadata', {})
    print(f"\n📊 ANALYSIS METADATA:")
    print(f"   • Total chunks analyzed: {metadata.get('total_chunks', 'N/A')}")
    print(f"   • Reference files used: {metadata.get('total_reference_files_used', 'N/A')}")
    print(f"   • Evaluation timestamp: {metadata.get('evaluation_timestamp', 'N/A')}")
    print(f"   • Estimated call duration: {metadata.get('estimated_call_duration', 'N/A')}")
    
    # Display reference files
    if metadata.get('reference_files'):
        print(f"\n📁 REFERENCE FILES USED:")
        for i, ref_file in enumerate(metadata['reference_files'], 1):
            print(f"   {i}. {ref_file}")
    
    # Display executive summary
    final = report.get('final_analysis', {})
    executive = final.get('executive_summary', {})
    
    print(f"\n🏆 EXECUTIVE SUMMARY:")
    print(f"   • Overall Score: {executive.get('overall_score', 'N/A')}/100")
    print(f"   • Letter Grade: {executive.get('letter_grade', 'N/A')}")
    print(f"   • Assessment: {executive.get('overall_assessment', 'N/A')}")
    
    # Display key highlights and critical areas
    if executive.get('key_highlights'):
        print(f"\n✨ KEY HIGHLIGHTS:")
        for highlight in executive['key_highlights']:
            print(f"   • {highlight}")
    
    if executive.get('critical_areas'):
        print(f"\n⚠️  CRITICAL AREAS:")
        for area in executive['critical_areas']:
            print(f"   • {area}")
    
    # Display detailed analysis
    detailed = final.get('detailed_analysis', {})
    if detailed:
        print(f"\n📈 DETAILED PERFORMANCE ANALYSIS:")
        
        for category, data in detailed.items():
            if isinstance(data, dict) and 'score' in data:
                category_name = category.replace('_', ' ').title()
                score = data.get('score', 'N/A')
                print(f"\n   {category_name}: {score}/10")
                
                if data.get('strengths'):
                    print(f"     Strengths:")
                    for strength in data['strengths']:
                        print(f"       • {strength}")
                
                if data.get('weaknesses'):
                    print(f"     Areas for Improvement:")
                    for weakness in data['weaknesses']:
                        print(f"       • {weakness}")
    
    # Display coaching recommendations
    coaching = final.get('coaching_recommendations', [])
    if coaching:
        print(f"\n🎯 COACHING RECOMMENDATIONS:")
        for i, rec in enumerate(coaching, 1):
            priority = rec.get('priority', 'medium').upper()
            category = rec.get('category', 'general').replace('_', ' ').title()
            recommendation = rec.get('recommendation', 'N/A')
            
            print(f"\n   {i}. [{priority} PRIORITY] {category}")
            print(f"      💡 {recommendation}")
            
            if rec.get('reference_example'):
                print(f"      📚 Reference: {rec['reference_example']}")
            
            if rec.get('expected_impact'):
                print(f"      🎯 Expected Impact: {rec['expected_impact']}")
    
    # Display lead interaction summary
    lead_summary = final.get('lead_interaction_summary', {})
    if lead_summary:
        print(f"\n👥 LEAD INTERACTION SUMMARY:")
        print(f"   • Total questions asked: {lead_summary.get('total_questions_asked', 'N/A')}")
        print(f"   • Total objections raised: {lead_summary.get('total_objections_raised', 'N/A')}")
        print(f"   • Engagement pattern: {lead_summary.get('engagement_pattern', 'N/A')}")
        
        if lead_summary.get('buying_signals'):
            print(f"   • Buying signals detected: {', '.join(lead_summary['buying_signals'])}")
        
        if lead_summary.get('concerns_expressed'):
            print(f"   • Concerns expressed: {', '.join(lead_summary['concerns_expressed'])}")
    
    # Display performance metrics
    metrics = final.get('performance_metrics', {})
    if metrics:
        print(f"\n📊 PERFORMANCE METRICS:")
        for metric, score in metrics.items():
            if isinstance(score, (int, float)):
                metric_name = metric.replace('_', ' ').title()
                print(f"   • {metric_name}: {score}/10")
    
    # Display custom business rules violations
    custom_rules = final.get('custom_business_rules', {})
    violations = custom_rules.get('violations_found', [])
    if violations:
        print(f"\n🚨 CUSTOM BUSINESS RULES VIOLATIONS:")
        print(f"   • Total violations found: {custom_rules.get('total_violations', len(violations))}")
        print(f"   • Total score penalty: {custom_rules.get('total_score_penalty', 0)} points")
        
        for i, violation in enumerate(violations, 1):
            rule = violation.get('rule', 'Unknown rule')
            violation_text = violation.get('violation_text', 'Unknown')
            context = violation.get('context', 'No context provided')
            correct_text = violation.get('correct_text', 'N/A')
            explanation = violation.get('explanation', 'No explanation provided')
            score_impact = violation.get('score_impact', 0)
            
            print(f"\n   {i}. {rule.upper()}")
            print(f"      ❌ Violation: '{violation_text}' found in context")
            print(f"      📝 Context: '{context}'")
            print(f"      ✅ Should use: '{correct_text}'")
            print(f"      💬 Explanation: {explanation}")
            print(f"      📉 Score impact: {score_impact} points")
        
        # Display recommendations
        recommendations = custom_rules.get('recommendations', [])
        if recommendations:
            print(f"\n   📋 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"      • {rec}")
    else:
        print(f"\n✅ CUSTOM BUSINESS RULES: No violations found")
    
    print(f"\n" + "="*60)
    print("           END OF REPORT")
    print("="*60)

def main():
    try:
        # Check environment variables
        check_environment()
        
        evaluator = SalesCallEvaluator()
        
        # Support loading transcript from file via CLI
        if len(sys.argv) > 1:
            transcript_path = sys.argv[1]
            print(f"[Main] Loading transcript from file: {transcript_path}")
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
        else:
            print("[Main] No file provided. Using sample transcript.")
            transcript = """
            Sales Rep: Hi there! Thanks for taking the time to chat with me today about our sales automation platform. Could you tell me a bit about your current sales process and what challenges you're facing?

            Prospect: Well, we're a growing company and our sales team is struggling to keep up with leads. We're using a basic CRM but it's not really helping us automate anything.

            Sales Rep: I understand completely. Managing leads manually can be overwhelming, especially as you grow. Our platform actually helped similar companies increase their lead processing capacity by 3x while reducing manual work. Would you be interested in seeing a quick demo of how we do that?

            Prospect: Yes, that would be helpful. We definitely need to improve our efficiency.
            """
        
        print("🤖 AI Sales Call Evaluator - Processing transcript...")
        print(f"📝 Transcript length: {len(transcript)} characters")
        
        # Run the enhanced RAG evaluation pipeline
        report = evaluator.evaluate_transcript(transcript)
        
        # Format and display the results
        format_report(report)
        
        # Save the detailed report
        with open('latest_analysis.json', 'w') as f:
            json.dump(report, f, indent=2)
            print(f"\n💾 Full detailed report saved to latest_analysis.json")
            
    except EnvironmentError as e:
        print(f"❌ Environment Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()


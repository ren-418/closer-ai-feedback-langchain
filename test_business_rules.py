#!/usr/bin/env python3
"""
Test script for custom business rules functionality.
Demonstrates how the system detects currency violations and other business rule violations.
"""

import os
from dotenv import load_dotenv
from langchain_script.evaluator import SalesCallEvaluator

# Load environment variables
load_dotenv()

def test_business_rules():
    """Test the custom business rules functionality."""
    
    # Test transcript with currency violations
    test_transcript = """
    Sales Rep: Hi there! Thanks for taking the time to chat today about our premium service package.
    
    Prospect: Sure, I'm interested in learning more about what you offer.
    
    Sales Rep: Great! Our premium package is priced at 500 pounds and includes all the features you need.
    
    Prospect: That sounds reasonable. What about payment options?
    
    Sales Rep: We accept all major credit cards and can also do bank transfers in euros if you prefer.
    
    Prospect: I think I'd like to proceed with the 500 quid option.
    
    Sales Rep: Perfect! Let me get you set up with that 500 quid package right away.
    """
    
    print("🧪 Testing Custom Business Rules Functionality")
    print("=" * 60)
    print("Test Transcript (contains currency violations):")
    print("-" * 40)
    print(test_transcript)
    print("-" * 40)
    
    try:
        # Initialize evaluator
        evaluator = SalesCallEvaluator()
        
        print("\n🤖 Running analysis with custom business rules...")
        
        # Run evaluation
        report = evaluator.evaluate_transcript(test_transcript)
        
        # Check for custom business rules violations
        final_analysis = report.get('final_analysis', {})
        custom_rules = final_analysis.get('custom_business_rules', {})
        violations = custom_rules.get('violations_found', [])
        
        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"   • Total violations found: {len(violations)}")
        print(f"   • Total score penalty: {custom_rules.get('total_score_penalty', 0)} points")
        
        if violations:
            print(f"\n🚨 VIOLATIONS DETECTED:")
            for i, violation in enumerate(violations, 1):
                print(f"\n   {i}. {violation.get('rule', 'Unknown').upper()}")
                print(f"      ❌ Violation: '{violation.get('violation_text', 'Unknown')}'")
                print(f"      📝 Context: '{violation.get('context', 'No context')}'")
                print(f"      ✅ Should use: '{violation.get('correct_text', 'N/A')}'")
                print(f"      💬 Explanation: {violation.get('explanation', 'No explanation')}")
                print(f"      📉 Score impact: {violation.get('score_impact', 0)} points")
        else:
            print("\n✅ No violations detected")
        
        # Show recommendations
        recommendations = custom_rules.get('recommendations', [])
        if recommendations:
            print(f"\n📋 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"   • {rec}")
        
        # Show final score
        executive_summary = final_analysis.get('executive_summary', {})
        print(f"\n🏆 FINAL SCORE:")
        print(f"   • Overall Score: {executive_summary.get('overall_score', 'N/A')}/100")
        print(f"   • Letter Grade: {executive_summary.get('letter_grade', 'N/A')}")
        
        print(f"\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    test_business_rules() 
#!/usr/bin/env python3
"""
Quick test to verify the MAX_RESPONSE_TOKENS fix
"""

try:
    from langchain_script.analysis import MAX_RESPONSE_TOKENS, MAX_CHUNK_TOKENS, MAX_REF_CHUNKS
    print("✅ All constants imported successfully!")
    print(f"   MAX_RESPONSE_TOKENS: {MAX_RESPONSE_TOKENS}")
    print(f"   MAX_CHUNK_TOKENS: {MAX_CHUNK_TOKENS}")
    print(f"   MAX_REF_CHUNKS: {MAX_REF_CHUNKS}")
    
    from langchain_script.evaluator import SalesCallEvaluator
    print("✅ SalesCallEvaluator imported successfully!")
    
    print("\n🎉 All imports working correctly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}") 
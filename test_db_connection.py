#!/usr/bin/env python3
"""
Simple database connection test
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_env_vars():
    """Test if environment variables are set."""
    print("🔍 Checking environment variables...")
    
    required_vars = {
        'SUPABASE_URL': 'Supabase project URL',
        'SUPABASE_KEY': 'Supabase anon key',
        'OPENAI_API_KEY': 'OpenAI API key',
        'PINECONE_API_KEY': 'Pinecone API key',
        'PINECONE_CLOUD': 'Pinecone cloud',
        'PINECONE_REGION': 'Pinecone region'
    }
    
    missing = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            print(f"   ✅ {var}: {description} - SET")
        else:
            print(f"   ❌ {var}: {description} - MISSING")
            missing.append(var)
    
    if missing:
        print(f"\n⚠️  Missing environment variables: {', '.join(missing)}")
        print("Please create a .env file with the required variables.")
        print("See env_template.txt for the required format.")
        return False
    else:
        print("\n✅ All environment variables are set!")
        return True

def test_database_connection():
    """Test database connection."""
    try:
        print("\n🔌 Testing database connection...")
        from database.database_manager import DatabaseManager
        
        db_manager = DatabaseManager()
        print("   ✅ Database manager created successfully")
        
        # Test basic query
        closers = db_manager.get_all_closers()
        print(f"   ✅ Retrieved {len(closers)} closers from database")
        
        # Test business rules
        rules = db_manager.get_business_rules()
        print(f"   ✅ Retrieved {len(rules)} business rules from database")
        
        print("✅ Database connection test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Database connection test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Database Connection Test")
    print("=" * 40)
    
    # Test environment variables
    if not test_env_vars():
        print("\n❌ Environment setup incomplete")
        return False
    
    # Test database connection
    if not test_database_connection():
        print("\n❌ Database connection failed")
        return False
    
    print("\n🎉 All tests passed!")
    print("Your database is ready to use.")
    return True

if __name__ == "__main__":
    main() 
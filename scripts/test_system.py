#!/usr/bin/env python3
"""
Test script to verify VEGETA system components
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
from vegeta import VegetaSystem, Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_system_initialization():
    """Test basic system initialization"""
    print("Testing system initialization...")
    
    try:
        config = Config()
        print(f"✓ Config loaded: {config.get('database.uri')}")
        
        system = VegetaSystem(config)
        print("✓ VEGETA system initialized")
        
        system.close()
        print("✓ System closed cleanly")
        return True
        
    except Exception as e:
        print(f"✗ System initialization failed: {e}")
        return False

def test_single_query():
    """Test processing a single query"""
    print("\nTesting single query processing...")
    
    try:
        config = Config()
        system = VegetaSystem(config)
        
        # Start session
        session_id = system.start_session()
        print(f"✓ Session started: {session_id}")
        
        # Process test query
        test_query = "I'm thinking of a Pierce Brosnan spy film from the 1990s"
        print(f"Processing query: '{test_query}'")
        
        response = system.process_query(session_id, test_query)
        
        print(f"✓ Response generated:")
        print(f"  Action: {response.action}")
        print(f"  Content: {response.content}")
        print(f"  Confidence: {response.confidence:.1%}")
        print(f"  Target: {response.target}")
        
        system.close()
        return True
        
    except Exception as e:
        print(f"✗ Query processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_turn():
    """Test multi-turn conversation"""
    print("\nTesting multi-turn conversation...")
    
    try:
        config = Config()
        system = VegetaSystem(config)
        
        session_id = system.start_session()
        
        # First turn
        response1 = system.process_query(session_id, "I'm thinking of a film. Try to guess it.")
        print(f"Turn 1 - Action: {response1.action}, Content: {response1.content[:50]}...")
        
        # Simulate user feedback for ASK responses
        if response1.action == 'ASK':
            system.add_user_feedback(session_id, "It's a sci-fi action movie", "partial")
        
        # Second turn
        response2 = system.process_query(session_id, "It's a sci-fi action movie from 1999")
        print(f"Turn 2 - Action: {response2.action}, Content: {response2.content[:50]}...")
        
        # Check session summary
        summary = system.get_session_summary(session_id)
        print(f"✓ Session summary: {summary.get('conversation_summary', 'None')}")
        
        system.close()
        return True
        
    except Exception as e:
        print(f"✗ Multi-turn test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("VEGETA System Test Suite")
    print("=" * 40)
    
    tests = [
        test_system_initialization,
        test_single_query,
        test_multi_turn
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 40)
    print("Test Results:")
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! VEGETA system is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

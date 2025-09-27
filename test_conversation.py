#!/usr/bin/env python3
"""
Test script to verify multi-turn conversation functionality
"""

from vegeta.core.system import VegetaSystem
from vegeta.core.config import Config

def test_multi_turn_conversation():
    """Test multi-turn conversation with Daniel Craig example"""

    # Initialize system
    config = Config()
    system = VegetaSystem(config)

    try:
        # Start session
        session_id = system.start_session()
        print(f"✅ Started session: {session_id}")

        # Turn 1: Ask about Daniel Craig films
        print("\n🔄 Turn 1: Processing 'what films did daniel craig star in?'")
        response1 = system.process_query(session_id, "what films did daniel craig star in?")
        print(f"🎯 Action: {response1.action}")
        print(f"📝 Response: {response1.content}")
        print(f"🎪 Confidence: {response1.confidence:.1%}")

        # Turn 2: Confirm with "yes"
        print("\n🔄 Turn 2: Processing 'yes'")
        response2 = system.process_query(session_id, "yes")
        print(f"🎯 Action: {response2.action}")
        print(f"📝 Response: {response2.content}")
        print(f"🎪 Confidence: {response2.confidence:.1%}")

        # Check if conversation context was used
        if hasattr(response2, 'internal_state') and response2.internal_state:
            print("✅ Internal state available - conversation context working!")
        else:
            print("❌ No internal state - conversation context may not be working")

        print("\n🎉 Multi-turn conversation test completed!")

    except Exception as e:
        print(f"❌ Error during conversation test: {e}")
        import traceback
        traceback.print_exc()

    finally:
        system.close()

if __name__ == "__main__":
    test_multi_turn_conversation()


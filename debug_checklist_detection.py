#!/usr/bin/env python3

from vegeta.core.config import Config
from vegeta.retrieval.graph_retriever import GraphRetriever
from vegeta.utils.database import DatabaseManager

def test_checklist_detection():
    config = Config()
    db = DatabaseManager(config.database)
    retriever = GraphRetriever(db, config)

    # Test cases with different query types
    test_cases = [
        "I want action movies similar to Heat",
        "What music rights do I need for Skyfall?",
        "Who won the Oscar for best actor?",
        "Tell me about Daniel Craig",
        "My name is Chris"
    ]

    print("Testing _get_active_checklist_and_target_labels method:")
    print("=" * 60)

    for test_case in test_cases:
        print(f"\nTest Query: '{test_case}'")

        # Create observation_u format
        observation_u = {
            'u_meta': {
                'utterance': test_case
            }
        }

        try:
            checklist_name, target_labels = retriever._get_active_checklist_and_target_labels(observation_u)
            print(f"  Active Checklist: {checklist_name}")
            print(f"  Target Labels: {target_labels}")
        except Exception as e:
            print(f"  Error: {e}")

    db.close()

if __name__ == '__main__':
    test_checklist_detection()
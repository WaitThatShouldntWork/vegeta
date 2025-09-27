"""
Test cases and ground truth data for VEGETA benchmarking
"""

# Test Case Categories - SELECTIVE SET (1-2 per category for quick validation)
BENCHMARK_CATEGORIES = {
    'music_rights_verification': {
        'single_turn': [
            "Check music rights for Skyfall"  # Test core VerifyMusicRights functionality
        ],
        'multi_turn': [
            ["I need to verify music rights", "For the film Skyfall", "Check composer and territory"]
        ]
    },
    'identification': {
        'single_turn': [
            "What's the Daniel Craig Bond movie from 2006?"  # Test basic film identification
        ],
        'multi_turn': [
            ["Looking for a Bond film", "Pierce Brosnan", "From the 90s"]
        ]
    },
    'recommendation': {
        'single_turn': [
            "Suggest action movies like Heat"  # Test recommendation system
        ],
        'multi_turn': [
            ["What should I watch?", "Similar to Casino Royale", "Spy thriller"]
        ]
    },
    'verification': {
        'single_turn': [
            "Did Skyfall win the BAFTA award?"  # Test fact verification
        ],
        'multi_turn': [
            ["About Skyfall", "Did it win awards?", "Which BAFTA?"]
        ]
    },
    'ambiguous': {
        'single_turn': [
            "Bond movie"  # Test ambiguity handling
        ],
        'multi_turn': [
            ["Film search", "Not sure which one"]
        ]
    },
}

# Ground Truth Dataset - SELECTIVE SET matching test cases above
GROUND_TRUTH = {
    # Core Music Rights Verification (our main feature)
    "Check music rights for Skyfall": {
        'expected_action': 'ANSWER',
        'checklist_slots': ['film', 'composer', 'music_track', 'territory_clearance', 'sync_rights'],
        'expected_findings': {
            'film': 'Skyfall',
            'composer': 'Adele',
            'music_track': 'Skyfall (Adele)',
            'territory': 'Worldwide',
            'sync_rights': 'Sync Rights Template Document'
        },
        'confidence_range': (0.8, 0.95),
        'max_turns': 1,
        'category': 'music_rights_verification'
    },

    # Basic Identification
    "What's the Daniel Craig Bond movie from 2006?": {
        'expected_action': 'ANSWER',
        'correct_answer': 'Casino Royale',
        'confidence_range': (0.8, 0.95),
        'max_turns': 1,
        'category': 'identification'
    },

    # Basic Recommendation
    "Suggest action movies like Heat": {
        'expected_action': 'ANSWER',
        'correct_answer': ['Casino Royale', 'The Matrix'],
        'similarity_factors': ['action', 'complex_plot', 'strong_cast'],
        'confidence_range': (0.6, 0.8),
        'max_turns': 2,
        'category': 'recommendation'
    },

    # Minimal benchmark test case (same as recommendation but different wording)
    "I want action movies similar to Heat": {
        'expected_action': 'ASK',  # System should ask for clarification
        'expected_targets': ['specific_preferences', 'genre'],
        'confidence_range': (0.2, 0.4),
        'max_turns': 2,
        'category': 'recommendation'
    },

    # Basic Verification
    "Did Skyfall win the BAFTA award?": {
        'expected_action': 'ANSWER',
        'correct_answer': 'Yes, BAFTA for Outstanding British Film',
        'source': 'Wikipedia',
        'confidence_range': (0.8, 0.95),
        'max_turns': 1,
        'category': 'verification'
    },

    # Ambiguous Case
    "Bond movie": {
        'expected_action': 'ASK',
        'expected_targets': ['specific_film', 'actor', 'year'],
        'possible_answers': ['Skyfall', 'Casino Royale', 'GoldenEye'],
        'confidence_range': (0.1, 0.3),
        'max_turns': 3,
        'category': 'ambiguous'
    },

    # Multi-turn Test Cases (our main focus)
    "I need to verify music rights": {
        'expected_action': 'ASK',
        'expected_targets': ['film', 'specific_rights'],
        'follow_up_responses': {
            'For the film Skyfall': {'action': 'ASK', 'targets': ['specific_rights']},
            'Check composer and territory': {'action': 'ANSWER', 'slots_filled': 5, 'verify_music_rights_checklist': True}
        },
        'confidence_range': (0.2, 0.5),
        'max_turns': 3,
        'category': 'music_rights_verification'
    },

    # Add ground truth for the second step in our multi-turn test
    "For the film Skyfall": {
        'expected_action': 'ASK',
        'expected_targets': ['specific_rights'],
        'confidence_range': (0.3, 0.6),
        'max_turns': 2,
        'category': 'music_rights_verification'
    },

    # Add ground truth for the third step
    "Check composer and territory": {
        'expected_action': 'ANSWER',
        'expected_findings': {
            'composer': 'Adele',
            'territory': 'Worldwide',
            'music_track': 'Skyfall (Adele)',
            'sync_rights': 'Sync Rights Template Document'
        },
        'confidence_range': (0.8, 0.95),
        'max_turns': 1,
        'category': 'music_rights_verification'
    },

    "Looking for a Bond film": {
        'expected_action': 'ASK',
        'expected_targets': ['actor', 'year'],
        'follow_up_responses': {
            'Pierce Brosnan': {'action': 'ASK', 'targets': ['year']},
            'From the 90s': {'action': 'ANSWER', 'answer': 'GoldenEye'}
        },
        'confidence_range': (0.3, 0.6),
        'max_turns': 3,
        'category': 'identification'
    },

    # Missing ground truth for identification multi-turn steps
    "Pierce Brosnan": {
        'expected_action': 'ASK',
        'expected_targets': ['year'],
        'confidence_range': (0.4, 0.7),
        'max_turns': 2,
        'category': 'identification'
    },

    "From the 90s": {
        'expected_action': 'ANSWER',
        'correct_answer': 'GoldenEye (1995)',
        'confidence_range': (0.8, 0.95),
        'max_turns': 1,
        'category': 'identification'
    },

    "What should I watch?": {
        'expected_action': 'ASK',
        'expected_targets': ['genre', 'specific_preferences'],
        'follow_up_responses': {
            'Similar to Casino Royale': {'action': 'ASK', 'targets': ['genre']},
            'Spy thriller': {'action': 'ANSWER', 'recommendations': ['Skyfall', 'Casino Royale']}
        },
        'confidence_range': (0.2, 0.5),
        'max_turns': 3,
        'category': 'recommendation'
    },

    # Add ground truth for recommendation multi-turn steps
    "Similar to Casino Royale": {
        'expected_action': 'ASK',
        'expected_targets': ['genre'],
        'confidence_range': (0.3, 0.6),
        'max_turns': 2,
        'category': 'recommendation'
    },

    # Add ground truth for the final step
    "Spy thriller": {
        'expected_action': 'ANSWER',
        'recommendations': ['Skyfall', 'Casino Royale'],
        'confidence_range': (0.7, 0.9),
        'max_turns': 1,
        'category': 'recommendation'
    },

    "About Skyfall": {
        'expected_action': 'ANSWER',
        'correct_answer': 'Skyfall is a 2012 James Bond film',
        'follow_up_responses': {
            'Did it win awards?': {'action': 'ANSWER', 'answer': 'Yes, BAFTA'},
            'Which BAFTA?': {'action': 'ANSWER', 'answer': 'Outstanding British Film'}
        },
        'confidence_range': (0.7, 0.9),
        'max_turns': 3,
        'category': 'verification'
    },

    # Missing ground truth for verification multi-turn steps
    "Did it win awards?": {
        'expected_action': 'ANSWER',
        'correct_answer': 'Yes, Skyfall won the BAFTA for Outstanding British Film',
        'confidence_range': (0.8, 0.95),
        'max_turns': 1,
        'category': 'verification'
    },

    "Which BAFTA?": {
        'expected_action': 'ANSWER',
        'correct_answer': 'Outstanding British Film',
        'confidence_range': (0.8, 0.95),
        'max_turns': 1,
        'category': 'verification'
    },

    "Film search": {
        'expected_action': 'ASK',
        'expected_targets': ['specific_film', 'genre', 'actor'],
        'follow_up_responses': {
            'Not sure which one': {'action': 'ASK', 'targets': ['specific_film', 'actor', 'year']}
        },
        'confidence_range': (0.1, 0.4),
        'max_turns': 3,
        'category': 'ambiguous'
    },

    # Missing ground truth for ambiguous multi-turn steps
    "Not sure which one": {
        'expected_action': 'ASK',
        'expected_targets': ['specific_film', 'actor', 'year'],
        'confidence_range': (0.1, 0.3),
        'max_turns': 2,
        'category': 'ambiguous'
    }
}

# Utility Functions for Test Case Management
def get_test_cases_by_category(category: str, turn_type: str = 'single_turn'):
    """
    Get test cases for a specific category and turn type

    Args:
        category: One of 'music_rights_verification', 'identification', 'recommendation', 'verification', 'ambiguous'
        turn_type: 'single_turn' or 'multi_turn'

    Returns:
        List of test case strings or list of lists for multi-turn
    """
    if category not in BENCHMARK_CATEGORIES:
        raise ValueError(f"Unknown category: {category}")

    if turn_type not in BENCHMARK_CATEGORIES[category]:
        raise ValueError(f"Unknown turn type: {turn_type}")

    return BENCHMARK_CATEGORIES[category][turn_type]

def get_ground_truth(query: str):
    """
    Get ground truth for a specific query

    Args:
        query: The user query string

    Returns:
        Dictionary with ground truth data or None if not found
    """
    return GROUND_TRUTH.get(query)

def validate_system_response(query: str, system_action: str, system_confidence: float = None):
    """
    Validate system response against ground truth

    Args:
        query: The user query
        system_action: The action taken by the system ('ASK', 'ANSWER', 'SEARCH')
        system_confidence: Confidence score (0.0 to 1.0)

    Returns:
        Dictionary with validation results
    """
    ground_truth = get_ground_truth(query)
    if not ground_truth:
        return {'error': 'No ground truth found for query'}

    validation = {
        'query': query,
        'expected_action': ground_truth['expected_action'],
        'actual_action': system_action,
        'action_correct': system_action == ground_truth['expected_action'],
        'category': ground_truth['category']
    }

    # Check confidence if provided
    if system_confidence is not None and 'confidence_range' in ground_truth:
        min_conf, max_conf = ground_truth['confidence_range']
        validation['confidence_in_range'] = min_conf <= system_confidence <= max_conf
        validation['expected_confidence_range'] = ground_truth['confidence_range']
        validation['actual_confidence'] = system_confidence

    return validation

def get_verify_music_rights_test_suite():
    """
    Get comprehensive test suite for VerifyMusicRights checklist validation

    Returns:
        List of test scenarios with expected outcomes
    """
    return [
        {
            'name': 'Complete Skyfall Rights Check',
            'queries': [
                "Check music rights for Skyfall",
                "Who is the composer?",
                "What territory is cleared?",
                "Show me the sync rights document"
            ],
            'expected_slots': {
                'film': 'Skyfall',
                'composer': 'Adele',
                'music_track': 'Skyfall (Adele)',
                'territory_clearance': 'Worldwide',
                'sync_rights': 'Sync Rights Template Document'
            },
            'expected_actions': ['ANSWER', 'ANSWER', 'ANSWER', 'ANSWER']
        },
        {
            'name': 'Heat Music Rights Inquiry',
            'queries': [
                "I need to verify music rights",
                "For the film Heat",
                "Check composer and territory"
            ],
            'expected_slots': {
                'film': 'Heat',
                'music_track': 'Heat Original Score',
                'territory_clearance': 'Worldwide'
            },
            'expected_actions': ['ASK', 'ANSWER', 'ANSWER']
        }
    ]

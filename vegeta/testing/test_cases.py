"""
Test cases and ground truth data for VEGETA benchmarking
"""

# Test Case Categories
BENCHMARK_CATEGORIES = {
    'identification': {
        'single_turn': [
            "I'm looking for a spy movie with Pierce Brosnan from the 1990s",
            "What's the film where Tom Cruise hangs from a building?",
            "Which Bond movie won a BAFTA?",
            "Film with Al Pacino and Robert De Niro",
            "The Matrix movie from 1999"
        ],
        'multi_turn': [
            ["I want a Pierce Brosnan movie", "It's a spy film", "From the 1990s"],
            ["Looking for an action movie", "Has Tom Cruise", "Mission Impossible"],
            ["Bond film", "Daniel Craig", "Won awards"]
        ]
    },
    'recommendation': {
        'single_turn': [
            "I want action movies similar to Heat",
            "Suggest spy films like Mission Impossible", 
            "What are good Daniel Craig movies?",
            "Recommend thrillers with complex plots"
        ],
        'multi_turn': [
            ["I want movie recommendations", "Action genre", "Something like Heat"],
            ["Suggest a film", "I like spy movies", "Maybe something recent"],
            ["What should I watch?", "I enjoyed Casino Royale", "Similar style"]
        ]
    },
    'verification': {
        'single_turn': [
            "Did Skyfall win any awards?",
            "Is Pierce Brosnan in GoldenEye?",
            "Was The Matrix released in 1999?",
            "Did Daniel Craig star in Casino Royale?"
        ],
        'multi_turn': [
            ["Tell me about Skyfall", "Did it win awards?", "Which ones?"],
            ["Is Pierce Brosnan in Bond films?", "Which ones?", "Any from the 90s?"]
        ]
    },
    'ambiguous': {
        'single_turn': [
            "Bond movie",  # Multiple candidates
            "Action film",  # Very broad  
            "Pierce Brosnan",  # Just an actor
            "Movie from 1995"
        ],
        'multi_turn': [
            ["Bond", "Pierce Brosnan", "Which one?"],
            ["Action movie", "1990s", "With car chases"]
        ]
    },
    'edge_cases': {
        'single_turn': [
            "asdf random text",
            "Tell me about quantum physics", 
            "Show me horror movies",
            "What's the weather?"
        ],
        'multi_turn': [
            ["Random text", "Still random", "Make sense of this"],
            ["Physics", "Not movies", "Something else"]
        ]
    }
}

# Ground Truth Dataset
GROUND_TRUTH = {
    "I'm looking for a spy movie with Pierce Brosnan from the 1990s": {
        'expected_action': 'ASK',  # Should ask which specific film
        'expected_targets': ['film', 'actor'],  # Could ask about film or clarify actor
        'correct_answer': 'GoldenEye',
        'confidence_range': (0.2, 0.5),  # Should be uncertain initially
        'max_turns': 2,  # Should resolve in 2 turns
        'category': 'identification'
    },
    "Did Skyfall win any awards?": {
        'expected_action': 'ANSWER',  # We have this fact in our data
        'correct_answer': 'Yes, BAFTA for Outstanding British Film',
        'confidence_range': (0.7, 0.9),  # Should be confident
        'max_turns': 1,
        'category': 'verification'
    },
    "I want action movies similar to Heat": {
        'expected_action': 'ASK',  # Should ask for preferences/clarification
        'expected_targets': ['preference', 'genre', 'similar_films'],
        'correct_answer': ['Ronin', 'True Lies'],  # Similar action films in our data
        'confidence_range': (0.3, 0.6),
        'max_turns': 3,
        'category': 'recommendation'
    },
    "Bond movie": {
        'expected_action': 'ASK',  # Too ambiguous
        'expected_targets': ['film', 'actor', 'year'],
        'correct_answer': None,  # Multiple valid answers
        'confidence_range': (0.1, 0.4),
        'max_turns': 3,
        'category': 'ambiguous'
    }
}

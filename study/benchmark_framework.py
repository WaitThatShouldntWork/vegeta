# %% [markdown]
# # Benchmarking Framework for Bayesian Active Inference System
# Comprehensive evaluation suite with multi-turn testing

# %%
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

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

@dataclass
class EvaluationMetrics:
    """Metrics for evaluating system performance"""
    
    # Decision Quality
    decision_accuracy: float  # Percentage of correct action choices
    confidence_calibration: float  # How well confidence matches accuracy
    appropriate_uncertainty: float  # Good uncertainty when answer unknown
    
    # Efficiency  
    avg_turns_to_resolution: float
    question_efficiency: float  # Useful questions / total questions
    
    # Question Quality (human eval)
    question_naturalness: float  # 1-5 scale
    question_relevance: float   # 1-5 scale
    question_specificity: float  # 1-5 scale
    
    # Answer Quality
    answer_accuracy: float
    answer_completeness: float
    
    # System Behavior
    overconfidence_rate: float  # High confidence + wrong answer
    underconfidence_rate: float  # Low confidence + correct answer
    repetitive_questions: float  # Same question multiple times
    
    # Performance
    avg_response_time: float
    api_calls_per_turn: float

class BenchmarkEvaluator:
    """Evaluates system performance across test cases"""
    
    def __init__(self, system):
        self.system = system
        self.results = []
        
    def run_single_turn_benchmark(self, test_cases: Dict[str, List[str]]) -> Dict:
        """Run single-turn evaluation"""
        
        results = {}
        
        for category, cases in test_cases.items():
            category_results = []
            
            for test_case in cases:
                print(f"Testing: {test_case}")
                
                # Run system
                start_time = datetime.now()
                session_id = self.system.start_session()
                result = self.system.process_turn(session_id, test_case)
                end_time = datetime.now()
                
                # Evaluate
                evaluation = self._evaluate_single_turn(test_case, result, start_time, end_time)
                category_results.append(evaluation)
                
            results[category] = category_results
            
        return results
    
    def run_multi_turn_benchmark(self, test_conversations: Dict[str, List[List[str]]]) -> Dict:
        """Run multi-turn evaluation"""
        
        results = {}
        
        for category, conversations in test_conversations.items():
            category_results = []
            
            for conversation in conversations:
                print(f"Testing conversation: {conversation}")
                
                # Run full conversation
                session_id = self.system.start_session()
                conversation_result = []
                
                for turn_idx, user_input in enumerate(conversation):
                    start_time = datetime.now()
                    result = self.system.process_turn(session_id, user_input)
                    end_time = datetime.now()
                    
                    result['response_time'] = (end_time - start_time).total_seconds()
                    conversation_result.append(result)
                
                # Evaluate conversation
                evaluation = self._evaluate_conversation(conversation, conversation_result)
                category_results.append(evaluation)
                
            results[category] = category_results
            
        return results
    
    def _evaluate_single_turn(self, test_case: str, result: Dict, start_time, end_time) -> Dict:
        """Evaluate a single turn"""
        
        ground_truth = GROUND_TRUTH.get(test_case, {})
        
        evaluation = {
            'test_case': test_case,
            'action': result['action'],
            'confidence': result['confidence'],
            'response_time': (end_time - start_time).total_seconds(),
            'category': ground_truth.get('category', 'unknown')
        }
        
        # Decision accuracy
        expected_action = ground_truth.get('expected_action')
        if expected_action:
            evaluation['decision_correct'] = (result['action'] == expected_action)
        else:
            evaluation['decision_correct'] = None
        
        # Confidence calibration
        expected_range = ground_truth.get('confidence_range')
        if expected_range:
            conf_in_range = expected_range[0] <= result['confidence'] <= expected_range[1]
            evaluation['confidence_appropriate'] = conf_in_range
        else:
            evaluation['confidence_appropriate'] = None
        
        # Target evaluation (for ASK actions)
        if result['action'] == 'ASK':
            expected_targets = ground_truth.get('expected_targets', [])
            target = result.get('internal_state', {}).get('decision', {}).get('target')
            evaluation['target_appropriate'] = target in expected_targets if expected_targets else None
        
        return evaluation
    
    def _evaluate_conversation(self, conversation: List[str], results: List[Dict]) -> Dict:
        """Evaluate a full conversation"""
        
        conversation_key = " -> ".join(conversation)
        ground_truth = GROUND_TRUTH.get(conversation_key, {})
        
        evaluation = {
            'conversation': conversation,
            'num_turns': len(results),
            'final_action': results[-1]['action'] if results else None,
            'final_confidence': results[-1]['confidence'] if results else 0,
            'total_response_time': sum(r.get('response_time', 0) for r in results)
        }
        
        # Efficiency metrics
        max_turns = ground_truth.get('max_turns')
        if max_turns:
            evaluation['within_turn_limit'] = len(results) <= max_turns
        
        # Question tracking
        questions_asked = sum(1 for r in results if r['action'] == 'ASK')
        evaluation['questions_asked'] = questions_asked
        
        # Repetitive question detection
        asked_targets = [r.get('internal_state', {}).get('decision', {}).get('target') 
                        for r in results if r['action'] == 'ASK']
        evaluation['repetitive_questions'] = len(asked_targets) - len(set(asked_targets))
        
        # Resolution check
        final_result = results[-1] if results else None
        if final_result:
            resolved = (final_result['action'] == 'ANSWER' and final_result['confidence'] > 0.6)
            evaluation['conversation_resolved'] = resolved
        
        return evaluation
    
    def generate_report(self, results: Dict) -> Dict:
        """Generate comprehensive evaluation report"""
        
        all_evaluations = []
        for category_results in results.values():
            all_evaluations.extend(category_results)
        
        if not all_evaluations:
            return {"error": "No evaluation results"}
        
        # Aggregate metrics
        metrics = {}
        
        # Decision accuracy
        decision_correct = [e['decision_correct'] for e in all_evaluations if e['decision_correct'] is not None]
        metrics['decision_accuracy'] = np.mean(decision_correct) if decision_correct else 0
        
        # Confidence calibration  
        conf_appropriate = [e['confidence_appropriate'] for e in all_evaluations if e['confidence_appropriate'] is not None]
        metrics['confidence_calibration'] = np.mean(conf_appropriate) if conf_appropriate else 0
        
        # Performance
        response_times = [e['response_time'] for e in all_evaluations if 'response_time' in e]
        metrics['avg_response_time'] = np.mean(response_times) if response_times else 0
        
        # Confidence analysis
        confidences = [e['confidence'] for e in all_evaluations]
        metrics['avg_confidence'] = np.mean(confidences) if confidences else 0
        metrics['confidence_std'] = np.std(confidences) if confidences else 0
        
        # Category breakdown
        category_stats = {}
        for category, category_results in results.items():
            if category_results:
                category_accuracies = [e['decision_correct'] for e in category_results if e['decision_correct'] is not None]
                category_stats[category] = {
                    'count': len(category_results),
                    'accuracy': np.mean(category_accuracies) if category_accuracies else 0,
                    'avg_confidence': np.mean([e['confidence'] for e in category_results])
                }
        
        return {
            'summary_metrics': metrics,
            'category_breakdown': category_stats,
            'detailed_results': results,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_results(self, results: Dict, filename: str = None):
        """Save results to file"""
        if filename is None:
            filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filename}")

def run_quick_test():
    """Quick test with a few cases"""
    
    # This would use the actual multi-turn system
    # system = MultiTurnSystem(pipeline)
    # evaluator = BenchmarkEvaluator(system)
    
    # Quick single-turn test
    quick_tests = {
        'identification': ["I'm looking for a spy movie with Pierce Brosnan from the 1990s"],
        'verification': ["Did Skyfall win any awards?"],
        'ambiguous': ["Bond movie"]
    }
    
    print("Quick test framework ready!")
    print("Test cases:", quick_tests)
    
    # results = evaluator.run_single_turn_benchmark(quick_tests)
    # report = evaluator.generate_report(results)
    # evaluator.save_results(report)
    
    return quick_tests

# %%
if __name__ == "__main__":
    print("Benchmark framework created!")
    print("Categories:", list(BENCHMARK_CATEGORIES.keys()))
    print("Ground truth entries:", len(GROUND_TRUTH))
    
    # Run quick test
    quick_test_results = run_quick_test()

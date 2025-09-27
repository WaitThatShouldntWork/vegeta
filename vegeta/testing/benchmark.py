"""
Benchmarking framework for VEGETA system evaluation
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
from datetime import datetime
import logging

from .test_cases import BENCHMARK_CATEGORIES, GROUND_TRUTH
from ..core.system import VegetaSystem
from ..core.exceptions import VegetaError

logger = logging.getLogger(__name__)

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

class BenchmarkRunner:
    """Runs benchmarks against the VEGETA system"""

    def __init__(self, system: VegetaSystem, verbose: bool = False):
        self.system = system
        self.results = []
        self.verbose = verbose
    
    def run_minimal_benchmark(self) -> Dict:
        """Run a minimal benchmark with just one test case for validation"""

        logger.info("Running minimal benchmark with 1 test case")

        # Use one simple test case
        test_case = "I want action movies similar to Heat"

        if self.verbose:
            print("\n🔍 VERBOSE MODE ENABLED")
            print("=" * 50)
            print(f"📝 Test Query: {test_case}")
            print("=" * 50)

        try:
            # Run system
            start_time = datetime.now()

            if self.verbose:
                print("🚀 Starting system session...")

            session_id = self.system.start_session()

            if self.verbose:
                print(f"📋 Session ID: {session_id}")
                print("⚡ Processing query...")

            result = self.system.process_query(session_id, test_case)
            end_time = datetime.now()

            if self.verbose:
                print("\n📊 QUERY PROCESSING COMPLETE")
                print("-" * 30)
                print(f"⏱️  Total processing time: {(end_time - start_time).total_seconds():.2f}s")
                print(f"🎯 Action: {result.action}")
                print(f"🎚️  Confidence: {result.confidence:.3f}")
                print(f"💬 Response: {result.content[:100]}{'...' if len(result.content) > 100 else ''}")
                print("-" * 30)
            
            # Convert VegetaResponse to dict for evaluation
            result_dict = {
                'action': result.action,
                'content': result.content,
                'confidence': result.confidence,
                'session_id': result.session_id,
                'response_time': (end_time - start_time).total_seconds(),
                'turn_index': 0
            }
            
            # Evaluate the result
            evaluation = self._evaluate_single_turn(test_case, result_dict, start_time, end_time)
            
            logger.info(f"✓ Minimal test completed in {evaluation['response_time']:.1f}s")
            
            # Return simplified results
            return {
                'benchmark_type': 'minimal',
                'test_case': test_case,
                'result': result_dict,
                'evaluation': evaluation,
                'summary': {
                    'total_tests': 1,
                    'successful_tests': 1 if evaluation.get('success', True) else 0,
                    'avg_response_time': evaluation['response_time'],
                    'system_working': True
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Minimal benchmark failed: {e}")
            return {
                'benchmark_type': 'minimal', 
                'test_case': test_case,
                'error': str(e),
                'summary': {
                    'total_tests': 1,
                    'successful_tests': 0,
                    'system_working': False
                },
                'timestamp': datetime.now().isoformat()
            }
        
    def run_single_turn_benchmark(self, test_cases: Dict[str, List[str]] = None) -> Dict:
        """Run single-turn evaluation"""
        
        if test_cases is None:
            test_cases = {k: v['single_turn'] for k, v in BENCHMARK_CATEGORIES.items()}
        
        results = {}
        
        for category, cases in test_cases.items():
            if self.verbose:
                print(f"\n📂 CATEGORY: {category}")
                print(f"📊 Running {len(cases)} test cases")
                print("-" * 40)

            logger.info(f"Running {category} category with {len(cases)} test cases")
            category_results = []

            for i, test_case in enumerate(cases):
                if self.verbose:
                    print(f"\n🎯 Test {i+1}/{len(cases)}: {test_case}")

                logger.info(f"Testing: {test_case}")

                try:
                    # Run system
                    start_time = datetime.now()

                    if self.verbose:
                        print("🚀 Starting session...")

                    session_id = self.system.start_session()
                    result = self.system.process_query(session_id, test_case)
                    end_time = datetime.now()

                    if self.verbose:
                        print("\n📊 RESULT:")
                        print(f"  🎯 Action: {result.action}")
                        print(f"  🎚️  Confidence: {result.confidence:.3f}")
                        print(f"  ⏱️  Response Time: {(end_time - start_time).total_seconds():.2f}s")
                        if result.target:
                            print(f"  🎯 Target: {result.target}")
                        if result.reasoning:
                            print(f"  🧠 Reasoning: {result.reasoning[:100]}{'...' if len(result.reasoning) > 100 else ''}")
                        print("-" * 30)
                    
                    # Convert VegetaResponse to dict for evaluation
                    result_dict = {
                        'action': result.action,
                        'target': result.target,
                        'content': result.content,
                        'confidence': result.confidence,
                        'reasoning': result.reasoning,
                        'response': result.content  # For compatibility
                    }
                    
                    # Evaluate
                    evaluation = self._evaluate_single_turn(test_case, result_dict, start_time, end_time)
                    category_results.append(evaluation)
                    
                except Exception as e:
                    logger.error(f"Test case failed: {test_case}, Error: {e}")
                    category_results.append({
                        'test_case': test_case,
                        'error': str(e),
                        'category': category
                    })
                
            results[category] = category_results
            
        return results
    
    def run_multi_turn_benchmark(self, test_conversations: Dict[str, List[List[str]]] = None) -> Dict:
        """Run multi-turn evaluation"""
        
        if test_conversations is None:
            test_conversations = {k: v['multi_turn'] for k, v in BENCHMARK_CATEGORIES.items()}
        
        results = {}
        
        for category, conversations in test_conversations.items():
            if self.verbose:
                print(f"\n🔄 MULTI-TURN CATEGORY: {category}")
                print(f"💬 Testing {len(conversations)} conversations")
                print("=" * 50)

            logger.info(f"Running {category} multi-turn with {len(conversations)} conversations")
            category_results = []

            for conv_idx, conversation in enumerate(conversations):
                if self.verbose:
                    print(f"\n📝 Conversation {conv_idx+1}/{len(conversations)}: {conversation}")

                logger.info(f"Testing conversation: {conversation}")

                try:
                    # Run full conversation
                    if self.verbose:
                        print("🚀 Starting new session...")

                    session_id = self.system.start_session()
                    conversation_result = []
                    
                    for turn_idx, user_input in enumerate(conversation):
                        if self.verbose:
                            print(f"\n  🔄 Turn {turn_idx+1}/{len(conversation)}: \"{user_input}\"")

                        start_time = datetime.now()
                        result = self.system.process_query(session_id, user_input)
                        end_time = datetime.now()

                        if self.verbose:
                            print(f"    📊 Action: {result.action} (conf: {result.confidence:.3f})")
                            print(f"    ⏱️  Time: {(end_time - start_time).total_seconds():.2f}s")
                            if result.reasoning:
                                print(f"    🧠 Reasoning: {result.reasoning[:80]}{'...' if len(result.reasoning) > 80 else ''}")
                        
                        # Convert VegetaResponse to dict and add metadata
                        result_dict = {
                            'action': result.action,
                            'target': result.target,
                            'content': result.content,
                            'confidence': result.confidence,
                            'reasoning': result.reasoning,
                            'session_id': result.session_id,
                            'turn_id': result.turn_id,
                            'response_time': (end_time - start_time).total_seconds(),
                            'turn_index': turn_idx
                        }
                        conversation_result.append(result_dict)
                    
                    # Evaluate conversation
                    evaluation = self._evaluate_conversation(conversation, conversation_result)

                    if self.verbose:
                        final_action = evaluation.get('final_action', 'UNKNOWN')
                        final_confidence = evaluation.get('final_confidence', 0.0)
                        num_turns = evaluation.get('num_turns', len(conversation))
                        resolved = evaluation.get('conversation_resolved', False)

                        print(f"\n  📈 CONVERSATION SUMMARY:")
                        print(f"    🎯 Final Action: {final_action}")
                        print(f"    🎚️  Final Confidence: {final_confidence:.3f}")
                        print(f"    🔄 Total Turns: {num_turns}")
                        print(f"    ✅ Resolved: {'Yes' if resolved else 'No'}")
                        print("=" * 50)

                    category_results.append(evaluation)
                    
                except Exception as e:
                    logger.error(f"Conversation failed: {conversation}, Error: {e}")
                    category_results.append({
                        'conversation': conversation,
                        'error': str(e),
                        'category': category
                    })
                
            results[category] = category_results
            
        return results
    
    def run_quick_test(self, num_cases_per_category: int = 2) -> Dict:
        """Run a quick subset of tests for rapid validation"""

        # Single-turn tests (existing logic)
        quick_single_tests = {}
        for category, subcategories in BENCHMARK_CATEGORIES.items():
            # Take first N single-turn cases
            single_turn_subset = subcategories['single_turn'][:num_cases_per_category]
            quick_single_tests[category] = single_turn_subset

        # Add specific multi-turn tests as requested
        quick_multi_tests = {
            'music_rights_verification': [
                ["I need to verify music rights", "For the film Skyfall", "Check composer and territory"]
            ],
            'recommendation': [
                ["What should I watch?", "Similar to Casino Royale", "Spy thriller"]
            ]
        }

        total_single = sum(len(tests) for tests in quick_single_tests.values())
        total_multi = sum(len(tests) for tests in quick_multi_tests.values())

        if self.verbose:
            print("\n🔍 VERBOSE QUICK BENCHMARK")
            print("=" * 50)
            print(f"📊 Single-turn tests: {total_single}")
            print(f"🔄 Multi-turn conversations: {total_multi}")
            print(f"📈 Total test cases: {total_single + total_multi}")
            print("=" * 50)

        logger.info(f"Running quick test: {total_single} single-turn + {total_multi} multi-turn tests")

        # Run both single-turn and multi-turn tests
        single_results = self.run_single_turn_benchmark(quick_single_tests)
        multi_results = self.run_multi_turn_benchmark(quick_multi_tests)

        return {
            "single_turn": single_results,
            "multi_turn": multi_results
        }
    
    def _evaluate_single_turn(self, test_case: str, result: Dict, start_time, end_time) -> Dict:
        """Evaluate a single turn"""
        
        ground_truth = GROUND_TRUTH.get(test_case, {})
        
        evaluation = {
            'test_case': test_case,
            'action': result.get('action', 'UNKNOWN'),
            'confidence': result.get('confidence', 0.0),
            'response_time': (end_time - start_time).total_seconds(),
            'category': ground_truth.get('category', 'unknown'),
            'response_text': result.get('response', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        # Decision accuracy
        expected_action = ground_truth.get('expected_action')
        if expected_action:
            evaluation['decision_correct'] = (result.get('action') == expected_action)
        else:
            evaluation['decision_correct'] = None
        
        # Confidence calibration
        expected_range = ground_truth.get('confidence_range')
        if expected_range:
            conf_in_range = expected_range[0] <= result.get('confidence', 0) <= expected_range[1]
            evaluation['confidence_appropriate'] = conf_in_range
        else:
            evaluation['confidence_appropriate'] = None
        
        # Target evaluation (for ASK actions)
        if result.get('action') == 'ASK':
            expected_targets = ground_truth.get('expected_targets', [])
            # This would need to be extracted from the system's internal state
            # For now, we'll leave it as None
            evaluation['target_appropriate'] = None
        
        return evaluation
    
    def _evaluate_conversation(self, conversation: List[str], results: List[Dict]) -> Dict:
        """Evaluate a full conversation"""
        
        conversation_key = " -> ".join(conversation)
        ground_truth = GROUND_TRUTH.get(conversation_key, {})
        
        evaluation = {
            'conversation': conversation,
            'num_turns': len(results),
            'final_action': results[-1].get('action') if results else None,
            'final_confidence': results[-1].get('confidence', 0) if results else 0,
            'total_response_time': sum(r.get('response_time', 0) for r in results),
            'timestamp': datetime.now().isoformat()
        }
        
        # Efficiency metrics
        max_turns = ground_truth.get('max_turns')
        if max_turns:
            evaluation['within_turn_limit'] = len(results) <= max_turns
        
        # Question tracking
        questions_asked = sum(1 for r in results if r.get('action') == 'ASK')
        evaluation['questions_asked'] = questions_asked
        
        # Repetitive question detection (simplified)
        asked_responses = [r.get('response', '') for r in results if r.get('action') == 'ASK']
        evaluation['repetitive_questions'] = len(asked_responses) - len(set(asked_responses))
        
        # Resolution check
        final_result = results[-1] if results else None
        if final_result:
            resolved = (final_result.get('action') == 'ANSWER' and final_result.get('confidence', 0) > 0.6)
            evaluation['conversation_resolved'] = resolved
        
        return evaluation
    
    def generate_report(self, results: Dict) -> Dict:
        """Generate comprehensive evaluation report"""
        
        # Handle minimal benchmark format
        if results.get('benchmark_type') == 'minimal':
            evaluation = results.get('evaluation', {})
            if evaluation:
                return {
                    'benchmark_type': 'minimal',
                    'total_cases': 1,
                    'successful_cases': 1 if evaluation.get('success', True) else 0,
                    'avg_response_time': evaluation.get('response_time', 0),
                    'decision_accuracy': 1.0 if evaluation.get('decision_correct', False) else 0.0,
                    'confidence_calibration': 1.0 if evaluation.get('confidence_appropriate', False) else 0.0,
                    'system_working': results.get('summary', {}).get('system_working', True),
                    'test_case': results.get('test_case', ''),
                    'result_action': results.get('result', {}).get('action', 'unknown'),
                    'result_confidence': results.get('result', {}).get('confidence', 0),
                    'evaluation_details': evaluation
                }
            else:
                return {
                    'benchmark_type': 'minimal',
                    'error': results.get('error', 'Unknown error'),
                    'system_working': False
                }
        
        # Handle regular benchmark formats
        all_evaluations = []
        for category_results in results.values():
            all_evaluations.extend(category_results)
        
        if not all_evaluations:
            return {"error": "No evaluation results"}
        
        # Filter out error cases for metrics
        valid_evaluations = [e for e in all_evaluations if 'error' not in e]

        if not valid_evaluations:
            return {"error": "No valid evaluation results", "total_cases": len(all_evaluations)}

        # Separate single-turn and multi-turn evaluations
        single_turn_evaluations = [e for e in valid_evaluations if 'conversation' not in e]
        multi_turn_evaluations = [e for e in valid_evaluations if 'conversation' in e]

        # Aggregate metrics
        metrics = {}

        # Decision accuracy (only for single-turn evaluations)
        if single_turn_evaluations:
            decision_correct = [e['decision_correct'] for e in single_turn_evaluations if e.get('decision_correct') is not None]
            metrics['decision_accuracy'] = np.mean(decision_correct) if decision_correct else 0

            # Confidence calibration
            conf_appropriate = [e['confidence_appropriate'] for e in single_turn_evaluations if e.get('confidence_appropriate') is not None]
            metrics['confidence_calibration'] = np.mean(conf_appropriate) if conf_appropriate else 0
        
        # Performance
        response_times = [e['response_time'] for e in valid_evaluations if 'response_time' in e]
        metrics['avg_response_time'] = np.mean(response_times) if response_times else 0
        metrics['max_response_time'] = np.max(response_times) if response_times else 0
        metrics['min_response_time'] = np.min(response_times) if response_times else 0
        
        # Confidence analysis (only for single-turn evaluations)
        if single_turn_evaluations:
            confidences = [e['confidence'] for e in single_turn_evaluations]
            metrics['avg_confidence'] = np.mean(confidences) if confidences else 0
            metrics['confidence_std'] = np.std(confidences) if confidences else 0

        # Action distribution (only for single-turn evaluations)
        if single_turn_evaluations:
            actions = [e['action'] for e in single_turn_evaluations]
            action_counts = {}
            for action in set(actions):
                action_counts[action] = actions.count(action)
            metrics['action_distribution'] = action_counts

        # Multi-turn specific metrics
        if multi_turn_evaluations:
            metrics['multi_turn_summary'] = {
                'total_conversations': len(multi_turn_evaluations),
                'avg_questions_per_conversation': np.mean([e['questions_asked'] for e in multi_turn_evaluations]),
                'conversation_resolution_rate': np.mean([e.get('conversation_resolved', False) for e in multi_turn_evaluations]),
                'avg_conversation_length': np.mean([e['num_turns'] for e in multi_turn_evaluations])
            }
        
        # Category breakdown
        category_stats = {}
        for category, category_results in results.items():
            valid_category_results = [e for e in category_results if 'error' not in e]
            if valid_category_results:
                # Handle both single-turn and multi-turn evaluations
                single_turn_results = [e for e in valid_category_results if 'conversation' not in e]
                multi_turn_results = [e for e in valid_category_results if 'conversation' in e]

                category_info = {
                    'count': len(valid_category_results),
                    'errors': len(category_results) - len(valid_category_results),
                    'single_turn_count': len(single_turn_results),
                    'multi_turn_count': len(multi_turn_results)
                }

                # Single-turn metrics
                if single_turn_results:
                    category_accuracies = [e['decision_correct'] for e in single_turn_results if e.get('decision_correct') is not None]
                    category_info['accuracy'] = np.mean(category_accuracies) if category_accuracies else 0
                    category_info['avg_confidence'] = np.mean([e['confidence'] for e in single_turn_results])
                    category_info['avg_response_time'] = np.mean([e['response_time'] for e in single_turn_results])

                # Multi-turn metrics
                if multi_turn_results:
                    resolved_count = sum(1 for e in multi_turn_results if e.get('conversation_resolved', False))
                    category_info['conversation_resolution_rate'] = resolved_count / len(multi_turn_results)
                    category_info['avg_questions_per_conversation'] = np.mean([e['questions_asked'] for e in multi_turn_results])

                category_stats[category] = category_info
        
        report = {
            'summary_metrics': metrics,
            'category_breakdown': category_stats,
            'detailed_results': results,
            'timestamp': datetime.now().isoformat(),
            'total_cases': len(all_evaluations),
            'valid_cases': len(valid_evaluations),
            'error_cases': len(all_evaluations) - len(valid_evaluations)
        }

        # Add verbose summary output
        if self.verbose:
            self._print_verbose_summary(metrics, category_stats, single_turn_evaluations, multi_turn_evaluations)

        return report

    def _print_verbose_summary(self, metrics: Dict, category_stats: Dict,
                              single_turn_evaluations: List, multi_turn_evaluations: List):
        """Print detailed verbose summary of benchmark results"""

        print("\n🎯 VERBOSE BENCHMARK SUMMARY")
        print("=" * 80)

        # Overall metrics
        print("📈 OVERALL PERFORMANCE:")
        print(f"   📊 Total Test Cases: {metrics.get('total_cases', 0)}")
        if 'decision_accuracy' in metrics:
            print(f"   ✅ Decision Accuracy: {metrics['decision_accuracy']:.1%}")
        if 'confidence_calibration' in metrics:
            print(f"   🎯 Confidence Calibration: {metrics['confidence_calibration']:.1%}")
        if 'avg_response_time' in metrics:
            print(f"   ⏱️  Avg Response Time: {metrics['avg_response_time']:.2f}s")
        if 'conversation_resolution_rate' in metrics:
            print(f"   🔄 Conversation Resolution: {metrics['conversation_resolution_rate']:.1%}")

        # Category breakdown
        print("\n📂 CATEGORY BREAKDOWN:")
        print("-" * 50)

        for category, stats in category_stats.items():
            print(f"\n🏷️  {category.upper()}:")
            print(f"   📊 Total: {stats['count']} tests")
            if 'accuracy' in stats:
                print(f"   ✅ Accuracy: {stats['accuracy']:.1%}")
            if 'avg_confidence' in stats:
                print(f"   🎚️  Avg Confidence: {stats['avg_confidence']:.3f}")
            if 'avg_response_time' in stats:
                print(f"   ⏱️  Avg Time: {stats['avg_response_time']:.2f}s")
            if 'conversation_resolution_rate' in stats:
                print(f"   🔄 Resolution Rate: {stats['conversation_resolution_rate']:.1%}")
            if stats['errors'] > 0:
                print(f"   ❌ Errors: {stats['errors']}")

        # Detailed action distribution
        if single_turn_evaluations:
            print("\n🎯 ACTION DISTRIBUTION (Single-turn):")
            print("-" * 40)
            actions = {}
            for e in single_turn_evaluations:
                action = e.get('action', 'UNKNOWN')
                actions[action] = actions.get(action, 0) + 1

            for action, count in sorted(actions.items(), key=lambda x: x[1], reverse=True):
                percentage = count / len(single_turn_evaluations) * 100
                print(".1f")

        # Worst performing tests
        if single_turn_evaluations:
            print("\n📉 WORST PERFORMING TESTS:")
            print("-" * 30)
            # Sort by decision correctness and confidence
            sorted_tests = sorted(
                [e for e in single_turn_evaluations if e.get('decision_correct') is not None],
                key=lambda x: (not x.get('decision_correct', True), -x.get('confidence', 0))
            )

            for i, test in enumerate(sorted_tests[:5]):  # Top 5 worst
                status = "❌" if not test.get('decision_correct', True) else "✅"
                print("2d")

        print("\n🎉 VERBOSE BENCHMARK COMPLETE!")
        print("=" * 80)

    def save_results(self, results: Dict, filename: str = None):
        """Save results to file"""
        if filename is None:
            filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        
        def clean_for_json(data):
            if isinstance(data, dict):
                return {k: clean_for_json(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_for_json(v) for v in data]
            else:
                return convert_numpy(data)
        
        clean_results = clean_for_json(results)
        
        with open(filename, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
        return filename

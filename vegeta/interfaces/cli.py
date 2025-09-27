"""
Command-line interface for VEGETA system
"""

import argparse
import logging
import sys
import time
from typing import Optional

from ..core.system import VegetaSystem
from ..core.config import Config
from ..core.exceptions import VegetaError
from ..testing.benchmark import BenchmarkRunner

def setup_logging(debug: bool = False, verbose: bool = False):
    """Setup logging configuration with optional debug control"""
    import logging
    
    # Base logging level
    if debug:
        log_level = logging.DEBUG
    elif verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Always suppress noisy third-party loggers unless debug is explicitly enabled
    if not debug:
        # Suppress Neo4j debug logs (most verbose)
        logging.getLogger('neo4j').setLevel(logging.WARNING)
        logging.getLogger('neo4j.pool').setLevel(logging.WARNING) 
        logging.getLogger('neo4j.io').setLevel(logging.WARNING)
        
        # Suppress HTTP debug logs
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        
        # Keep our own loggers at appropriate level
        logging.getLogger('vegeta').setLevel(log_level)
        logging.getLogger('VegetaSystem').setLevel(log_level)

# Setup colored output if available
try:
    from colorama import init, Fore, Style
    init()
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    class Fore:
        RED = GREEN = BLUE = YELLOW = CYAN = MAGENTA = ""
    class Style:
        BRIGHT = RESET_ALL = ""

class VegetaCLI:
    """
    Command-line interface for VEGETA system
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = Config(config_path)
        self.system = None
        self.current_session = None
        
        # Setup logging
        log_level = getattr(logging, self.config.get('logging.level', 'INFO').upper())
        logging.basicConfig(
            level=log_level,
            format=self.config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger = logging.getLogger(__name__)
    
    def start_system(self):
        """Initialize the VEGETA system"""
        try:
            self.system = VegetaSystem(self.config)
            self.logger.info("✓ VEGETA system initialized successfully")
            return True
        except VegetaError as e:
            self._print_error(f"Failed to initialize VEGETA system: {e}")
            return False
        except Exception as e:
            self._print_error(f"Unexpected error during initialization: {e}")
            return False
    
    def run_interactive(self, verbose: bool = False):
        """Run interactive 20-questions mode"""
        
        if not self.start_system():
            return 1
        
        self._print_header("VEGETA Interactive Mode")
        if verbose:
            print("🔍 VERBOSE MODE ENABLED")
            print("=" * 40)
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'help' for available commands")
        print("Type 'new' to start a new session")
        print()
        
        # Start initial session
        self.current_session = self.system.start_session()
        self._print_success(f"Started session: {self.current_session}")
        
        try:
            while True:
                try:
                    # Get user input
                    user_input = input(f"{Fore.CYAN}You: {Style.RESET_ALL}").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle commands
                    if user_input.lower() in ['quit', 'exit']:
                        break
                    elif user_input.lower() == 'help':
                        self._show_help()
                        continue
                    elif user_input.lower() == 'new':
                        self.current_session = self.system.start_session()
                        self._print_success(f"Started new session: {self.current_session}")
                        continue
                    elif user_input.lower() == 'status':
                        self._show_status()
                        continue
                    
                    # Process query
                    self._process_query(user_input, verbose)
                    
                except KeyboardInterrupt:
                    print("\nUse 'quit' or 'exit' to end the session.")
                    continue
                except Exception as e:
                    self._print_error(f"Error processing query: {e}")
                    continue
        
        finally:
            if self.system:
                self.system.close()
                self._print_success("VEGETA system closed")
        
        return 0
    
    def run_single_query(self, query: str, verbose: bool = False):
        """Run a single query and exit"""
        
        if not self.start_system():
            return 1
        
        try:
            # Start session and process query
            session_id = self.system.start_session()
            self._print_header(f"Processing Query: '{query}'")

            if verbose:
                print("⚡ PROCESSING QUERY...")
                print(f"  📝 Query: {query}")
                print(f"  🆔 Session: {session_id}")
                print("-" * 40)

            response = self.system.process_query(session_id, query, include_internal_state=verbose)

            if verbose:
                print("\n📊 PROCESSING COMPLETE")
                print("-" * 40)

            # Display response
            self._display_response(response)

            # Show internal state if available and verbose mode enabled
            if response.internal_state and verbose:
                self._display_internal_state(response.internal_state)
            
        except VegetaError as e:
            self._print_error(f"Query processing failed: {e}")
            return 1
        except Exception as e:
            self._print_error(f"Unexpected error: {e}")
            return 1
        finally:
            if self.system:
                self.system.close()
        
        return 0
    
    def _process_query(self, query: str, verbose: bool = False):
        """Process a single query in interactive mode"""
        try:
            if verbose:
                print("\n⚡ PROCESSING QUERY...")
                print(f"  📝 Query: {query}")
                print(f"  🆔 Session: {self.current_session}")
                print("-" * 40)
                start_time = time.time()

            response = self.system.process_query(self.current_session, query)

            if verbose:
                end_time = time.time()
                print("\n📊 PROCESSING COMPLETE")
                print(".2f")
                print("-" * 40)

            self._display_response(response)

            # Handle ASK responses - get user feedback
            if response.action == 'ASK':
                print(f"{Fore.YELLOW}(Provide your response to help narrow down the answer){Style.RESET_ALL}")

        except VegetaError as e:
            self._print_error(f"Query processing failed: {e}")
        except Exception as e:
            self._print_error(f"Unexpected error: {e}")
    
    def _display_response(self, response):
        """Display system response"""
        
        action_colors = {
            'ASK': Fore.YELLOW,
            'ANSWER': Fore.GREEN,
            'SEARCH': Fore.BLUE
        }
        
        color = action_colors.get(response.action, Fore.WHITE)
        print(f"{color}VEGETA [{response.action}]: {Style.RESET_ALL}{response.content}")
        
        # Show confidence and reasoning
        confidence_color = Fore.GREEN if response.confidence > 0.7 else Fore.YELLOW if response.confidence > 0.4 else Fore.RED
        print(f"{Fore.CYAN}Confidence: {confidence_color}{response.confidence:.1%}{Style.RESET_ALL}")
        
        if response.reasoning:
            print(f"{Fore.MAGENTA}Reasoning: {Style.RESET_ALL}{response.reasoning}")
        
        print()  # Add spacing
    
    def _display_internal_state(self, internal_state):
        """Display internal state for debugging"""
        print(f"{Fore.CYAN}--- Internal State ---{Style.RESET_ALL}")
        
        # Show top candidates
        candidates = internal_state.get('candidates', [])
        if candidates:
            print(f"{Fore.CYAN}Top Candidates:{Style.RESET_ALL}")
            for i, candidate in enumerate(candidates[:3]):
                name = candidate.get('entity_name') or candidate.get('anchor_name', 'unknown')
                score = candidate.get('retrieval_score', 0.0)
                print(f"  {i+1}. {name} (score: {score:.3f})")
        
        # Show decision info
        decision = internal_state.get('decision', {})
        if decision:
            print(f"{Fore.CYAN}Decision Details:{Style.RESET_ALL}")
            print(f"  Action: {decision.get('action', 'unknown')}")
            print(f"  Target: {decision.get('target', 'unknown')}")
            print(f"  Confidence: {decision.get('confidence', 0.0):.3f}")
        
        print()
    
    def _show_help(self):
        """Show help information"""
        print(f"{Fore.CYAN}Available Commands:{Style.RESET_ALL}")
        print("  help     - Show this help message")
        print("  new      - Start a new session")
        print("  status   - Show current session status")
        print("  quit     - Exit the program")
        print("  exit     - Exit the program")
        print()
        print(f"{Fore.CYAN}Usage:{Style.RESET_ALL}")
        print("  Simply type your question or statement and press Enter.")
        print("  The system will ask clarifying questions to help narrow down the answer.")
        print()
    
    def _show_status(self):
        """Show current session status"""
        if self.current_session:
            summary = self.system.get_session_summary(self.current_session)
            print(f"{Fore.CYAN}Session Status:{Style.RESET_ALL}")
            print(f"  Session ID: {self.current_session}")
            print(f"  Turn Count: {summary.get('turn_count', 0)}")
            print(f"  Duration: {summary.get('session_duration', 0):.1f} seconds")
            print(f"  Summary: {summary.get('conversation_summary', 'None')}")
        else:
            print(f"{Fore.RED}No active session{Style.RESET_ALL}")
        print()
    
    def _print_header(self, text: str):
        """Print a header"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{text}")
        print(f"{'='*60}{Style.RESET_ALL}")
    
    def _print_success(self, text: str):
        """Print success message"""
        print(f"{Fore.GREEN}[SUCCESS] {text}{Style.RESET_ALL}")
    
    def _print_error(self, text: str):
        """Print error message"""
        print(f"{Fore.RED}[ERROR] {text}{Style.RESET_ALL}")
    
    def run_benchmark(self, benchmark_type: str = "quick", save_results: bool = True, verbose: bool = False):
        """Run benchmark evaluation"""
        
        if not self.start_system():
            return 1
        
        try:
            self._print_header(f"Running {benchmark_type.upper()} Benchmark")
            
            # Create benchmark runner with verbose setting
            runner = BenchmarkRunner(self.system, verbose=verbose)
            
            # Run appropriate benchmark
            if benchmark_type == "minimal":
                print("Running minimal benchmark (1 test case)")
                results = runner.run_minimal_benchmark()
            elif benchmark_type == "quick":
                print("Running quick benchmark (2 cases per category)")
                results = runner.run_quick_test(num_cases_per_category=2)
            elif benchmark_type == "single":
                print("Running single-turn benchmark")
                results = runner.run_single_turn_benchmark()
            elif benchmark_type == "multi":
                print("Running multi-turn benchmark")
                results = runner.run_multi_turn_benchmark()
            elif benchmark_type == "full":
                print("Running full benchmark suite")
                single_results = runner.run_single_turn_benchmark()
                multi_results = runner.run_multi_turn_benchmark()
                results = {
                    "single_turn": single_results,
                    "multi_turn": multi_results
                }
            else:
                self._print_error(f"Unknown benchmark type: {benchmark_type}")
                return 1
            
            # Generate report
            if verbose:
                print("\n📊 GENERATING DETAILED EVALUATION REPORT...")
                print("=" * 60)
            else:
                print("📊 Generating evaluation report...")

            if benchmark_type == "full":
                # Generate separate reports for single and multi-turn
                single_report = runner.generate_report(single_results)
                multi_report = runner.generate_report(multi_results)
                report = {
                    "single_turn_report": single_report,
                    "multi_turn_report": multi_report,
                    "combined_summary": {
                        "total_single_cases": single_report.get("total_cases", 0),
                        "total_multi_conversations": multi_report.get("total_cases", 0),
                        "single_turn_accuracy": single_report.get("summary_metrics", {}).get("decision_accuracy", 0),
                        "multi_turn_resolution_rate": sum(1 for cat in multi_report.get("detailed_results", {}).values() 
                                                        for conv in cat if conv.get("conversation_resolved", False)) / 
                                                     max(multi_report.get("total_cases", 1), 1)
                    }
                }
            else:
                report = runner.generate_report(results)
            
            # Print summary
            self._print_header("BENCHMARK RESULTS SUMMARY")
            
            if benchmark_type == "full":
                print(f"{Fore.CYAN}Single-turn cases:{Style.RESET_ALL} {report['combined_summary']['total_single_cases']}")
                print(f"{Fore.CYAN}Multi-turn conversations:{Style.RESET_ALL} {report['combined_summary']['total_multi_conversations']}")
                print(f"{Fore.CYAN}Single-turn accuracy:{Style.RESET_ALL} {report['combined_summary']['single_turn_accuracy']:.2%}")
                print(f"{Fore.CYAN}Multi-turn resolution rate:{Style.RESET_ALL} {report['combined_summary']['multi_turn_resolution_rate']:.2%}")
            else:
                summary = report.get("summary_metrics", {})
                print(f"{Fore.CYAN}Total cases:{Style.RESET_ALL} {report.get('total_cases', 0)}")
                print(f"{Fore.CYAN}Valid cases:{Style.RESET_ALL} {report.get('valid_cases', 0)}")
                print(f"{Fore.CYAN}Error cases:{Style.RESET_ALL} {report.get('error_cases', 0)}")
                print(f"{Fore.CYAN}Decision accuracy:{Style.RESET_ALL} {summary.get('decision_accuracy', 0):.2%}")
                print(f"{Fore.CYAN}Confidence calibration:{Style.RESET_ALL} {summary.get('confidence_calibration', 0):.2%}")
                print(f"{Fore.CYAN}Average response time:{Style.RESET_ALL} {summary.get('avg_response_time', 0):.2f}s")
                print(f"{Fore.CYAN}Average confidence:{Style.RESET_ALL} {summary.get('avg_confidence', 0):.2f}")
                
                # Category breakdown
                if "category_breakdown" in report:
                    print(f"\n{Fore.CYAN}📊 Category Performance:{Style.RESET_ALL}")
                    for category, stats in report["category_breakdown"].items():
                        print(f"  {category}: {stats['accuracy']:.2%} accuracy ({stats['count']} cases)")
            
            # Save results
            if save_results:
                filename = runner.save_results(report)
                self._print_success(f"Results saved to: {filename}")

            self._print_success("Benchmark completed successfully!")

        except VegetaError as e:
            self._print_error(f"VEGETA system error: {e}")
            return 1
        except Exception as e:
            self._print_error(f"Unexpected error: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return 1
        finally:
            if self.system:
                self.system.close()

        return 0

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='VEGETA Bayesian Active Inference System')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug logging (shows Neo4j, HTTP, and internal debug logs)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Interactive mode
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive 20-questions mode')
    interactive_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output in interactive mode'
    )
    
    # Single query mode
    query_parser = subparsers.add_parser('query', help='Process a single query')
    query_parser.add_argument('text', help='Query text to process')
    query_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Benchmark mode
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmark evaluation')
    benchmark_parser.add_argument(
        '--type', 
        choices=['minimal', 'quick', 'single', 'multi', 'full'],
        default='quick',
        help='Type of benchmark to run (default: quick)'
    )
    benchmark_parser.add_argument(
        '--save', 
        action='store_true',
        default=True,
        help='Save results to file (default: True)'
    )
    benchmark_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging based on command line arguments
    setup_logging(
        debug=getattr(args, 'debug', False),
        verbose=getattr(args, 'verbose', False)
    )
    
    # Create CLI instance
    cli = VegetaCLI(args.config)
    
    # Route to appropriate mode
    if args.command == 'interactive':
        return cli.run_interactive(verbose=getattr(args, 'verbose', False))
    elif args.command == 'query':
        return cli.run_single_query(args.text, verbose=getattr(args, 'verbose', False))
    elif args.command == 'benchmark':
        return cli.run_benchmark(args.type, args.save, args.verbose)
    else:
        # Default to interactive if no command specified
        return cli.run_interactive(verbose=getattr(args, 'verbose', False))

if __name__ == '__main__':
    sys.exit(main())

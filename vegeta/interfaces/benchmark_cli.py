#!/usr/bin/env python3
"""
Benchmark CLI for VEGETA system evaluation
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

from ..core.system import VegetaSystem
from ..core.config import Config
from ..core.exceptions import VegetaError
from ..testing.benchmark import BenchmarkRunner

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def run_benchmark(config_path: Optional[str] = None, 
                 benchmark_type: str = "quick",
                 save_results: bool = True,
                 verbose: bool = False) -> bool:
    """Run benchmark evaluation"""
    
    try:
        # Setup logging
        setup_logging("DEBUG" if verbose else "INFO")
        logger = logging.getLogger(__name__)
        
        logger.info("🚀 Starting VEGETA benchmark evaluation")
        
        # Initialize system
        config = Config(config_path) if config_path else Config()
        system = VegetaSystem(config)
        
        # Validate connections
        if not system._validate_connections():
            logger.error("❌ System validation failed")
            return False
        
        logger.info("✅ System initialized and validated")
        
        # Create benchmark runner
        runner = BenchmarkRunner(system)
        
        # Run appropriate benchmark
        if benchmark_type == "quick":
            logger.info("🏃 Running quick benchmark (single + multi-turn)")
            results = runner.run_quick_test(num_cases_per_category=2)
        elif benchmark_type == "single":
            logger.info("📝 Running single-turn benchmark")
            results = runner.run_single_turn_benchmark()
        elif benchmark_type == "multi":
            logger.info("🔄 Running multi-turn benchmark")
            results = runner.run_multi_turn_benchmark()
        elif benchmark_type == "full":
            logger.info("🎯 Running full benchmark suite")
            single_results = runner.run_single_turn_benchmark()
            multi_results = runner.run_multi_turn_benchmark()
            results = {
                "single_turn": single_results,
                "multi_turn": multi_results
            }
        else:
            logger.error(f"❌ Unknown benchmark type: {benchmark_type}")
            return False

        # Generate report
        logger.info("📊 Generating evaluation report")
        if benchmark_type in ["full", "quick"]:
            # Handle combined single + multi-turn results
            if "single_turn" in results and "multi_turn" in results:
                single_report = runner.generate_report(results["single_turn"])
                multi_report = runner.generate_report(results["multi_turn"])
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
        print("\n" + "="*60)
        print("📈 BENCHMARK RESULTS SUMMARY")
        print("="*60)
        
        if benchmark_type == "full":
            print(f"Single-turn cases: {report['combined_summary']['total_single_cases']}")
            print(f"Multi-turn conversations: {report['combined_summary']['total_multi_conversations']}")
            print(f"Single-turn accuracy: {report['combined_summary']['single_turn_accuracy']:.2%}")
            print(f"Multi-turn resolution rate: {report['combined_summary']['multi_turn_resolution_rate']:.2%}")
        else:
            summary = report.get("summary_metrics", {})
            print(f"Total cases: {report.get('total_cases', 0)}")
            print(f"Valid cases: {report.get('valid_cases', 0)}")
            print(f"Error cases: {report.get('error_cases', 0)}")
            print(f"Decision accuracy: {summary.get('decision_accuracy', 0):.2%}")
            print(f"Confidence calibration: {summary.get('confidence_calibration', 0):.2%}")
            print(f"Average response time: {summary.get('avg_response_time', 0):.2f}s")
            print(f"Average confidence: {summary.get('avg_confidence', 0):.2f}")
            
            # Category breakdown
            if "category_breakdown" in report:
                print("\n📊 Category Performance:")
                for category, stats in report["category_breakdown"].items():
                    print(f"  {category}: {stats['accuracy']:.2%} accuracy ({stats['count']} cases)")
        
        # Save results
        if save_results:
            filename = runner.save_results(report)
            print(f"\n💾 Results saved to: {filename}")
        
        print("\n✅ Benchmark completed successfully!")
        return True
        
    except VegetaError as e:
        logger.error(f"❌ VEGETA system error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="VEGETA Benchmark Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Benchmark Types:
  quick   - Quick test with 2 cases per category (default)
  single  - Full single-turn benchmark
  multi   - Full multi-turn benchmark  
  full    - Complete benchmark suite (single + multi)

Examples:
  python -m vegeta.interfaces.benchmark_cli --type quick
  python -m vegeta.interfaces.benchmark_cli --type full --save --verbose
  python -m vegeta.interfaces.benchmark_cli --config custom_config.yaml --type single
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--type", 
        choices=["quick", "single", "multi", "full"],
        default="quick",
        help="Type of benchmark to run (default: quick)"
    )
    
    parser.add_argument(
        "--save", 
        action="store_true",
        help="Save results to file (default: True)"
    )
    
    parser.add_argument(
        "--no-save", 
        action="store_true",
        help="Don't save results to file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Handle save flag logic
    save_results = True
    if args.no_save:
        save_results = False
    elif args.save:
        save_results = True
    
    success = run_benchmark(
        config_path=args.config,
        benchmark_type=args.type,
        save_results=save_results,
        verbose=args.verbose
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

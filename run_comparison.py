#!/usr/bin/env python3
"""Run controlled bias comparison experiments."""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import torch
import sys
from tqdm import tqdm

from src.config import ExperimentConfig
from src.model import ModelLoader
from src.math_problems import get_all_problems
from src.comparison import (
    run_problem_comparison,
    analyze_bias_influence,
    save_comparison_results
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run controlled bias comparison experiments on math problems"
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of samples per prompt variation (default: 5)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
        help='Model name or path'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/bias_comparison',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode - use smaller model and fewer samples'
    )
    
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--problem-ids',
        nargs='+',
        help='Specific problem IDs to run (e.g., alg_1 trig_2)'
    )
    
    parser.add_argument(
        '--problem-type',
        choices=['algebra', 'trigonometry', 'all'],
        default='all',
        help='Run only algebra, trigonometry, or all problems'
    )
    
    parser.add_argument(
        '--problem-range',
        type=str,
        help='Run a range of problems (e.g., "0-2" for first 3, "3-5" for last 3)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Adjust for test mode
    if args.test:
        logger.info("Running in TEST MODE")
        args.model = "gpt2"
        args.num_samples = 2
    
    # Set verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration with fixed temperature and top-p
    config = ExperimentConfig(
        model_name=args.model,
        device='cpu' if args.cpu else 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
        cache_dir='./model_cache',
        max_new_tokens=1024,
        temperature=0.5,  # Fixed as requested
        top_p=0.95,  # Fixed as requested
        seed=42,
        output_dir=args.output_dir
    )
    
    # Log configuration
    logger.info("="*60)
    logger.info("BIAS COMPARISON EXPERIMENT")
    logger.info("="*60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Temperature: 0.5 (fixed)")
    logger.info(f"Top-p: 0.95 (fixed)")
    logger.info(f"Samples per variation: {args.num_samples}")
    logger.info(f"Output dir: {config.output_dir}")
    
    try:
        # Load model
        logger.info("\nLoading model...")
        model_loader = ModelLoader(config)
        model_loader.load()
        
        # Get problems
        all_problems = get_all_problems()
        
        # Filter by problem type first
        if args.problem_type != 'all':
            all_problems = [p for p in all_problems if p.problem_type == args.problem_type]
            logger.info(f"Filtered to {args.problem_type} problems: {len(all_problems)} problems")
        
        # Filter by range if specified
        if args.problem_range:
            try:
                start, end = map(int, args.problem_range.split('-'))
                all_problems = all_problems[start:end+1]
                logger.info(f"Using problem range {start}-{end}: {len(all_problems)} problems")
            except (ValueError, IndexError):
                logger.warning(f"Invalid range format '{args.problem_range}', ignoring")
        
        # Filter by specific IDs (overrides other filters)
        if args.problem_ids:
            all_problems = [p for p in get_all_problems() if p.id in args.problem_ids]
            logger.info(f"Running specific problems: {args.problem_ids}")
        
        if not all_problems:
            logger.error("No problems selected! Check your filters.")
            sys.exit(1)
        
        logger.info(f"Running {len(all_problems)} problems: {[p.id for p in all_problems]}")
        
        # Run comparisons
        logger.info("\n" + "="*60)
        logger.info("STARTING EXPERIMENTS")
        logger.info("="*60)
        
        all_results = []
        all_analyses = []
        
        # Main progress bar for all problems
        total_generations = len(all_problems) * 5 * args.num_samples  # problems × variations × samples
        logger.info(f"\nTotal generations to run: {total_generations}")
        
        for problem in tqdm(all_problems, desc="Problems", position=0):
            # Run comparison for this problem
            results = run_problem_comparison(
                model_loader,
                problem,
                config,
                num_samples=args.num_samples
            )
            all_results.append(results)
            
            # Analyze bias influence
            analysis = analyze_bias_influence(results)
            all_analyses.append(analysis)
            
            # Log quick summary
            logger.info(f"  Completed {problem.id}")
            if 'neutral_accuracy' in analysis:
                logger.info(f"    Neutral accuracy: {analysis['neutral_accuracy']:.2%}")
            for bias_type, bias_data in analysis.get('bias_effectiveness', {}).items():
                if 'maintains_correct' in bias_data:
                    logger.info(f"    {bias_type}: {bias_data['maintains_correct']:.2%} maintain correct")
                elif 'causes_error' in bias_data:
                    logger.info(f"    {bias_type}: {bias_data['causes_error']:.2%} cause error")
        
        # Save all results with descriptive folder name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add subset info to folder name
        subset_info = ""
        if args.problem_type != 'all':
            subset_info = f"_{args.problem_type}"
        elif args.problem_range:
            subset_info = f"_range_{args.problem_range.replace('-', '_to_')}"
        elif args.problem_ids:
            subset_info = f"_{'_'.join(args.problem_ids[:2])}"  # First 2 IDs
        
        output_dir = Path(config.output_dir) / f"{timestamp}{subset_info}"
        save_comparison_results(all_results, all_analyses, output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETE")
        print("="*60)
        print(f"Problems tested: {len(all_results)}")
        print(f"Total samples generated: {len(all_results) * 5 * args.num_samples}")
        print(f"Results saved to: {output_dir}")
        
        # Quick stats
        avg_neutral = sum(a.get('neutral_accuracy', 0) for a in all_analyses) / len(all_analyses)
        print(f"\nAverage neutral accuracy: {avg_neutral:.2%}")
        
        # Check bias effectiveness
        wrong_bias_effect = []
        for a in all_analyses:
            for bias_type in ['biased_wrong', 'strong_bias_wrong']:
                if bias_type in a.get('bias_effectiveness', {}):
                    wrong_bias_effect.append(a['bias_effectiveness'][bias_type].get('causes_error', 0))
        
        if wrong_bias_effect:
            avg_wrong_effect = sum(wrong_bias_effect) / len(wrong_bias_effect)
            print(f"Average wrong bias effectiveness: {avg_wrong_effect:.2%}")
        
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up
        if 'model_loader' in locals():
            del model_loader
        if config.device == "cuda":
            torch.cuda.empty_cache()
    
    logger.info("\nExperiment complete!")


if __name__ == "__main__":
    main()
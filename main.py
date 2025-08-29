#!/usr/bin/env python3
"""Generate unfaithful CoT examples and run thought anchors analysis."""

import json
import argparse
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import torch
from tqdm import tqdm
from dotenv import load_dotenv

from src.config import ExperimentConfig
from src.model import ModelLoader
from src.prompts import (
    get_biased_prompt_pairs, 
    extract_cot_reasoning,
    detect_unfaithfulness,
    format_cot_for_anchors
)

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate unfaithful CoT examples with DeepSeek-R1-Distill-Qwen-14B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (5 examples)
  python main.py
  
  # Generate more examples with custom temperature
  python main.py --max-examples 10 --temperature 0.9
  
  # Test mode with smaller model
  python main.py --test --max-examples 2
  
  # Run with specific bias type
  python main.py --bias-type suggested_answer --max-examples 5
  
  # Run thought anchors analysis after generation
  python main.py --run-anchors --max-examples 10
        """
    )
    
    # Experiment parameters
    parser.add_argument(
        '--max-examples',
        type=int,
        default=5,
        help='Maximum number of examples to generate (default: 5)'
    )
    
    parser.add_argument(
        '--bias-type',
        type=str,
        choices=['suggested_answer', 'sycophancy', 'spurious_few_shot', 'all'],
        default='all',
        help='Type of bias to test (default: all)'
    )
    
    # Model parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Generation temperature (default: 0.8)'
    )
    
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.95,
        help='Top-p sampling (default: 0.95)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=1024,
        help='Maximum new tokens to generate (default: 1024)'
    )
    
    # Paths
    parser.add_argument(
        '--model',
        type=str,
        default='deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
        help='Model name or path (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)'
    )
    
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='./model_cache',
        help='Model cache directory (default: ./model_cache)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/unfaithful_cot',
        help='Output directory for results (default: ./outputs/unfaithful_cot)'
    )
    
    # Flags
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode with minimal settings'
    )
    
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage even if GPU is available'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--run-anchors',
        action='store_true',
        help='Run thought anchors analysis after generation'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    return parser.parse_args()


def generate_cot_examples(
    model_loader: ModelLoader,
    prompt_pairs: List,
    config: ExperimentConfig,
    max_examples: int = 5
) -> Dict[str, List]:
    """Generate CoT examples for both faithful and unfaithful prompts."""
    
    results = {
        "faithful": [],
        "unfaithful": []
    }
    
    logger.info("\n" + "="*60)
    logger.info("GENERATING COT EXAMPLES")
    logger.info("="*60)
    
    for pair in tqdm(prompt_pairs[:max_examples], desc="Generating examples"):
        logger.info(f"\n[{pair.id}] {pair.bias_type}")
        
        # Generate faithful response
        logger.info("  Generating faithful response...")
        faithful_cot, faithful_answer = model_loader.generate(
            pair.faithful_prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p
        )
        
        # Generate unfaithful response
        logger.info("  Generating unfaithful response...")
        unfaithful_cot, unfaithful_answer = model_loader.generate(
            pair.unfaithful_prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p
        )
        
        # Detect unfaithfulness
        unfaithfulness_check = detect_unfaithfulness(
            pair, unfaithful_cot, unfaithful_answer
        )
        
        # Format for thought anchors
        faithful_sentences = format_cot_for_anchors(faithful_cot)
        unfaithful_sentences = format_cot_for_anchors(unfaithful_cot)
        
        # Store results
        results["faithful"].append({
            "id": pair.id,
            "bias_type": pair.bias_type,
            "prompt": pair.faithful_prompt,
            "full_response": f"{faithful_cot}\n\nFinal answer: {faithful_answer}" if faithful_cot else faithful_answer,
            "cot_reasoning": faithful_cot,
            "final_answer": faithful_answer,
            "correct_answer": pair.correct_answer,
            "is_correct": faithful_answer and pair.correct_answer.lower() in faithful_answer.lower(),
            "sentences": faithful_sentences
        })
        
        results["unfaithful"].append({
            "id": pair.id,
            "bias_type": pair.bias_type,
            "prompt": pair.unfaithful_prompt,
            "full_response": f"{unfaithful_cot}\n\nFinal answer: {unfaithful_answer}" if unfaithful_cot else unfaithful_answer,
            "cot_reasoning": unfaithful_cot,
            "final_answer": unfaithful_answer,
            "correct_answer": pair.correct_answer,
            "biased_answer": pair.biased_answer,
            "gave_biased_answer": unfaithful_answer and pair.biased_answer and pair.biased_answer.lower() in unfaithful_answer.lower(),
            "unfaithfulness": unfaithfulness_check,
            "sentences": unfaithful_sentences
        })
        
        # Print summary
        logger.info(f"  Faithful: {faithful_answer} (Correct: {pair.correct_answer})")
        logger.info(f"  Unfaithful: {unfaithful_answer} (Biased toward: {pair.biased_answer})")
        if unfaithfulness_check["is_unfaithful"]:
            logger.info(f"  ⚠️ UNFAITHFUL: {unfaithfulness_check['unfaithfulness_type']}")
            logger.info(f"     Evidence: {unfaithfulness_check['evidence']}")
    
    return results


def save_results(results: Dict, output_dir: Path):
    """Save results in multiple formats."""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results as JSON
    with open(output_dir / "cot_examples.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save sentences for thought anchors analysis
    for condition in ["faithful", "unfaithful"]:
        sentences_file = output_dir / f"{condition}_sentences.jsonl"
        with open(sentences_file, "w") as f:
            for example in results[condition]:
                if example["sentences"]:
                    entry = {
                        "id": example["id"],
                        "sentences": example["sentences"],
                        "final_answer": example["final_answer"],
                        "metadata": {
                            "bias_type": example["bias_type"],
                            "is_unfaithful": example.get("unfaithfulness", {}).get("is_unfaithful", False)
                        }
                    }
                    f.write(json.dumps(entry) + "\n")
    
    # Create readable summary
    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("UNFAITHFUL COT GENERATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        # Count unfaithful examples
        unfaithful_count = sum(
            1 for ex in results["unfaithful"] 
            if ex.get("unfaithfulness", {}).get("is_unfaithful", False)
        )
        
        f.write(f"Total examples: {len(results['faithful'])}\n")
        f.write(f"Unfaithful examples detected: {unfaithful_count}\n\n")
        
        # Detailed results
        for i, (faithful, unfaithful) in enumerate(zip(results["faithful"], results["unfaithful"])):
            f.write(f"\n{'='*40}\n")
            f.write(f"Example {i+1}: {faithful['id']}\n")
            f.write(f"Bias type: {faithful['bias_type']}\n")
            f.write(f"{'='*40}\n\n")
            
            f.write("FAITHFUL VERSION:\n")
            f.write(f"Prompt: {faithful['prompt']}\n")
            f.write(f"Answer: {faithful['final_answer']} (Correct: {faithful['correct_answer']})\n")
            f.write(f"Is correct: {faithful['is_correct']}\n\n")
            
            f.write("UNFAITHFUL VERSION:\n")
            f.write(f"Prompt: {unfaithful['prompt']}\n")
            f.write(f"Answer: {unfaithful['final_answer']} (Biased: {unfaithful['biased_answer']})\n")
            f.write(f"Gave biased answer: {unfaithful.get('gave_biased_answer', False)}\n")
            
            if unfaithful.get("unfaithfulness", {}).get("is_unfaithful"):
                f.write(f"⚠️ UNFAITHFUL: {unfaithful['unfaithfulness']['unfaithfulness_type']}\n")
                f.write(f"Evidence: {unfaithful['unfaithfulness']['evidence']}\n")
    
    logger.info(f"\n✅ Results saved to {output_dir}")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Adjust for test mode
    if args.test:
        logger.info("Running in TEST MODE")
        args.model = "gpt2"
        args.max_examples = min(args.max_examples, 2)
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = ExperimentConfig(
        model_name=args.model,
        device='cpu' if args.cpu else 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
        cache_dir=args.cache_dir,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    # Log configuration
    logger.info("Experiment Configuration:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Max examples: {args.max_examples}")
    logger.info(f"  Bias type: {args.bias_type}")
    logger.info(f"  Temperature: {config.temperature}")
    logger.info(f"  Top-p: {config.top_p}")
    logger.info(f"  Max tokens: {config.max_new_tokens}")
    logger.info(f"  Output dir: {config.output_dir}")
    
    try:
        # Initialize and load model
        logger.info("\n" + "="*50)
        logger.info("Loading model...")
        logger.info("="*50)
        model_loader = ModelLoader(config)
        model_loader.load()
        
        # Get prompt pairs
        prompt_pairs = get_biased_prompt_pairs()
        
        # Filter by bias type if specified
        if args.bias_type != 'all':
            prompt_pairs = [p for p in prompt_pairs if p.bias_type == args.bias_type]
            logger.info(f"Filtered to {len(prompt_pairs)} prompts with bias type: {args.bias_type}")
        
        # Generate examples
        logger.info("\n" + "="*50)
        logger.info("Starting generation...")
        logger.info("="*50)
        
        results = generate_cot_examples(
            model_loader,
            prompt_pairs,
            config,
            max_examples=args.max_examples
        )
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.output_dir) / timestamp
        save_results(results, output_dir)
        
        # Print summary stats
        unfaithful_count = sum(
            1 for ex in results["unfaithful"] 
            if ex.get("unfaithfulness", {}).get("is_unfaithful", False)
        )
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Generated {len(results['faithful'])} prompt pairs")
        print(f"Detected {unfaithful_count} unfaithful examples")
        print(f"Results saved to: {output_dir}")
        
        # Run thought anchors analysis if requested
        if args.run_anchors:
            logger.info("\n" + "="*50)
            logger.info("Running thought anchors analysis...")
            logger.info("="*50)
            
            import subprocess
            anchor_cmd = [
                "python", "run_anchors_analysis.py",
                "--output-dir", str(output_dir)
            ]
            subprocess.run(anchor_cmd, check=True)
            logger.info("Thought anchors analysis complete!")
        
    except KeyboardInterrupt:
        logger.warning("Generation interrupted by user")
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        import sys
        sys.exit(1)
    finally:
        # Clean up model
        if 'model_loader' in locals():
            del model_loader
        if config.device == "cuda":
            torch.cuda.empty_cache()
    
    logger.info("\nGeneration complete!")


if __name__ == "__main__":
    main()
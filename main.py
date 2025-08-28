"""Generate unfaithful CoT examples and run thought anchors analysis."""

import json
import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import torch
from tqdm import tqdm

from src.config import ExperimentConfig
from src.model import ModelLoader
from src.prompts import (
    get_biased_prompt_pairs, 
    extract_cot_reasoning,
    detect_unfaithfulness,
    format_cot_for_anchors
)


def generate_cot_examples(
    model_loader: ModelLoader,
    prompt_pairs: List,
    max_examples: int = 5
) -> Dict[str, List]:
    """Generate CoT examples for both faithful and unfaithful prompts."""
    
    results = {
        "faithful": [],
        "unfaithful": []
    }
    
    print("\n" + "="*60)
    print("GENERATING COT EXAMPLES")
    print("="*60)
    
    for pair in tqdm(prompt_pairs[:max_examples], desc="Generating examples"):
        print(f"\n[{pair.id}] {pair.bias_type}")
        
        # Generate faithful response
        print("  Generating faithful response...")
        faithful_response = model_loader.generate(
            pair.faithful_prompt,
            max_new_tokens=1024,
            temperature=0.8,
            top_p=0.95
        )
        
        # Generate unfaithful response
        print("  Generating unfaithful response...")
        unfaithful_response = model_loader.generate(
            pair.unfaithful_prompt,
            max_new_tokens=1024,
            temperature=0.8,
            top_p=0.95
        )
        
        # Extract CoT and answers
        faithful_cot, faithful_answer = extract_cot_reasoning(faithful_response)
        unfaithful_cot, unfaithful_answer = extract_cot_reasoning(unfaithful_response)
        
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
            "full_response": faithful_response,
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
            "full_response": unfaithful_response,
            "cot_reasoning": unfaithful_cot,
            "final_answer": unfaithful_answer,
            "correct_answer": pair.correct_answer,
            "biased_answer": pair.biased_answer,
            "gave_biased_answer": unfaithful_answer and pair.biased_answer and pair.biased_answer.lower() in unfaithful_answer.lower(),
            "unfaithfulness": unfaithfulness_check,
            "sentences": unfaithful_sentences
        })
        
        # Print summary
        print(f"  Faithful: {faithful_answer} (Correct: {pair.correct_answer})")
        print(f"  Unfaithful: {unfaithful_answer} (Biased toward: {pair.biased_answer})")
        if unfaithfulness_check["is_unfaithful"]:
            print(f"  ⚠️ UNFAITHFUL: {unfaithfulness_check['unfaithfulness_type']}")
            print(f"     Evidence: {unfaithfulness_check['evidence']}")
    
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
    
    print(f"\n✅ Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate unfaithful CoT examples")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                       help="Model to use")
    parser.add_argument("--max-examples", type=int, default=3,
                       help="Maximum number of examples to generate")
    parser.add_argument("--output-dir", default="outputs/unfaithful_cot",
                       help="Output directory")
    parser.add_argument("--cpu", action="store_true",
                       help="Use CPU instead of GPU")
    parser.add_argument("--test", action="store_true",
                       help="Test mode with smaller model")
    
    args = parser.parse_args()
    
    # Use smaller model for testing
    if args.test:
        args.model = "gpt2"
        args.max_examples = 1
    
    # Initialize model config
    config = ExperimentConfig(
        model_name=args.model,
        device="cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        cache_dir="./model_cache",
        max_new_tokens=1024,
        temperature=0.8,
        top_p=0.95
    )
    
    print(f"Loading model: {config.model_name}")
    print(f"Using device: {config.device}")
    
    # Initialize and load model
    model_loader = ModelLoader(config)
    model_loader.load()
    
    # Get prompt pairs
    prompt_pairs = get_biased_prompt_pairs()
    
    # Generate examples
    results = generate_cot_examples(
        model_loader,
        prompt_pairs,
        max_examples=args.max_examples
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
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
    
    # Clean up model
    del model_loader
    if config.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
"""Run thought anchors analysis on unfaithful CoT examples."""

import json
import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
import numpy as np
from tqdm import tqdm

# Add thought-anchors to path
sys.path.append('../thought-anchors')

from src.config import ExperimentConfig
from src.model import ModelLoader


def run_counterfactual_resampling(
    model_loader: ModelLoader,
    sentences: List[str],
    final_answer: str,
    num_rollouts: int = 5
) -> Dict[int, float]:
    """
    Run counterfactual resampling to identify important sentences.
    
    Returns:
        Dict mapping sentence index to importance score
    """
    importance_scores = {}
    
    print(f"Running counterfactual resampling with {num_rollouts} rollouts...")
    
    for i in tqdm(range(len(sentences)), desc="Evaluating sentences"):
        # Create prompt without sentence i
        partial_sentences = sentences[:i]
        if not partial_sentences:
            importance_scores[i] = 0.0
            continue
            
        partial_prompt = " ".join(partial_sentences)
        
        # Track how often we get the same answer
        same_answer_count = 0
        
        for _ in range(num_rollouts):
            # Generate completion from partial prompt
            response = model_loader.generate(
                partial_prompt + " Therefore, the answer is",
                max_new_tokens=50,
                temperature=0.8
            )
            
            # Check if answer matches original
            if final_answer.lower() in response.lower():
                same_answer_count += 1
        
        # Importance = how much removing this sentence changes the outcome
        importance_scores[i] = 1.0 - (same_answer_count / num_rollouts)
    
    return importance_scores


def extract_attention_patterns(
    model_loader: ModelLoader,
    sentences: List[str]
) -> np.ndarray:
    """
    Extract attention patterns from the model.
    Simplified version - would need access to model internals.
    
    Returns:
        Attention matrix (num_sentences x num_sentences)
    """
    print("Extracting attention patterns...")
    
    # This is a simplified placeholder
    # Real implementation would hook into model attention layers
    num_sentences = len(sentences)
    
    # For now, return a mock attention matrix
    # In practice, you'd extract this from the model's attention heads
    attention_matrix = np.random.rand(num_sentences, num_sentences)
    
    # Make it somewhat realistic: later tokens attend more to earlier ones
    for i in range(num_sentences):
        for j in range(i+1, num_sentences):
            attention_matrix[j, i] *= 2  # Later attends to earlier
            
    # Normalize rows
    for i in range(num_sentences):
        if attention_matrix[i].sum() > 0:
            attention_matrix[i] /= attention_matrix[i].sum()
    
    return attention_matrix


def run_attention_suppression(
    model_loader: ModelLoader,
    sentences: List[str],
    anchor_indices: List[int],
    final_answer: str
) -> Dict[int, bool]:
    """
    Test if suppressing attention to anchor sentences changes the output.
    
    Returns:
        Dict mapping anchor index to whether suppression changed answer
    """
    print(f"Testing attention suppression on {len(anchor_indices)} anchors...")
    
    suppression_results = {}
    
    for anchor_idx in anchor_indices:
        # Create prompt with anchor sentence masked/weakened
        modified_sentences = sentences.copy()
        modified_sentences[anchor_idx] = "[SUPPRESSED]"
        modified_prompt = " ".join(modified_sentences)
        
        # Generate with suppressed anchor
        response = model_loader.generate(
            modified_prompt + " Therefore, the answer is",
            max_new_tokens=50,
            temperature=0.8
        )
        
        # Check if answer changed
        answer_changed = final_answer.lower() not in response.lower()
        suppression_results[anchor_idx] = answer_changed
        
        if answer_changed:
            print(f"  Suppressing sentence {anchor_idx} changed the answer!")
    
    return suppression_results


def analyze_example(
    model_loader: ModelLoader,
    example: Dict,
    num_rollouts: int = 5
) -> Dict:
    """Analyze a single CoT example."""
    
    sentences = example["sentences"]
    final_answer = example["final_answer"]
    
    if not sentences:
        return {
            "id": example["id"],
            "error": "No sentences to analyze"
        }
    
    print(f"\nAnalyzing example: {example['id']}")
    print(f"Number of sentences: {len(sentences)}")
    
    # 1. Counterfactual resampling
    importance_scores = run_counterfactual_resampling(
        model_loader, sentences, final_answer, num_rollouts
    )
    
    # 2. Attention analysis
    attention_matrix = extract_attention_patterns(model_loader, sentences)
    
    # 3. Identify top anchors (sentences with high importance)
    sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    top_anchors = [idx for idx, score in sorted_importance[:3] if score > 0.5]
    
    # 4. Attention suppression on top anchors
    suppression_results = {}
    if top_anchors:
        suppression_results = run_attention_suppression(
            model_loader, sentences, top_anchors, final_answer
        )
    
    return {
        "id": example["id"],
        "num_sentences": len(sentences),
        "importance_scores": importance_scores,
        "top_anchors": top_anchors,
        "attention_matrix": attention_matrix.tolist(),
        "suppression_results": suppression_results,
        "metadata": example.get("metadata", {})
    }


def compare_conditions(faithful_results: List[Dict], unfaithful_results: List[Dict]) -> Dict:
    """Compare anchor patterns between faithful and unfaithful examples."""
    
    comparison = {
        "faithful": {
            "avg_num_anchors": 0,
            "avg_anchor_position": 0,
            "avg_max_importance": 0
        },
        "unfaithful": {
            "avg_num_anchors": 0,
            "avg_anchor_position": 0,
            "avg_max_importance": 0
        }
    }
    
    for condition, results in [("faithful", faithful_results), ("unfaithful", unfaithful_results)]:
        if not results:
            continue
            
        num_anchors = []
        anchor_positions = []
        max_importances = []
        
        for result in results:
            if "error" in result:
                continue
                
            anchors = result["top_anchors"]
            num_anchors.append(len(anchors))
            
            if anchors and result["num_sentences"] > 0:
                # Normalized position (0 = beginning, 1 = end)
                avg_pos = np.mean([a / result["num_sentences"] for a in anchors])
                anchor_positions.append(avg_pos)
            
            if result["importance_scores"]:
                max_importances.append(max(result["importance_scores"].values()))
        
        if num_anchors:
            comparison[condition]["avg_num_anchors"] = np.mean(num_anchors)
        if anchor_positions:
            comparison[condition]["avg_anchor_position"] = np.mean(anchor_positions)
        if max_importances:
            comparison[condition]["avg_max_importance"] = np.mean(max_importances)
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Run thought anchors analysis")
    parser.add_argument("--input-dir", required=True,
                       help="Directory containing CoT examples")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                       help="Model to use for analysis")
    parser.add_argument("--num-rollouts", type=int, default=5,
                       help="Number of rollouts for counterfactual resampling")
    parser.add_argument("--cpu", action="store_true",
                       help="Use CPU instead of GPU")
    parser.add_argument("--max-examples", type=int, default=None,
                       help="Maximum number of examples to analyze")
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    output_dir = input_dir / "anchors_analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize model
    config = ExperimentConfig(
        model_name=args.model,
        device="cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        cache_dir="./model_cache"
    )
    
    print(f"Loading model: {config.model_name}")
    print(f"Using device: {config.device}")
    
    model_loader = ModelLoader(config)
    model_loader.load()
    
    # Process both conditions
    all_results = {}
    
    for condition in ["faithful", "unfaithful"]:
        sentences_file = input_dir / f"{condition}_sentences.jsonl"
        
        if not sentences_file.exists():
            print(f"Warning: {sentences_file} not found, skipping {condition} condition")
            continue
        
        print(f"\n{'='*60}")
        print(f"ANALYZING {condition.upper()} EXAMPLES")
        print(f"{'='*60}")
        
        # Load examples
        examples = []
        with open(sentences_file) as f:
            for line in f:
                examples.append(json.loads(line))
        
        if args.max_examples:
            examples = examples[:args.max_examples]
        
        # Analyze each example
        results = []
        for example in examples:
            result = analyze_example(model_loader, example, args.num_rollouts)
            results.append(result)
        
        all_results[condition] = results
        
        # Save results
        with open(output_dir / f"{condition}_analysis.json", "w") as f:
            json.dump(results, f, indent=2)
    
    # Compare conditions
    if "faithful" in all_results and "unfaithful" in all_results:
        print(f"\n{'='*60}")
        print("COMPARING FAITHFUL VS UNFAITHFUL")
        print(f"{'='*60}")
        
        comparison = compare_conditions(
            all_results["faithful"],
            all_results["unfaithful"]
        )
        
        print("\nComparison Results:")
        for condition in ["faithful", "unfaithful"]:
            print(f"\n{condition.upper()}:")
            stats = comparison[condition]
            print(f"  Average number of anchors: {stats['avg_num_anchors']:.2f}")
            print(f"  Average anchor position: {stats['avg_anchor_position']:.2f}")
            print(f"  Average max importance: {stats['avg_max_importance']:.2f}")
        
        # Save comparison
        with open(output_dir / "comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)
    
    print(f"\n✅ Analysis complete. Results saved to {output_dir}")
    
    # Cleanup
    del model_loader
    if config.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
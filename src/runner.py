"""Main experiment runner for Qwen-14B misalignment experiments."""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import time

from .config import ExperimentConfig, MisalignmentResult, ExperimentSummary
from .model import ModelLoader
from .prompts import PromptManager

logger = logging.getLogger(__name__)


class QwenMisalignmentRunner:
    """Runner for misalignment experiments with Qwen-14B."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize the experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.model_loader = ModelLoader(config)
        self.prompt_manager = PromptManager()
        self.results: List[MisalignmentResult] = []
        
        # Setup output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = Path(self.config.output_dir) / f"experiment_{datetime.now():%Y%m%d_%H%M%S}.log"
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    def load_model(self):
        """Load the model and tokenizer."""
        logger.info("Loading Qwen-14B model...")
        self.model_loader.load()
        logger.info("Model loaded successfully!")
    
    def run_scenario(
        self,
        scenario: str,
        max_attempts: Optional[int] = None,
        early_stop: Optional[int] = None
    ) -> ExperimentSummary:
        """Run experiments for a specific scenario.
        
        Args:
            scenario: Scenario name (murder, blackmail, leaking)
            max_attempts: Maximum number of attempts (overrides config)
            early_stop: Stop after this many successful misalignments
            
        Returns:
            Summary of experiment results
        """
        max_attempts = max_attempts or self.config.max_attempts
        early_stop = early_stop or self.config.early_stop_threshold
        
        logger.info(f"Starting scenario: {scenario}")
        logger.info(f"Max attempts: {max_attempts}, Early stop: {early_stop}")
        
        # Get prompts for this scenario
        prompts = self.prompt_manager.get_scenario_prompts(
            scenario,
            include_variations=True,
            shuffle=True,
            seed=self.config.seed
        )
        
        # Track results
        scenario_results = []
        misalignment_count = 0
        misalignment_types = {}
        confidence_scores = []
        
        start_time = datetime.now().isoformat()
        
        # Progress bar
        pbar = tqdm(total=max_attempts, desc=f"Scenario: {scenario}")
        
        attempt_num = 0
        prompt_idx = 0
        
        while attempt_num < max_attempts:
            # Get next prompt (cycle through if needed)
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1
            attempt_num += 1
            
            try:
                # Generate response
                full_prompt = self.prompt_manager.format_for_model(
                    prompt["system"],
                    prompt["user"]
                )
                
                cot_reasoning, final_answer = self.model_loader.generate(
                    full_prompt,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_new_tokens=self.config.max_new_tokens
                )
                
                # Check for misalignment
                is_misaligned, misalignment_type, confidence = self.prompt_manager.check_misalignment(
                    final_answer,
                    scenario,
                    cot_reasoning
                )
                
                # Create result
                result = MisalignmentResult(
                    attempt_num=attempt_num,
                    scenario=scenario,
                    prompt=prompt,
                    cot_reasoning=cot_reasoning,
                    final_answer=final_answer,
                    is_misaligned=is_misaligned,
                    misalignment_type=misalignment_type,
                    confidence_score=confidence,
                    generation_params={
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                        "max_new_tokens": self.config.max_new_tokens
                    }
                )
                
                scenario_results.append(result)
                self.results.append(result)
                
                # Update tracking
                if is_misaligned:
                    misalignment_count += 1
                    if misalignment_type:
                        misalignment_types[misalignment_type] = misalignment_types.get(misalignment_type, 0) + 1
                    confidence_scores.append(confidence)
                    
                    # Log successful misalignment
                    logger.info(f"✅ Misalignment found! Type: {misalignment_type}, Confidence: {confidence:.2f}")
                    
                    # Save intermediate results
                    self._save_intermediate_result(result)
                    
                    # Check early stop
                    if misalignment_count >= early_stop:
                        logger.info(f"Early stopping: {misalignment_count} misalignments found")
                        break
                else:
                    logger.debug(f"❌ No misalignment detected in attempt {attempt_num}")
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    "misaligned": misalignment_count,
                    "rate": f"{misalignment_count/attempt_num:.1%}"
                })
                
                # Small delay to avoid overwhelming the model
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in attempt {attempt_num}: {e}")
                
                # Create error result
                result = MisalignmentResult(
                    attempt_num=attempt_num,
                    scenario=scenario,
                    prompt=prompt,
                    cot_reasoning="",
                    final_answer="",
                    is_misaligned=False,
                    misalignment_type=None,
                    confidence_score=0.0,
                    error=str(e),
                    generation_params={}
                )
                scenario_results.append(result)
                pbar.update(1)
        
        pbar.close()
        
        # Calculate summary statistics
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        misalignment_rate = misalignment_count / len(scenario_results) if scenario_results else 0.0
        
        summary = ExperimentSummary(
            scenario=scenario,
            total_attempts=len(scenario_results),
            successful_misalignments=misalignment_count,
            misalignment_rate=misalignment_rate,
            misalignment_types=misalignment_types,
            avg_confidence=avg_confidence,
            start_time=start_time,
            end_time=datetime.now().isoformat(),
            config=self.config
        )
        
        # Save final results
        self._save_scenario_results(scenario, scenario_results, summary)
        
        logger.info(f"Scenario {scenario} complete: {misalignment_count}/{len(scenario_results)} misalignments")
        
        return summary
    
    def run_all_scenarios(self) -> Dict[str, ExperimentSummary]:
        """Run all configured scenarios.
        
        Returns:
            Dictionary of scenario summaries
        """
        summaries = {}
        
        for scenario in self.config.scenarios:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running scenario: {scenario}")
            logger.info(f"{'='*50}\n")
            
            summary = self.run_scenario(scenario)
            summaries[scenario] = summary
            
            # Save progress
            self.save_all_results()
        
        return summaries
    
    def _save_intermediate_result(self, result: MisalignmentResult):
        """Save a single result immediately after finding misalignment."""
        output_file = Path(self.config.output_dir) / f"misalignment_{result.scenario}_{result.attempt_num}.json"
        
        with open(output_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Also append to a cumulative file for safety
        cumulative_file = Path(self.config.output_dir) / f"{result.scenario}_cumulative.jsonl"
        with open(cumulative_file, 'a') as f:
            json.dump(result.to_dict(), f)
            f.write('\n')
    
    def _save_scenario_results(
        self,
        scenario: str,
        results: List[MisalignmentResult],
        summary: ExperimentSummary
    ):
        """Save all results for a scenario."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = Path(self.config.output_dir) / f"{scenario}_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        
        # Save summary
        summary_file = Path(self.config.output_dir) / f"{scenario}_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)
        
        # Save misaligned examples as text for easy reading
        misaligned = [r for r in results if r.is_misaligned]
        if misaligned:
            examples_file = Path(self.config.output_dir) / f"{scenario}_misaligned_examples_{timestamp}.txt"
            with open(examples_file, 'w') as f:
                for result in misaligned:
                    f.write(f"{'='*80}\n")
                    f.write(f"Attempt {result.attempt_num} - {result.misalignment_type}\n")
                    f.write(f"Confidence: {result.confidence_score:.2f}\n")
                    f.write(f"{'='*80}\n\n")
                    f.write(f"SYSTEM: {result.prompt['system']}\n\n")
                    f.write(f"USER: {result.prompt['user']}\n\n")
                    f.write(f"COT REASONING:\n{result.cot_reasoning}\n\n")
                    f.write(f"FINAL ANSWER:\n{result.final_answer}\n\n")
                    f.write("\n")
    
    def save_all_results(self):
        """Save all accumulated results."""
        if not self.results:
            logger.warning("No results to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(self.config.output_dir) / f"all_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        
        logger.info(f"Saved {len(self.results)} results to {output_file}")
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        self.model_loader.unload()
        logger.info("Cleanup complete")
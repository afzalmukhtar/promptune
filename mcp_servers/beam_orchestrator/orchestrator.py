"""
Beam Orchestrator - Core Promptune optimization algorithm.

Coordinates evaluator and optimizers through beam search.
Supports custom evaluation targets (black-box systems) via EvaluationTarget protocol.
"""

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from mcp_servers.evaluator.evaluator import evaluate_prompt
from mcp_servers.few_shot_optimizer.optimizer import select_examples
from mcp_servers.meta_prompt_optimizer.optimizer import optimize as meta_optimize
from mcp_servers.utils.logger import Component, logger
from mcp_servers.utils.output_saver import save_optimization_result
from schemas import EvaluationResult, TrainingExample

if TYPE_CHECKING:
    from mcp_servers.targets.base import EvaluationTarget

load_dotenv()


@dataclass
class BeamConfig:
    """Configuration for beam search optimization."""

    beam_width: int = 3
    max_iterations: int = 10
    target_score: float = 0.90
    convergence_threshold: float = 0.02
    convergence_patience: int = 3
    optimizers: list[str] = field(default_factory=lambda: ["meta_prompt", "few_shot"])


@dataclass
class IterationResult:
    """Result of a single optimization iteration."""

    iteration: int
    best_score: float
    beam_scores: list[float]
    beam_prompts: list[str]
    candidates_generated: int
    candidates_evaluated: int


@dataclass
class OptimizationResult:
    """Final result of beam optimization."""

    best_prompt: str
    best_score: float
    iterations: int
    converged: bool
    convergence_reason: str
    history: list[IterationResult]


def get_default_model() -> str:
    """Get the default model from environment variables."""
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_model = os.getenv("AZURE_OPENAI_MODEL")
    if azure_endpoint and azure_model:
        return f"azure/{azure_model}"
    ollama_base = os.getenv("OLLAMA_API_BASE")
    if ollama_base:
        return "ollama/llama3.2"
    return "gpt-4o-mini"


async def _evaluate_candidates(
    prompts: list[str],
    examples: list[TrainingExample],
    model: str,
    log_details: bool = False,
    target: "EvaluationTarget | None" = None,
) -> list[tuple[str, EvaluationResult]]:
    """Evaluate multiple prompts and return with results."""
    results = []
    for i, prompt in enumerate(prompts):
        if log_details:
            logger.stage(Component.EVALUATOR, f"Evaluating candidate {i + 1}/{len(prompts)}")
            logger.prompt_preview(prompt)
        eval_result = await evaluate_prompt(prompt, examples, model, target=target)
        if log_details:
            logger.result("Score", f"{eval_result.score:.0%}")
            if eval_result.weaknesses:
                logger.detail(f"Weaknesses: {', '.join(eval_result.weaknesses[:2])}")
        results.append((prompt, eval_result))
    return results


async def _generate_meta_prompt_candidates(
    prompt: str,
    eval_result: EvaluationResult,
    num_candidates: int,
    cross_pollination: list[str] | None,
    model: str,
    log_details: bool = False,
) -> list[str]:
    """Generate candidates using meta-prompt optimizer."""
    if log_details:
        logger.stage(Component.META_OPTIMIZER, "Generating improved prompts")
        logger.info(f"Requesting {num_candidates} candidate(s)")
        if cross_pollination:
            logger.detail("Using cross-pollination from best prompt")

    feedback = {
        "score": eval_result.score,
        "weaknesses": eval_result.weaknesses,
        "suggestions": eval_result.suggestions,
        "strengths": eval_result.strengths,
    }
    result = await meta_optimize(
        prompt=prompt,
        feedback=feedback,
        num_candidates=num_candidates,
        cross_pollination_prompts=cross_pollination,
        model=model,
    )
    candidates = [c.prompt for c in result.candidates if c.prompt]

    if log_details:
        logger.success(f"Generated {len(candidates)} candidate(s)")
        for i, c in enumerate(result.candidates[:2]):
            if c.strategy:
                logger.detail(f"Strategy {i + 1}: {c.strategy[:60]}...")

    return candidates


async def _generate_few_shot_candidates(
    prompt: str,
    examples: list[TrainingExample],
    model: str,
    log_details: bool = False,
) -> list[str]:
    """Generate candidate by optimizing few-shot examples."""
    if len(examples) < 2:
        return []

    if log_details:
        logger.stage(Component.FEW_SHOT_OPTIMIZER, "Selecting optimal examples")
        logger.info(f"Pool size: {len(examples)} examples")

    result = await select_examples(
        prompt=prompt,
        example_pool=examples,
        num_examples=min(3, len(examples)),
        strategy="balanced",
        model=model,
    )

    if log_details and result.selected_examples:
        logger.success(f"Selected {len(result.selected_examples)} examples")
        logger.detail(result.selection_reasoning)

    return [result.prompt_with_examples] if result.prompt_with_examples else []


async def optimize_beam(
    initial_prompt: str,
    training_examples: list[TrainingExample],
    config: BeamConfig | None = None,
    model: str | None = None,
    verbose: bool = True,
    target: "EvaluationTarget | None" = None,
    save_output: bool | str = False,
) -> OptimizationResult:
    """
    Run beam search optimization on a prompt.

    Args:
        initial_prompt: Starting prompt to optimize
        training_examples: Examples to evaluate against
        config: Beam search configuration
        model: Model to use for all components
        verbose: Enable detailed logging
        target: Optional custom evaluation target (black-box system).
                If provided, uses target.invoke(prompt, input) for evaluation.
        save_output: Save result to disk. True saves to 'outputs/', string specifies dir.

    Returns:
        OptimizationResult with best prompt and optimization history
    """
    config = config or BeamConfig()
    model = model or get_default_model()
    logger.verbose = verbose

    if verbose:
        logger.header("Promptune - Beam Search Prompt Tuning")
        logger.stage(Component.ORCHESTRATOR, "Initializing optimization")
        logger.info(f"Model: {model}")
        logger.info(f"Beam width: {config.beam_width}")
        logger.info(f"Max iterations: {config.max_iterations}")
        logger.info(f"Target score: {config.target_score:.0%}")
        logger.info(f"Optimizers: {', '.join(config.optimizers)}")
        logger.info(f"Training examples: {len(training_examples)}")
        if target:
            logger.info(f"Target: {target}")

    beam = [initial_prompt]
    history: list[IterationResult] = []
    prev_best_score = 0.0
    no_improvement_count = 0

    for iteration in range(1, config.max_iterations + 1):
        if verbose:
            logger.iteration_start(iteration, config.max_iterations)

        # Evaluate current beam
        if verbose:
            logger.stage(Component.EVALUATOR, f"Evaluating beam ({len(beam)} prompts)")

        evaluated = await _evaluate_candidates(
            beam, training_examples, model, log_details=verbose, target=target
        )
        scores = [e[1].score for e in evaluated]
        best_idx = scores.index(max(scores))
        best_score = scores[best_idx]
        best_prompt = beam[best_idx]
        best_eval = evaluated[best_idx][1]

        if verbose:
            logger.info(f"Best score in beam: {best_score:.0%}")
            if best_eval.strengths:
                logger.detail(f"Strengths: {', '.join(best_eval.strengths[:2])}")
            if best_eval.weaknesses:
                logger.detail(f"Weaknesses: {', '.join(best_eval.weaknesses[:2])}")

        # Record iteration
        iter_result = IterationResult(
            iteration=iteration,
            best_score=best_score,
            beam_scores=scores,
            beam_prompts=beam.copy(),
            candidates_generated=0,
            candidates_evaluated=len(beam),
        )

        # Check target score reached
        if best_score >= config.target_score:
            if verbose:
                logger.success(f"🎉 Target score {config.target_score:.0%} reached!")
            history.append(iter_result)
            result = OptimizationResult(
                best_prompt=best_prompt,
                best_score=best_score,
                iterations=iteration,
                converged=True,
                convergence_reason=f"Target score {config.target_score} reached",
                history=history,
            )
            if save_output:
                output_dir = save_output if isinstance(save_output, str) else "outputs"
                paths = save_optimization_result(result, output_dir=output_dir)
                if verbose:
                    logger.success(f"Saved to {paths['prompt']}")
            return result

        # Check convergence (no improvement)
        improvement = best_score - prev_best_score
        if improvement < config.convergence_threshold:
            no_improvement_count += 1
            if verbose:
                logger.warning(
                    f"No significant improvement ({no_improvement_count}/{config.convergence_patience})"
                )
        else:
            no_improvement_count = 0
            if verbose and iteration > 1:
                logger.success(f"Improved by {improvement:.1%}")

        if no_improvement_count >= config.convergence_patience:
            if verbose:
                logger.info("Converged - no further improvement possible")
            history.append(iter_result)
            result = OptimizationResult(
                best_prompt=best_prompt,
                best_score=best_score,
                iterations=iteration,
                converged=True,
                convergence_reason=f"No improvement for {config.convergence_patience} iterations",
                history=history,
            )
            if save_output:
                output_dir = save_output if isinstance(save_output, str) else "outputs"
                paths = save_optimization_result(result, output_dir=output_dir)
                if verbose:
                    logger.success(f"Saved to {paths['prompt']}")
            return result

        prev_best_score = best_score

        # Generate new candidates
        if verbose:
            logger.stage(Component.ORCHESTRATOR, "Generating candidates")

        candidates: list[str] = []

        # Cross-pollination: use best prompt as reference
        cross_poll = [best_prompt] if len(beam) > 1 else None

        for prompt_idx, (prompt, eval_result) in enumerate(evaluated):
            if verbose:
                logger.info(f"Processing beam member {prompt_idx + 1}/{len(evaluated)}")

            # Meta-prompt optimizer
            if "meta_prompt" in config.optimizers:
                meta_candidates = await _generate_meta_prompt_candidates(
                    prompt=prompt,
                    eval_result=eval_result,
                    num_candidates=2,
                    cross_pollination=cross_poll if prompt != best_prompt else None,
                    model=model,
                    log_details=verbose,
                )
                candidates.extend(meta_candidates)

            # Few-shot optimizer
            if "few_shot" in config.optimizers:
                few_shot_candidates = await _generate_few_shot_candidates(
                    prompt=prompt,
                    examples=training_examples,
                    model=model,
                    log_details=verbose,
                )
                candidates.extend(few_shot_candidates)

        iter_result.candidates_generated = len(candidates)

        if verbose:
            logger.stage(
                Component.ORCHESTRATOR,
                f"Selecting top {config.beam_width} from {len(candidates)} candidates",
            )

        # Combine beam + candidates and evaluate
        all_prompts = list(set(beam + candidates))  # Dedupe
        all_evaluated = await _evaluate_candidates(
            all_prompts, training_examples, model, target=target
        )
        iter_result.candidates_evaluated = len(all_evaluated)

        # Select top-k for new beam
        all_evaluated.sort(key=lambda x: x[1].score, reverse=True)
        beam = [e[0] for e in all_evaluated[: config.beam_width]]

        if verbose:
            new_best = all_evaluated[0][1].score
            logger.iteration_end(new_best, new_best > best_score)

        history.append(iter_result)

    # Max iterations reached
    if verbose:
        logger.stage(Component.ORCHESTRATOR, "Max iterations reached - final evaluation")

    final_evaluated = await _evaluate_candidates(beam, training_examples, model, target=target)
    final_scores = [e[1].score for e in final_evaluated]
    best_idx = final_scores.index(max(final_scores))

    if verbose:
        logger.separator()
        logger.success(f"Optimization complete! Final score: {final_scores[best_idx]:.0%}")

    result = OptimizationResult(
        best_prompt=beam[best_idx],
        best_score=final_scores[best_idx],
        iterations=config.max_iterations,
        converged=False,
        convergence_reason="Max iterations reached",
        history=history,
    )
    if save_output:
        output_dir = save_output if isinstance(save_output, str) else "outputs"
        paths = save_optimization_result(result, output_dir=output_dir)
        if verbose:
            logger.success(f"Saved to {paths['prompt']}")
    return result


async def step(
    beam: list[str],
    training_examples: list[TrainingExample],
    optimizers: list[str] | None = None,
    model: str | None = None,
) -> dict:
    """
    Run a single optimization step.

    Args:
        beam: Current beam of prompts
        training_examples: Examples to evaluate against
        optimizers: Which optimizers to use
        model: Model to use

    Returns:
        Dict with new beam, scores, and stats
    """
    model = model or get_default_model()
    optimizers = optimizers or ["meta_prompt", "few_shot"]

    # Evaluate current beam
    evaluated = await _evaluate_candidates(beam, training_examples, model)
    scores = [e[1].score for e in evaluated]
    best_idx = scores.index(max(scores))
    best_prompt = beam[best_idx]

    # Generate candidates
    candidates: list[str] = []
    cross_poll = [best_prompt] if len(beam) > 1 else None

    for prompt, eval_result in evaluated:
        if "meta_prompt" in optimizers:
            meta_candidates = await _generate_meta_prompt_candidates(
                prompt=prompt,
                eval_result=eval_result,
                num_candidates=2,
                cross_pollination=cross_poll if prompt != best_prompt else None,
                model=model,
            )
            candidates.extend(meta_candidates)

        if "few_shot" in optimizers:
            few_shot_candidates = await _generate_few_shot_candidates(
                prompt=prompt,
                examples=training_examples,
                model=model,
            )
            candidates.extend(few_shot_candidates)

    # Evaluate all and select top
    all_prompts = list(set(beam + candidates))
    all_evaluated = await _evaluate_candidates(all_prompts, training_examples, model)
    all_evaluated.sort(key=lambda x: x[1].score, reverse=True)

    new_beam = [e[0] for e in all_evaluated[: len(beam)]]
    new_scores = [e[1].score for e in all_evaluated[: len(beam)]]

    return {
        "new_beam": new_beam,
        "scores": new_scores,
        "candidates_generated": len(candidates),
        "candidates_evaluated": len(all_evaluated),
    }

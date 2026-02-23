"""
Beam Orchestrator - Core Promptune optimization algorithm.

Coordinates evaluator and optimizers through beam search.
Supports custom evaluation targets (black-box systems) via EvaluationTarget protocol.
Uses PromptuneConfig for 3-model-role setup (target/tuner/judge).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mcp_servers.evaluator.evaluator import evaluate_prompt
from mcp_servers.few_shot_optimizer.optimizer import select_examples
from mcp_servers.meta_prompt_optimizer.optimizer import optimize as meta_optimize
from mcp_servers.utils.config import PromptuneConfig
from mcp_servers.utils.logger import Component, logger
from mcp_servers.utils.output_saver import save_optimization_result
from schemas import EvaluationResult, NegativeTrainingExample, TrainingExample

if TYPE_CHECKING:
    from mcp_servers.targets.base import EvaluationTarget


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


async def _evaluate_candidates(
    prompts: list[str],
    examples: list[TrainingExample],
    config: PromptuneConfig,
    iteration: int = 1,
    log_details: bool = False,
    target: "EvaluationTarget | None" = None,
    negative_examples: list[NegativeTrainingExample] | None = None,
) -> list[tuple[str, EvaluationResult]]:
    """Evaluate multiple prompts and return with results."""
    results = []
    for i, prompt in enumerate(prompts):
        if log_details:
            logger.stage(Component.EVALUATOR, f"Evaluating candidate {i + 1}/{len(prompts)}")
            logger.prompt_preview(prompt)
        eval_result = await evaluate_prompt(
            prompt, examples, config=config,
            target=target, iteration=iteration,
            batch_size=config.optimization.batch_size,
            negative_examples=negative_examples,
        )
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
    tuner_model: str,
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
        model=tuner_model,
        prompt_understanding=eval_result.prompt_understanding,
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
    tuner_model: str,
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
        model=tuner_model,
    )

    if log_details and result.selected_examples:
        logger.success(f"Selected {len(result.selected_examples)} examples")
        logger.detail(result.selection_reasoning)

    return [result.prompt_with_examples] if result.prompt_with_examples else []


async def _generate_adversarial_candidates(
    prompt: str,
    eval_result: EvaluationResult,
    tuner_model: str,
    negative_examples: list[NegativeTrainingExample] | None = None,
    log_details: bool = False,
) -> list[str]:
    """Generate candidates using adversarial optimizer."""
    try:
        from mcp_servers.adversarial_optimizer.optimizer import optimize as adversarial_optimize
    except ImportError:
        return []

    if log_details:
        logger.stage(Component.ADVERSARIAL_OPTIMIZER, "Adversarial prompt hardening")

    feedback = {
        "score": eval_result.score,
        "weaknesses": eval_result.weaknesses,
        "suggestions": eval_result.suggestions,
        "strengths": eval_result.strengths,
    }
    result = await adversarial_optimize(
        prompt=prompt,
        feedback=feedback,
        model=tuner_model,
        negative_examples=negative_examples,
        prompt_understanding=eval_result.prompt_understanding,
    )
    candidates = [c.prompt for c in result.candidates if c.prompt]

    if log_details:
        logger.success(f"Generated {len(candidates)} hardened candidate(s)")

    return candidates


async def _generate_example_augmentor_candidates(
    prompt: str,
    examples: list[TrainingExample],
    tuner_model: str,
    negative_examples: list[NegativeTrainingExample] | None = None,
    log_details: bool = False,
) -> list[str]:
    """Generate candidates using example augmentor (positive + negative examples)."""
    try:
        from mcp_servers.example_augmentor.optimizer import augment as example_augment
    except ImportError:
        return []

    if log_details:
        logger.stage(Component.EXAMPLE_AUGMENTOR, "Augmenting with positive/negative examples")

    result = await example_augment(
        prompt=prompt,
        positive_examples=examples,
        negative_examples=negative_examples,
        model=tuner_model,
    )

    if log_details and result:
        logger.success("Generated augmented prompt candidate")

    return [result] if result else []


async def _generate_clarity_candidates(
    prompt: str,
    eval_result: EvaluationResult,
    tuner_model: str,
    log_details: bool = False,
) -> list[str]:
    """Generate candidates using clarity rewriter."""
    try:
        from mcp_servers.clarity_rewriter.optimizer import rewrite as clarity_rewrite
    except ImportError:
        return []

    if log_details:
        logger.stage(Component.CLARITY_REWRITER, "Rewriting for clarity")

    result = await clarity_rewrite(
        prompt=prompt,
        model=tuner_model,
        prompt_understanding=eval_result.prompt_understanding,
    )
    candidates = [result] if result else []

    if log_details and candidates:
        logger.success("Generated clarity-rewritten candidate")

    return candidates


async def optimize_beam(
    initial_prompt: str,
    training_examples: list[TrainingExample],
    config: PromptuneConfig,
    verbose: bool = True,
    target: "EvaluationTarget | None" = None,
    save_output: bool | str = False,
    negative_examples: list[NegativeTrainingExample] | None = None,
) -> OptimizationResult:
    """
    Run beam search optimization on a prompt.

    Args:
        initial_prompt: Starting prompt to optimize
        training_examples: Examples to evaluate against
        config: PromptuneConfig with model roles and optimization settings
        verbose: Enable detailed logging
        target: Optional custom evaluation target (black-box system).
                If provided, uses target.invoke(prompt, input) for evaluation.
        save_output: Save result to disk. True saves to 'outputs/', string specifies dir.
        negative_examples: Optional negative examples for adversarial/augmentor optimizers

    Returns:
        OptimizationResult with best prompt and optimization history
    """
    opt = config.optimization
    tuner_model = config.models.tuner
    logger.verbose = verbose

    if verbose:
        logger.header("Promptune - Beam Search Prompt Tuning")
        logger.stage(Component.ORCHESTRATOR, "Initializing optimization")
        logger.info(f"Target model: {config.models.target}")
        logger.info(f"Tuner model: {tuner_model}")
        logger.info(f"Judge model: {config.models.judge}")
        logger.info(f"Beam width: {opt.beam_width}")
        logger.info(f"Max iterations: {opt.max_iterations}")
        logger.info(f"Target score: {opt.target_score:.0%}")
        logger.info(f"Batch size: {opt.batch_size}")
        logger.info(f"Optimizers: {', '.join(opt.optimizers)}")
        logger.info(f"Training examples: {len(training_examples)}")
        if negative_examples:
            logger.info(f"Negative examples: {len(negative_examples)}")
        if target:
            logger.info(f"Target: {target}")

    beam = [initial_prompt]
    history: list[IterationResult] = []
    global_best_score = 0.0
    no_improvement_count = 0

    for iteration in range(1, opt.max_iterations + 1):
        if verbose:
            logger.iteration_start(iteration, opt.max_iterations)

        # Evaluate current beam
        if verbose:
            logger.stage(Component.EVALUATOR, f"Evaluating beam ({len(beam)} prompts)")

        evaluated = await _evaluate_candidates(
            beam, training_examples, config,
            iteration=iteration, log_details=verbose, target=target,
            negative_examples=negative_examples,
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
            if best_eval.prompt_understanding:
                logger.detail(
                    f"Prompt compliance: {best_eval.prompt_understanding.overall_compliance:.0%}"
                )

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
        if best_score >= opt.target_score:
            if verbose:
                logger.success(f"Target score {opt.target_score:.0%} reached!")
            history.append(iter_result)
            result = OptimizationResult(
                best_prompt=best_prompt,
                best_score=best_score,
                iterations=iteration,
                converged=True,
                convergence_reason=f"Target score {opt.target_score} reached",
                history=history,
            )
            if save_output:
                output_dir = save_output if isinstance(save_output, str) else "outputs"
                paths = save_optimization_result(result, output_dir=output_dir)
                if verbose:
                    logger.success(f"Saved to {paths['prompt']}")
            return result

        # Check convergence (no improvement)
        improvement = best_score - global_best_score
        if improvement < opt.convergence_threshold:
            no_improvement_count += 1
            if verbose:
                logger.warning(
                    f"No significant improvement ({no_improvement_count}/{opt.convergence_patience})"
                )
        else:
            no_improvement_count = 0
            if verbose and iteration > 1:
                logger.success(f"Improved by {improvement:.1%}")

        if no_improvement_count >= opt.convergence_patience:
            if verbose:
                logger.info("Converged - no further improvement possible")
            history.append(iter_result)
            result = OptimizationResult(
                best_prompt=best_prompt,
                best_score=best_score,
                iterations=iteration,
                converged=True,
                convergence_reason=f"No improvement for {opt.convergence_patience} iterations",
                history=history,
            )
            if save_output:
                output_dir = save_output if isinstance(save_output, str) else "outputs"
                paths = save_optimization_result(result, output_dir=output_dir)
                if verbose:
                    logger.success(f"Saved to {paths['prompt']}")
            return result

        if best_score > global_best_score:
            global_best_score = best_score

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
            if "meta_prompt" in opt.optimizers:
                meta_candidates = await _generate_meta_prompt_candidates(
                    prompt=prompt,
                    eval_result=eval_result,
                    num_candidates=2,
                    cross_pollination=cross_poll if prompt != best_prompt else None,
                    tuner_model=tuner_model,
                    log_details=verbose,
                )
                candidates.extend(meta_candidates)

            # Few-shot optimizer
            if "few_shot" in opt.optimizers:
                few_shot_candidates = await _generate_few_shot_candidates(
                    prompt=prompt,
                    examples=training_examples,
                    tuner_model=tuner_model,
                    log_details=verbose,
                )
                candidates.extend(few_shot_candidates)

            # Adversarial optimizer
            if "adversarial" in opt.optimizers:
                adv_candidates = await _generate_adversarial_candidates(
                    prompt=prompt,
                    eval_result=eval_result,
                    tuner_model=tuner_model,
                    negative_examples=negative_examples,
                    log_details=verbose,
                )
                candidates.extend(adv_candidates)

            # Example augmentor
            if "example_augmentor" in opt.optimizers:
                aug_candidates = await _generate_example_augmentor_candidates(
                    prompt=prompt,
                    examples=training_examples,
                    tuner_model=tuner_model,
                    negative_examples=negative_examples,
                    log_details=verbose,
                )
                candidates.extend(aug_candidates)

            # Clarity rewriter
            if "clarity_rewriter" in opt.optimizers:
                clarity_candidates = await _generate_clarity_candidates(
                    prompt=prompt,
                    eval_result=eval_result,
                    tuner_model=tuner_model,
                    log_details=verbose,
                )
                candidates.extend(clarity_candidates)

        iter_result.candidates_generated = len(candidates)

        if verbose:
            logger.stage(
                Component.ORCHESTRATOR,
                f"Selecting top {opt.beam_width} from {len(candidates)} candidates",
            )

        # Evaluate only new candidates (reuse beam scores from above)
        beam_eval_cache = {prompt: eval_result for prompt, eval_result in evaluated}
        new_candidates = list(set(c for c in candidates if c not in beam_eval_cache))
        new_evaluated = await _evaluate_candidates(
            new_candidates, training_examples, config,
            iteration=iteration, target=target,
            negative_examples=negative_examples,
        ) if new_candidates else []
        all_evaluated = list(evaluated) + new_evaluated
        iter_result.candidates_evaluated = len(all_evaluated)

        # Select top-k for new beam
        all_evaluated.sort(key=lambda x: x[1].score, reverse=True)
        beam = [e[0] for e in all_evaluated[: opt.beam_width]]

        if verbose:
            new_best = all_evaluated[0][1].score
            logger.iteration_end(new_best, new_best > best_score)

        history.append(iter_result)

    # Max iterations reached
    if verbose:
        logger.stage(Component.ORCHESTRATOR, "Max iterations reached - final evaluation")

    final_evaluated = await _evaluate_candidates(
        beam, training_examples, config,
        iteration=opt.max_iterations, target=target,
        negative_examples=negative_examples,
    )
    final_scores = [e[1].score for e in final_evaluated]
    best_idx = final_scores.index(max(final_scores))

    if verbose:
        logger.separator()
        logger.success(f"Optimization complete! Final score: {final_scores[best_idx]:.0%}")

    result = OptimizationResult(
        best_prompt=beam[best_idx],
        best_score=final_scores[best_idx],
        iterations=opt.max_iterations,
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
    config: PromptuneConfig,
    optimizers: list[str] | None = None,
    negative_examples: list[NegativeTrainingExample] | None = None,
) -> dict:
    """
    Run a single optimization step.

    Args:
        beam: Current beam of prompts
        training_examples: Examples to evaluate against
        config: PromptuneConfig with model roles
        optimizers: Which optimizers to use (overrides config)
        negative_examples: Optional negative examples for reverse empirical + optimizers

    Returns:
        Dict with new beam, scores, and stats
    """
    optimizers = optimizers or config.optimization.optimizers
    tuner_model = config.models.tuner

    # Evaluate current beam
    evaluated = await _evaluate_candidates(
        beam, training_examples, config,
        negative_examples=negative_examples,
    )
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
                tuner_model=tuner_model,
            )
            candidates.extend(meta_candidates)

        if "few_shot" in optimizers:
            few_shot_candidates = await _generate_few_shot_candidates(
                prompt=prompt,
                examples=training_examples,
                tuner_model=tuner_model,
            )
            candidates.extend(few_shot_candidates)

        if "adversarial" in optimizers:
            adv_candidates = await _generate_adversarial_candidates(
                prompt=prompt,
                eval_result=eval_result,
                tuner_model=tuner_model,
                negative_examples=negative_examples,
            )
            candidates.extend(adv_candidates)

        if "example_augmentor" in optimizers:
            aug_candidates = await _generate_example_augmentor_candidates(
                prompt=prompt,
                examples=training_examples,
                tuner_model=tuner_model,
                negative_examples=negative_examples,
            )
            candidates.extend(aug_candidates)

        if "clarity_rewriter" in optimizers:
            clarity_candidates = await _generate_clarity_candidates(
                prompt=prompt,
                eval_result=eval_result,
                tuner_model=tuner_model,
            )
            candidates.extend(clarity_candidates)

    # Evaluate all and select top
    all_prompts = list(set(beam + candidates))
    all_evaluated = await _evaluate_candidates(
        all_prompts, training_examples, config,
        negative_examples=negative_examples,
    )
    all_evaluated.sort(key=lambda x: x[1].score, reverse=True)

    new_beam = [e[0] for e in all_evaluated[: len(beam)]]
    new_scores = [e[1].score for e in all_evaluated[: len(beam)]]

    return {
        "new_beam": new_beam,
        "scores": new_scores,
        "candidates_generated": len(candidates),
        "candidates_evaluated": len(all_evaluated),
    }

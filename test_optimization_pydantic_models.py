"""
Test cases for generic Pydantic models from optimization code.

This module contains test cases for complex generic Pydantic models that demonstrate
type inference capabilities with multi-TypeVar generics, nested models, and complex
type relationships.
"""

import typing
import types
from typing import TypeVar

from pydantic import BaseModel, Field
import pytest

from unification_type_inference import infer_return_type_unified as infer_return_type

# TypeVars for optimization models
T_TaskParameters = TypeVar('T_TaskParameters')
T_TaskResults = TypeVar('T_TaskResults')
T_Hyperparameters = TypeVar('T_Hyperparameters')


class TaskRun(BaseModel, typing.Generic[T_TaskParameters, T_TaskResults, T_Hyperparameters]):
    """
    The task run. This is the 'data' we use to optimize the hyperparameters.
    """
    task_parameters: T_TaskParameters = Field(..., description="The task parameters.")
    hyperparameters: T_Hyperparameters = Field(
        ...,
        description="The hyperparameters used for the task. We optimize these.",
    )
    all_chat_chains: dict = Field(..., description="The chat chains from the task execution.")
    return_value: T_TaskResults | None = Field(
        ..., description="The results of the task. (None for exceptions/failure.)"
    )
    exception: list[str] | str | None = Field(..., description="Exception that occurred during the task execution.")


class TaskReflection(BaseModel):
    """
    The reflections on the task.
    """
    feedback: str = Field(..., description="Feedback on the task results.")
    evaluation: str = Field(..., description="Evaluation of the outputs.")
    hyperparameter_suggestion: str = Field(..., description="How to change hyperparameters.")
    hyperparameter_missing: str = Field(..., description="What hyperparameters are missing.")


class TaskInfo(BaseModel, typing.Generic[T_TaskParameters, T_TaskResults, T_Hyperparameters]):
    """
    The task run and the reflection on the experiment.
    """
    task_parameters: T_TaskParameters = Field(..., description="The task parameters.")
    hyperparameters: T_Hyperparameters = Field(..., description="The hyperparameters used for the task.")
    reflection: TaskReflection = Field(..., description="The reflection on the task.")


class OptimizationInfo(BaseModel, typing.Generic[T_TaskParameters, T_TaskResults, T_Hyperparameters]):
    """
    The optimization information.
    """
    older_task_summary: str | None = Field(None, description="Summary of previous experiments.")
    task_infos: list[TaskInfo[T_TaskParameters, T_TaskResults, T_Hyperparameters]] = Field(
        ..., description="The most recent tasks we have run."
    )
    best_hyperparameters: T_Hyperparameters = Field(..., description="The best hyperparameters found so far.")


class OptimizationStep(BaseModel, typing.Generic[T_TaskParameters, T_TaskResults, T_Hyperparameters]):
    """
    The next optimization steps.
    """
    best_hyperparameters: T_Hyperparameters = Field(..., description="The best hyperparameters found so far.")
    suggestion: str = Field(..., description="Suggestions for the next experiments.")
    task_parameters_suggestions: list[T_TaskParameters] = Field(..., description="Task parameters to try next.")
    hyperparameter_suggestions: list[T_Hyperparameters] = Field(..., description="Hyperparameters to try next.")


def test_capture_task_run_generic_signature():
    """Test generic signature for capture_task_run function."""
    
    def extract_task_parameters(
        task_run: TaskRun[T_TaskParameters, T_TaskResults, T_Hyperparameters]
    ) -> T_TaskParameters:
        """Extract task parameters from task run."""
        return task_run.task_parameters
    
    def extract_hyperparameters(
        task_run: TaskRun[T_TaskParameters, T_TaskResults, T_Hyperparameters]
    ) -> T_Hyperparameters:
        """Extract hyperparameters from task run."""
        return task_run.hyperparameters
    
    # Test with concrete types
    task_run = TaskRun[str, int, float](
        task_parameters="test",
        hyperparameters=0.01,
        all_chat_chains={},
        return_value=4,
        exception=None
    )
    
    # Test type inference
    params_type = infer_return_type(extract_task_parameters, task_run)
    assert params_type is str
    
    hyperparams_type = infer_return_type(extract_hyperparameters, task_run)
    assert hyperparams_type is float


def test_optimize_hyperparameters_generic_signature():
    """Test generic signature for optimize_hyperparameters function."""
    
    def extract_task_suggestions(
        step: OptimizationStep[T_TaskParameters, T_TaskResults, T_Hyperparameters]
    ) -> list[T_TaskParameters]:
        """Extract task parameter suggestions from optimization step."""
        return step.task_parameters_suggestions
    
    def extract_hyperparameter_suggestions(
        step: OptimizationStep[T_TaskParameters, T_TaskResults, T_Hyperparameters]
    ) -> list[T_Hyperparameters]:
        """Extract hyperparameter suggestions from optimization step."""
        return step.hyperparameter_suggestions
    
    # Test with concrete types
    step = OptimizationStep[str, int, float](
        best_hyperparameters=0.01,
        suggestion="Test optimization",
        task_parameters_suggestions=["task1", "task2"],
        hyperparameter_suggestions=[0.01, 0.02]
    )
    
    # Test type inference
    task_suggestions_type = infer_return_type(extract_task_suggestions, step)
    assert typing.get_origin(task_suggestions_type) is list
    assert typing.get_args(task_suggestions_type) == (str,)
    
    hyperparam_suggestions_type = infer_return_type(extract_hyperparameter_suggestions, step)
    assert typing.get_origin(hyperparam_suggestions_type) is list
    assert typing.get_args(hyperparam_suggestions_type) == (float,)


def test_reflect_on_task_run_generic_signature():
    """Test generic signature for reflect_on_task_run function."""
    
    def extract_reflection_feedback(
        reflection: TaskReflection
    ) -> str:
        """Extract feedback from reflection."""
        return reflection.feedback
    
    # Test with concrete types
    reflection = TaskReflection(
        feedback="Generic reflection",
        evaluation="Generic evaluation",
        hyperparameter_suggestion="Generic suggestion",
        hyperparameter_missing="Generic missing"
    )
    
    # Test type inference
    feedback_type = infer_return_type(extract_reflection_feedback, reflection)
    assert feedback_type is str


def test_summarize_optimization_info_generic_signature():
    """Test generic signature for summarize_optimization_info function."""
    
    def summarize_optimization_info(
        optimization_info: OptimizationInfo[T_TaskParameters, T_TaskResults, T_Hyperparameters]
    ) -> str:
        """Summarize optimization info with generic types."""
        return f"Summary for {len(optimization_info.task_infos)} tasks"
    
    def extract_summary_length(
        summary: str
    ) -> int:
        """Extract length from summary."""
        return len(summary)
    
    # Test with concrete types
    reflection = TaskReflection(
        feedback="Good",
        evaluation="Excellent",
        hyperparameter_suggestion="Increase",
        hyperparameter_missing="None"
    )
    
    task_info = TaskInfo[str, int, float](
        task_parameters="test",
        hyperparameters=0.01,
        reflection=reflection
    )
    
    opt_info = OptimizationInfo[str, int, float](
        older_task_summary=None,
        task_infos=[task_info],
        best_hyperparameters=0.01
    )
    
    summary = summarize_optimization_info(opt_info)
    
    # Test type inference
    summary_length_type = infer_return_type(extract_summary_length, summary)
    assert summary_length_type is int


def test_suggest_next_optimization_step_generic_signature():
    """Test generic signature for suggest_next_optimization_step function."""
    
    def suggest_next_optimization_step(
        optimization_info: OptimizationInfo[T_TaskParameters, T_TaskResults, T_Hyperparameters]
    ) -> OptimizationStep[T_TaskParameters, T_TaskResults, T_Hyperparameters]:
        """Suggest next optimization step with generic types."""
        return OptimizationStep[T_TaskParameters, T_TaskResults, T_Hyperparameters](
            best_hyperparameters=optimization_info.best_hyperparameters,
            suggestion="Next step suggestion",
            task_parameters_suggestions=[],
            hyperparameter_suggestions=[]
        )
    
    def extract_best_hyperparameters(
        step: OptimizationStep[T_TaskParameters, T_TaskResults, T_Hyperparameters]
    ) -> T_Hyperparameters:
        """Extract best hyperparameters from step."""
        return step.best_hyperparameters
    
    # Test with concrete types
    reflection = TaskReflection(
        feedback="Good",
        evaluation="Excellent",
        hyperparameter_suggestion="Increase",
        hyperparameter_missing="None"
    )
    
    task_info = TaskInfo[str, int, float](
        task_parameters="test",
        hyperparameters=0.01,
        reflection=reflection
    )
    
    opt_info = OptimizationInfo[str, int, float](
        older_task_summary=None,
        task_infos=[task_info],
        best_hyperparameters=0.01
    )
    
    step = suggest_next_optimization_step(opt_info)
    
    # Test type inference
    best_hyperparams_type = infer_return_type(extract_best_hyperparameters, step)
    assert best_hyperparams_type is float


def test_probability_for_improvement_generic_signature():
    """Test generic signature for probability_for_improvement function."""
    
    class ImprovementProbability(BaseModel):
        considerations: list[str] = Field(..., description="Considerations for improvement.")
        probability: float = Field(..., description="Probability of improvement.")
    
    def extract_probability(
        improvement_prob: ImprovementProbability
    ) -> float:
        """Extract probability from improvement probability."""
        return improvement_prob.probability
    
    # Test with concrete types
    improvement_prob = ImprovementProbability(
        considerations=["Generic consideration"],
        probability=0.5
    )
    
    # Test type inference
    probability_type = infer_return_type(extract_probability, improvement_prob)
    assert probability_type is float


def test_complex_generic_function_chaining():
    """Test complex generic function chaining from optimization workflow."""
    
    def process_task_run(
        task_run: TaskRun[T_TaskParameters, T_TaskResults, T_Hyperparameters]
    ) -> TaskInfo[T_TaskParameters, T_TaskResults, T_Hyperparameters]:
        """Process task run into task info."""
        reflection = TaskReflection(
            feedback="Processed",
            evaluation="Good",
            hyperparameter_suggestion="Continue",
            hyperparameter_missing="None"
        )
        return TaskInfo[T_TaskParameters, T_TaskResults, T_Hyperparameters](
            task_parameters=task_run.task_parameters,
            hyperparameters=task_run.hyperparameters,
            reflection=reflection
        )
    
    def aggregate_task_infos(
        task_infos: list[TaskInfo[T_TaskParameters, T_TaskResults, T_Hyperparameters]]
    ) -> OptimizationInfo[T_TaskParameters, T_TaskResults, T_Hyperparameters]:
        """Aggregate task infos into optimization info."""
        return OptimizationInfo[T_TaskParameters, T_TaskResults, T_Hyperparameters](
            older_task_summary=None,
            task_infos=task_infos,
            best_hyperparameters=task_infos[0].hyperparameters if task_infos else None
        )
    
    def extract_aggregated_hyperparameters(
        opt_info: OptimizationInfo[T_TaskParameters, T_TaskResults, T_Hyperparameters]
    ) -> T_Hyperparameters:
        """Extract hyperparameters from aggregated info."""
        return opt_info.best_hyperparameters
    
    # Test with concrete types
    task_run = TaskRun[str, int, float](
        task_parameters="test_task",
        hyperparameters=0.05,
        all_chat_chains={},
        return_value=100,
        exception=None
    )
    
    task_info = process_task_run(task_run)
    opt_info = aggregate_task_infos([task_info])
    
    # Test type inference
    hyperparams_type = infer_return_type(extract_aggregated_hyperparameters, opt_info)
    assert hyperparams_type is float


def test_generic_function_with_multiple_typevars():
    """Test generic function with multiple TypeVars in complex signature."""
    
    def transform_task_data(
        task_run: TaskRun[T_TaskParameters, T_TaskResults, T_Hyperparameters],
        transformer: typing.Callable[[T_TaskResults], T_TaskResults]
    ) -> TaskRun[T_TaskParameters, T_TaskResults, T_Hyperparameters]:
        """Transform task data using a generic transformer function."""
        return TaskRun[T_TaskParameters, T_TaskResults, T_Hyperparameters](
            task_parameters=task_run.task_parameters,
            hyperparameters=task_run.hyperparameters,
            all_chat_chains=task_run.all_chat_chains,
            return_value=transformer(task_run.return_value) if task_run.return_value is not None else None,
            exception=task_run.exception
        )
    
    def extract_transformed_result(
        transformed_run: TaskRun[T_TaskParameters, T_TaskResults, T_Hyperparameters]
    ) -> T_TaskResults | None:
        """Extract transformed result from task run."""
        return transformed_run.return_value
    
    # Test with concrete types
    def double_transformer(x: int) -> int:
        return x * 2
    
    original_run = TaskRun[str, int, float](
        task_parameters="test",
        hyperparameters=0.01,
        all_chat_chains={},
        return_value=21,
        exception=None
    )
    
    transformed_run = transform_task_data(original_run, double_transformer)
    
    # Test type inference
    result_type = infer_return_type(extract_transformed_result, transformed_run)
    origin = typing.get_origin(result_type)
    assert origin is typing.Union or origin is types.UnionType or result_type is int
    if origin is typing.Union or origin is types.UnionType:
        assert int in typing.get_args(result_type)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

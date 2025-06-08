"""Utility functions for the project."""

import os
import logging
import argparse
import pickle

from evaluate import load
from src.models.huggingface_models import HuggingfaceModel
from src.utils import gptapi as gpt


BRIEF_PROMPTS = {
    # "default": "Answer the following question as briefly as possible.\n",
    "chat": "Answer the following question in a single brief but complete sentence.\n",
}


def get_parser(stages: list = ["generate", "compute"]) -> argparse.ArgumentParser:
    """Create and configure an argument parser for different pipeline stages.
    
    Args:
        stages: List of pipeline stages to include arguments for (e.g., ["generate", "compute"])
    
    Returns:
        Configured argparse.ArgumentParser object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, required=False, default=23)
    parser.add_argument(
        "--metric",
        type=str,
        default="gpt-4o",
        help="Metric to assign accuracy to generations (e.g., 'gpt-4o', 'squad')."
    )
    parser.add_argument(
        "--compute_accuracy_at_all_temps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute accuracy at all temperatures or only t<<1."
    )

    # Arguments for generation stage
    if "generate" in stages:
        parser.add_argument(
            "--model_name",
            type=str,
            required=True,
            help="Name of the language model to use (e.g., 'llama-7b')."
        )
        parser.add_argument(
            "--model_max_new_tokens",
            type=int,
            default=100,
            help="Maximum number of new tokens to generate per response."
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default="trivia_qa",
            choices=["trivia_qa", "squad", "bioasq", "nq", "svamp"],
            help="Dataset to use for training/evaluation."
        )
        parser.add_argument(
            "--ood_train_dataset",
            type=str,
            default=None,
            choices=["trivia_qa", "squad", "bioasq", "nq", "svamp"],
            help="Out-of-distribution dataset for few-shot prompting and p_ik training."
        )
        parser.add_argument(
            "--num_samples", type=int, default=400, help="Number of samples to process."
        )
        parser.add_argument(
            "--p_true_num_fewshot",
            type=int,
            default=10,
            help="Number of few-shot examples for p_true calculation."
        )
        parser.add_argument(
            "--p_true_hint",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Include hints in p_true generation prompts."
        )
        parser.add_argument(
            "--num_generations",
            type=int,
            default=10,
            help="Number of responses to generate per question."
        )
        parser.add_argument(
            "--temperature", type=float, default=1.0, help="Sampling temperature for generation."
        )
        parser.add_argument(
            "--use_mc_options",
            type=bool,
            default=True,
            help="Include multiple-choice options in the question prompt."
        )
        parser.add_argument(
            "--use_context",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Include context information in generation prompts."
        )
        parser.add_argument(
            "--get_training_set_generations_most_likely_only",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Only generate most likely answer for training set (reduces computation)."
        )
        parser.add_argument(
            "--compute_p_true", default=True, action=argparse.BooleanOptionalAction,
            help="Enable calculation of p_true uncertainty measure."
        )
        parser.add_argument(
            "--brief_always", default=False, action=argparse.BooleanOptionalAction,
            help="Force brief responses for all generations."
        )
        parser.add_argument(
            "--enable_brief", default=True, action=argparse.BooleanOptionalAction,
            help="Enable brief response mode (when applicable)."
        )
        parser.add_argument(
            "--compute_uncertainties",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Trigger uncertainty measures computation pipeline."
        )

    # Arguments for computation stage
    if "compute" in stages:
        parser.add_argument(
            "--rundir",
            type=str,
            help="Directory path for storing computation results."
        )
        parser.add_argument(
            "--runid",
            type=str,
            default=None,
            help="Unique identifier for the current run (used for file organization)."
        )
        parser.add_argument(
            "--num_eval_samples", type=int, default=int(1e19),
            help="Maximum number of evaluation samples to process (default: process all)."
        )
        parser.add_argument(
            "--compute_predictive_entropy",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Enable calculation of predictive entropy uncertainty measures."
        )
        parser.add_argument(
            "--compute_p_ik", default=True, action=argparse.BooleanOptionalAction,
            help="Enable calculation of p_ik uncertainty measure."
        )
        parser.add_argument(
            "--analyze_run", default=True, action=argparse.BooleanOptionalAction,
            help="Automatically analyze results after computation."
        )
        parser.add_argument(
            "--condition_on_question",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Condition entailment model on both question and response."
        )
        parser.add_argument(
            "--strict_entailment", default=True, action=argparse.BooleanOptionalAction,
            help="Use strict entailment criteria for semantic clustering."
        )
        parser.add_argument(
            "--use_all_generations", default=True, action=argparse.BooleanOptionalAction,
            help="Use all generated responses (vs subset) for uncertainty calculations."
        )
        parser.add_argument(
            "--use_num_generations", type=int, default=-1,
            help="Number of generations to use when use_all_generations=False (-1=all)."
        )
        parser.add_argument(
            "--entailment_model", default="deberta", type=str,
            help="Entailment model to use for semantic clustering (e.g., 'deberta', 'gpt-4o')."
        )
        parser.add_argument(
            "--entailment_cache_id",
            default=None,
            type=str,
            help="ID of cached entailment predictions to reuse (for GPT-4/LLaMa models)."
        )
        parser.add_argument(
            "--entailment_cache_only",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Only use cached entailment predictions (no new computations)."
        )
        parser.add_argument(
            "--reuse_entailment_model",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Use entailment model as p_true model (shared computation)."
        )
    return parser


def setup_logger() -> None:
    """Configure the logger to display timestamps and log levels.
    
    Sets up basic logging configuration with format:
    [YYYY-MM-DD HH:MM:SS] [LEVEL] message
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.getLogger().setLevel(logging.INFO)  


def split_dataset(dataset: list) -> tuple[list, list]:
    """Split dataset into answerable and unanswerable question indices.
    
    Args:
        dataset: List of dataset examples, each containing "answers" field
    
    Returns:
        Tuple of (answerable_indices, unanswerable_indices)
    """
    def get_answer_count(ex: dict) -> int:
        """Helper function to get number of valid answers in an example."""
        return len(ex["answers"]["text"])

    answerable_indices = [i for i, ex in enumerate(dataset) if get_answer_count(ex) > 0]
    unanswerable_indices = [i for i, ex in enumerate(dataset) if get_answer_count(ex) == 0]

    # Validate split covers entire dataset with no overlaps
    assert set(answerable_indices) | set(unanswerable_indices) == set(range(len(dataset))), \
        "Dataset split does not cover all indices"
    assert set(answerable_indices) & set(unanswerable_indices) == set(), \
        "Overlap between answerable and unanswerable indices"

    return answerable_indices, unanswerable_indices


def model_based_metric(predicted_answer: str, example: dict, model) -> float:
    """Evaluate if a predicted answer matches expected answers using a model.
    
    Args:
        predicted_answer: The generated answer to evaluate
        example: Dataset example containing question and expected answers
        model: Evaluation model (e.g., GPT-4, DeBERTa)
    
    Returns:
        1.0 if predicted answer matches, 0.0 otherwise
    """
    # Extract correct answers from example
    if 'answers' in example:
        correct_answers = example['answers']['text']
    elif 'reference' in example:
        correct_answers = example['reference']['answers']['text']
    else:
        raise ValueError("Example missing 'answers' or 'reference' field")

    # Construct evaluation prompt
    prompt = f"We are assessing the quality of answers to the following question: {example['question']}\n"
    if len(correct_answers) == 1:
        prompt += f"The expected answer is: {correct_answers[0]}.\n"
    else:
        prompt += f"The following are expected answers to this question: {correct_answers}.\n"
    prompt += f"The proposed answer is: {predicted_answer}\n"
    if len(correct_answers) == 1:
        prompt += "Within the context of the question, does the proposed answer mean the same as the expected answer?"
    else:
        prompt += "Within the context of the question, does the proposed answer mean the same as any of the expected answers?"
    prompt += " Respond only with yes or no.\nResponse:"

    # Get model prediction with low temperature first
    model_response = model.predict(prompt, 0.01).lower()
    if 'yes' in model_response:
        return 1.0
    elif 'no' in model_response:
        return 0.0

    # Retry with higher temperature if ambiguous response
    logging.warning("Redo LLM check due to ambiguous response")
    model_response = model.predict(prompt, 1).lower()
    if 'yes' in model_response:
        return 1.0
    elif 'no' in model_response:
        return 0.0

    # Default to 'no' if still ambiguous
    logging.warning("Answer neither 'no' nor 'yes'. Defaulting to no!")
    return 0.0


def llm_metric(predicted_answer: str, example: dict, model) -> float:
    """Wrapper for model-based metric using LLM evaluator."""
    return model_based_metric(predicted_answer, example, model)


def get_gpt_metric(metric_name: str):
    """Create a GPT-based evaluation metric function.
    
    Args:
        metric_name: Name of GPT model to use (e.g., 'gpt-4o')
    
    Returns:
        Evaluation function that uses the specified GPT model
    """
    logging.info("Loading metric model %s.", metric_name)

    class EntailmentGPT:
        """Wrapper class for GPT-based entailment prediction."""
        def __init__(self, model_name: str):
            self.model_name = model_name

        def predict(self, prompt: str, temperature: float) -> str:
            """Generate prediction using GPT API."""
            return gpt.predict(prompt, temperature, model=self.model_name)

    gpt_model = EntailmentGPT(metric_name)

    def gpt_metric(predicted_answer: str, example: dict, model) -> float:
        """Evaluation function using GPT model."""
        del model  # Unused parameter
        return model_based_metric(predicted_answer, example, gpt_model)

    return gpt_metric


def get_reference(example: dict) -> dict:
    """Extract reference answer information from dataset example.
    
    Args:
        example: Dataset example containing answer information
    
    Returns:
        Reference dictionary with answers and ID
    """
    if "answers" not in example:
        example = example["reference"]
    answers = example["answers"]
    return {
        "answers": {
            "answer_start": answers.get("answer_start", []),
            "text": answers["text"]
        },
        "id": example["id"]
    }


def init_model(args) -> HuggingfaceModel:
    """Initialize the language model based on arguments.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Initialized HuggingfaceModel instance
    """
    model_name = args.model_name.lower()
    if any(model_type in model_name for model_type in ["falcon", "phi", "llama"]):
        return HuggingfaceModel(
            model_name,
            stop_sequences="default",
            max_new_tokens=args.model_max_new_tokens
        )
    raise ValueError(f"Unknown model_name `{args.model_name}`. Supported models: Falcon, Phi, Llama")


def get_make_prompt(args):
    """Create a prompt construction function based on arguments.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Prompt construction function
    """
    def make_prompt(context: str, question: str, answer: str, brief: bool, brief_always: bool) -> str:
        """Construct a generation prompt.
        
        Args:
            context: Context information to include (if enabled)
            question: Question text
            answer: Answer text (for few-shot examples)
            brief: Whether to use brief mode
            brief_always: Whether to force brief mode
        
        Returns:
            Constructed prompt string
        """
        prompt = ""
        if brief_always:
            prompt += BRIEF_PROMPTS["chat"]
        if args.use_context and context is not None:
            prompt += f"Context: {context}\n"
        prompt += f"Question: {question}\n"
        if answer:
            prompt += f"Answer: {answer}\n\n"
        else:
            prompt += "Answer:"
        return prompt

    return make_prompt


def get_metric(metric: str):
    """Get evaluation metric function based on metric name.
    
    Args:
        metric: Name of metric (e.g., 'squad', 'gpt-4o')
    
    Returns:
        Evaluation function
    """
    if metric == "squad":
        squad_metric = load("squad_v2")

        def squad_evaluation(response: str, example: dict, *args, **kwargs) -> float:
            """Evaluate response using SQuAD v2 metric."""
            # Extract example ID
            if "id" in example:
                exid = example["id"]
            elif "id" in example["reference"]:
                exid = example["reference"]["id"]
            else:
                raise ValueError("Example missing ID field")

            # Format prediction and reference
            prediction = {
                "prediction_text": response,
                "no_answer_probability": 0.0,
                "id": exid
            }
            results = squad_metric.compute(
                predictions=[prediction],
                references=[get_reference(example)]
            )
            return 1.0 if (results["f1"] >= 50.0) else 0.0

        return squad_evaluation

    elif "gpt" in metric.lower():
        return get_gpt_metric(metric)

    raise ValueError(f"Unsupported metric: {metric}")


def save(object_to_save, runid: str, filename: str) -> None:
    """Save an object to a pickle file in the run directory.
    
    Args:
        object_to_save: Python object to serialize
        runid: Unique run identifier
        filename: Name of the output file
    """
    run_dir = f'run_record/{runid}/'
    os.makedirs(run_dir, exist_ok=True)
    with open(f'{run_dir}/{filename}', 'wb') as f:
        pickle.dump(object_to_save, f)
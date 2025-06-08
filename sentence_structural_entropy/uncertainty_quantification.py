"""Compute uncertainty measures after generating answers."""
from collections import defaultdict
import logging
import os
import pickle
import numpy as np

from src.uncertainty_measures.p_ik import get_p_ik
from src.uncertainty_measures.semantic_entropy import get_semantic_ids
from src.uncertainty_measures.semantic_entropy import logsumexp
from src.uncertainty_measures.semantic_entropy import length_normalized_entropy
from src.uncertainty_measures.semantic_entropy import semantic_entropy
from src.uncertainty_measures.semantic_entropy import cluster_assignment_entropy
from src.uncertainty_measures.semantic_entropy import EntailmentDeberta
from src.uncertainty_measures.semantic_entropy import EntailmentGPT4
from src.utils import utils
from src.uncertainty_measures.structural_entropy import compute_se
from src.uncertainty_measures.construct_semantic_graph import get_semantic_clusters as AMSC_semantic_ids
from src.uncertainty_measures.construct_semantic_graph import build_semantic_graph 
from analyze_results import analyze_run

utils.setup_logger()

def restore(filename: str, runid: str) -> str:
    """Generate the full path to a file within the run record directory.
    
    Args:
        filename (str): Name of the target file (e.g., 'train_generations.pkl')
        runid (str): Identifier of the current run (e.g., '20250608_110824')
    
    Returns:
        str: Absolute path to the file
    """
    return os.path.join("run_record", runid, filename)

def is_answerable(generation: dict) -> bool:
    """Check if a generation has at least one valid answer.
    
    Args:
        generation (dict): Generation data containing 'reference' field
    
    Returns:
        bool: True if answers exist, False otherwise
    """
    return len(generation['reference']['answers']['text']) > 0

def main(args, runid: str):
    """Main entry point for uncertainty quantification computation.
    
    Args:
        args: Command-line arguments
        runid (str): Identifier of the current run
    """
    # Load training data if p_ik computation is enabled
    train_generations = None
    if args.compute_p_ik:
        train_generations_path = restore("train_generations.pkl", runid)
        with open(train_generations_path, "rb") as infile:
            train_generations = pickle.load(infile)

    # Initialize entailment model if predictive entropy computation is enabled
    entailment_model = None
    if args.compute_predictive_entropy:
        logging.info("Beginning loading for entailment model.")
        if args.entailment_model == "deberta":
            entailment_model = EntailmentDeberta()
        elif args.entailment_model == "gpt-4o":
            entailment_model = EntailmentGPT4(args.entailment_cache_id, args.entailment_cache_only)
        else:
            raise ValueError(f"Unsupported entailment model: {args.entailment_model}")
        logging.info("Entailment model loading complete.")

    # Restore base results from generate_answers.py
    result_dict_path = restore("uncertainty_measures.pkl", runid)
    with open(result_dict_path, "rb") as infile:
        result_dict = pickle.load(infile)
    result_dict["semantic_ids"] = []

    # Load validation generations data
    validation_generations_path = restore("validation_generations.pkl", runid)
    with open(validation_generations_path, "rb") as infile:
        validation_generations = pickle.load(infile)

    # Initialize data structures for entropy tracking
    entropies = defaultdict(list)  # Stores different entropy values per sample
    validation_embeddings, validation_is_true, validation_answerable = [], [], []
    count = 0  # Counter for early termination

    # Process validation samples to compute uncertainties
    for idx, tid in enumerate(validation_generations):
        example = validation_generations[tid]
        question = example["question"]
        full_responses = example["responses"]
        most_likely_answer = example["most_likely_answer"]

        # Select responses based on user configuration
        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError("use_num_generations must be specified when use_all_generations is False")
            responses = [fr[0] for fr in full_responses[: args.use_num_generations]]
        else:
            responses = [fr[0] for fr in full_responses]

        # Track validation metadata
        validation_is_true.append(most_likely_answer["accuracy"])
        validation_answerable.append(is_answerable(example))
        validation_embeddings.append(most_likely_answer["embedding"])
        logging.info("validation_is_true: %f", validation_is_true[-1])

        if args.compute_predictive_entropy:
            # Extract token log likelihoods from responses
            if not args.use_all_generations:
                log_liks = [r[1] for r in full_responses[: args.use_num_generations]]
            else:
                log_liks = [r[1] for r in full_responses]
            for log_lik in log_liks:  # Ensure non-empty log likelihoods
                assert log_lik

            # Condition responses on question if required
            if args.condition_on_question and args.entailment_model == "deberta":
                responses = [f"{question} {r}" for r in responses]

            # Compute semantic IDs for clustering
            if args.entailment_model == "AMSC":
                semantic_ids, _ = AMSC_semantic_ids(responses)
            else:
                semantic_ids = get_semantic_ids(
                    responses,
                    model=entailment_model,
                    strict_entailment=args.strict_entailment,
                    example=example
                )
            result_dict["semantic_ids"].append(semantic_ids)


            # Length-normalized predictive entropy
            log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]  # Average per-token log likelihood
            length_ne = length_normalized_entropy(log_liks_agg)
            entropies["length_normalized_entropy"].append(length_ne)

            # Semantic entropy using Rao's quadratic entropy
            log_likelihood_per_id = logsumexp(semantic_ids, log_liks_agg, agg="sum_normalized")
            semantic_e = semantic_entropy(log_likelihood_per_id)
            entropies["semantic_entropy"].append(semantic_e)

            # Structural entropy from semantic graph
            semantic_matrix = build_semantic_graph(responses, question)
            structural_e = compute_se(semantic_matrix)
            entropies["structural_entropy"].append(structural_e)

            # Log detailed debugging information
            entropies_fmt = ", ".join([f"{k}: {v[-1]:.2f}" for k, v in entropies.items()])
            logging.info(80 * "#")
            logging.info("NEW ITEM %d at id=`%s`.", idx, tid)
            logging.info("Context:\n%s", example["context"])
            logging.info("Question:\n%s", question)
            logging.info("True Answers:\n%s", example["reference"])
            logging.info("Low Temperature Generation:\n%s", most_likely_answer["response"])
            logging.info("Low Temperature Generation Accuracy: %f", most_likely_answer["accuracy"])
            logging.info("High Temp Generation Responses:\n%s", [r[0] for r in full_responses])
            logging.info("Metrics: semantic_ids=%s, avg_token_log_likelihoods=%s, entropies=%s",
                         semantic_ids, log_liks_agg, entropies_fmt)

        # Early termination check
        count += 1
        if count >= args.num_eval_samples:
            logging.info("Breaking out of main loop after %d samples.", count)
            break

    # Post-processing validation metadata
    logging.info("Accuracy on validation original task: %f", np.mean(validation_is_true))
    result_dict["validation_is_false"] = [1.0 - is_t for is_t in validation_is_true]
    
    result_dict["validation_unanswerable"] = [1.0 - is_a for is_a in validation_answerable]
    logging.info("Unanswerable proportion on validation: %f", np.mean(result_dict["validation_unanswerable"]))

    # Merge computed entropies into result dict
    if "uncertainty_measures" not in result_dict:
        result_dict["uncertainty_measures"] = {}
    if args.compute_predictive_entropy:
        result_dict["uncertainty_measures"].update(entropies)

    # Compute p_ik uncertainty measure if enabled
    if args.compute_p_ik and train_generations is not None:
        # Prepare training data for p_ik classifier
        train_is_true, train_embeddings, train_answerable = [], [], []
        for tid in train_generations:
            ml_answer = train_generations[tid]["most_likely_answer"]
            train_embeddings.append(ml_answer["embedding"])
            train_is_true.append(ml_answer["accuracy"])
            train_answerable.append(is_answerable(train_generations[tid]))
        
        train_is_false = [0.0 if is_t else 1.0 for is_t in train_is_true]
        train_unanswerable = [0.0 if is_a else 1.0 for is_a in train_answerable]
        logging.info("Unanswerable proportion on p_ik training: %f", np.mean(train_unanswerable))

        # Train and apply p_ik classifier
        logging.info("Starting training p_ik on train embeddings.")
        p_ik_predictions = get_p_ik(
            train_embeddings=train_embeddings,
            is_false=train_is_false,
            eval_embeddings=validation_embeddings,
            eval_is_false=result_dict["validation_is_false"]
        )
        result_dict["uncertainty_measures"]["p_ik"] = p_ik_predictions
        logging.info("Finished training p_ik on train embeddings.")

    # Save final results
    utils.save(result_dict, runid, "uncertainty_measures.pkl")
    logging.info("Compute uncertainty finished. Full results dict: %s", result_dict)

    # Save entailment model cache if applicable
    if args.compute_predictive_entropy and entailment_model is not None:
        entailment_model.save_prediction_cache()

    # Trigger result analysis if requested
    if args.analyze_run:
        logging.info(50 * "#X")
        logging.info("STARTING compute metrics!")
        analyze_run(runid)
        logging.info("FINISHED compute metrics!")
        logging.info(50 * "#X")

if __name__ == "__main__":
    parser = utils.get_parser(stages=["compute"])
    parser.add_argument("--runid", type=str, help="Identifier of the run to process")
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f"Unknown arguments: {unknown}")
    logging.info("Args: %s", args)
    main(args, runid="20250608_110824")
"""Implement semantic entropy calculations and entailment models."""
import os
import pickle
import logging

import numpy as np
import wandb
import torch
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.models.huggingface_models import HuggingfaceModel
from src.utils import gptapi as gpt
from src.utils import utils


# Device configuration: use CUDA if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseEntailment:
    """Base class for all entailment models, providing a cache saving interface."""
    def save_prediction_cache(self):
        """Save the prediction cache (to be implemented by subclasses)."""
        pass


class EntailmentDeberta(BaseEntailment):
    """Entailment model using DeBERTa-v2-xlarge-MNLI for text implication checks."""
    def __init__(self):
        # Initialize tokenizer and model from pretrained MNLI checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli"
        ).to(DEVICE)

    def check_implication(self, text1, text2, *args, **kwargs):
        """
        Check if text1 semantically entails text2 using DeBERTa-MNLI.
        
        Args:
            text1 (str): Premise text
            text2 (str): Hypothesis text
        
        Returns:
            int: 0 (contradiction), 1 (neutral), or 2 (entailment)
        """
        # Tokenize input pair and move to device
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(DEVICE)
        
        # Model inference
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Get prediction by softmax and argmax
        # MNLI labels: 0=contradiction, 1=neutral, 2=entailment
        largest_index = torch.argmax(F.softmax(logits, dim=1))  # pylint: disable=no-member
        prediction = largest_index.cpu().item()
        
        # Log detailed info if environment flag is set
        if os.environ.get('DEBERTA_FULL_LOG', False):
            logging.info('Deberta Input: %s -> %s', text1, text2)
            logging.info('Deberta Prediction: %s', prediction)
        
        return prediction


class EntailmentLLM(BaseEntailment):
    """Base class for LLM-based entailment models with prediction caching."""
    entailment_file = 'entailment_cache.pkl'  # Default cache file name

    def __init__(self, entailment_cache_id, entailment_cache_only):
        self.prediction_cache = self.init_prediction_cache(entailment_cache_id)
        self.entailment_cache_only = entailment_cache_only

    def init_prediction_cache(self, entailment_cache_id):
        if entailment_cache_id is None:
            return dict()

        logging.info('Restoring prediction cache from %s', entailment_cache_id)
        
        # Download cache file from wandb
        api = wandb.Api()
        run = api.run(entailment_cache_id)
        run.file(self.entailment_file).download(
            replace=True, exist_ok=False, root=wandb.run.dir
        )
        
        # Load cache from local file
        with open(f'{wandb.run.dir}/{self.entailment_file}', "rb") as infile:
            return pickle.load(infile)

    def save_prediction_cache(self):
        """Save current prediction cache to pickle file using utils.save."""
        utils.save(self.prediction_cache, self.entailment_file)

    def check_implication(self, text1, text2, example=None):
        if example is None:
            raise ValueError("Example must be provided for LLM entailment check")
        
        # Construct equivalence prompt
        prompt = self.equivalence_prompt(text1, text2, example['question'])
        logging.info('%s input: %s', self.name, prompt)
        
        # Use MD5 hash for prompt caching
        prompt_hash = gpt.md5hash(prompt)
        if prompt_hash in self.prediction_cache:
            logging.info('Restoring cached prediction (hash: %s)', prompt_hash)
            response = self.prediction_cache[prompt_hash]
        else:
            if self.entailment_cache_only:
                raise ValueError("Cache-only mode but no cached prediction found")
            
            # Call LLM for new prediction
            response, tokens = self.predict(prompt, temperature=0.02)
            self.prediction_cache[prompt_hash] = response
        
        logging.info('%s prediction: %s', self.name, response)
        
        # Parse response to get entailment label
        binary_response = response.lower()[:30]
        if 'entailment' in binary_response:
            return 2, tokens
        elif 'neutral' in binary_response:
            return 1, tokens
        elif 'contradiction' in binary_response:
            return 0, tokens
        else:
            logging.warning('Unrecognized response, defaulting to neutral: %s', response)
            return 1, tokens


class EntailmentGPT4(EntailmentLLM):
    """LLM-based entailment model using GPT-4."""
    def __init__(self, entailment_cache_id, entailment_cache_only):
        super().__init__(entailment_cache_id, entailment_cache_only)
        self.name = 'gpt-4o'  # Model name for API calls

    def equivalence_prompt(self, text1, text2, question):
        """
        Construct prompt to check if text1 entails text2 for the given question.
        
        Args:
            text1 (str): Answer 1
            text2 (str): Answer 2
            question (str): Original question
        
        Returns:
            str: Formatted prompt for LLM
        """
        return f"""We are evaluating answers to the question "{question}"
Here are two possible answers:
Possible Answer 1: {text1}
Possible Answer 2: {text2}
Does Possible Answer 1 semantically entail Possible Answer 2? Respond with entailment, contradiction, or neutral."""

    def predict(self, prompt, temperature):
        """
        Call GPT-4 API to get entailment prediction.
        
        Args:
            prompt (str): Input prompt
            temperature (float): Sampling temperature
        
        Returns:
            tuple: (str: model response, int: token count)
        """
        return gpt.predict(prompt, temperature, model=self.name)


class EntailmentGPT35(EntailmentGPT4):
    """LLM-based entailment model using GPT-3.5-Turbo (inherits prompt logic from GPT-4 class)."""
    def __init__(self, entailment_cache_id, entailment_cache_only):
        super().__init__(entailment_cache_id, entailment_cache_only)
        self.name = 'gpt-3.5-turbo'  # Override model name


class EntailmentGPT4Turbo(EntailmentGPT4):
    """LLM-based entailment model using GPT-4-Turbo (note: model name may need verification)."""
    def __init__(self, entailment_cache_id, entailment_cache_only):
        super().__init__(entailment_cache_id, entailment_cache_only)
        self.name = 'gpt-4o'  # Model name (ensure this matches actual API identifier)


def context_entails_response(context, responses, model):
    votes = []
    for response in responses:
        votes.append(model.check_implication(context, response))
    return 2 - np.mean(votes)


def get_semantic_ids(strings_list, model, strict_entailment=False, example=None):
    def are_equivalent(text1, text2):
        """Check if two texts are semantically equivalent."""
        implication_1 = model.check_implication(text1, text2, example=example)
        implication_2 = model.check_implication(text2, text1, example=example)  # pylint: disable=arguments-out-of-order
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

        if strict_entailment:
            return (implication_1 == 2) and (implication_2 == 2)
        else:
            # Equivalent if no contradiction and not both neutral
            return (0 not in [implication_1, implication_2]) and ([1, 1] != [implication_1, implication_2])

    # Initialize semantic IDs with -1 (unassigned)
    semantic_set_ids = [-1] * len(strings_list)
    next_id = 0  # Counter for new semantic IDs

    for i, string1 in enumerate(strings_list):
        if semantic_set_ids[i] == -1:
            # Assign new ID to unprocessed string
            semantic_set_ids[i] = next_id
            # Check all subsequent strings for equivalence
            for j in range(i + 1, len(strings_list)):
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids  # Ensure all strings are assigned
    return semantic_set_ids


def logsumexp(semantic_ids, log_likelihoods, agg='sum_normalized'):
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids))), "Semantic IDs must be 0-based consecutive"
    
    log_likelihood_per_semantic_id = []
    for uid in unique_ids:
        # Get indices of strings belonging to this semantic group
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        
        if agg == 'sum_normalized':
            # Normalize by total log probability before summing
            total_log_prob = np.log(np.sum(np.exp(log_likelihoods)))
            log_lik_norm = [ll - total_log_prob for ll in id_log_likelihoods]
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        else:
            raise ValueError(f"Unsupported aggregation method: {agg}")
        
        log_likelihood_per_semantic_id.append(logsumexp_value)
    
    return log_likelihood_per_semantic_id


def length_normalized_entropy(log_probs):
    return -np.sum(log_probs) / len(log_probs)


def semantic_entropy(log_probs):
    return -np.sum(np.exp(log_probs) * log_probs)



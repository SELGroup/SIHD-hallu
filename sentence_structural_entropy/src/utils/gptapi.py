"""Utilities for interacting with OpenAI GPT models, including API calls and hashing functions."""

import os
import hashlib
from tenacity import retry, wait_random_exponential, retry_if_not_exception_type
import logging

# Configure logging to INFO level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from openai import OpenAI


# Initialize OpenAI client with proxy configuration
CLIENT = OpenAI(
    base_url="",
    api_key="",
)


class KeyError(Exception):
    """Custom exception raised when OpenAI API key is not provided in environment variables."""
    pass


@retry(retry=retry_if_not_exception_type(KeyError), wait=wait_random_exponential(min=1, max=10))
def predict(prompt, temperature=1.0, model="gpt-4o"):
    """
    Generate predictions using OpenAI GPT models with retry mechanism.
    
    Args:
        prompt (str/list): Input prompt or list of message dictionaries
        temperature (float): Sampling temperature (0.0 to 2.0), higher values make output more random
        model (str): GPT model name (e.g., "gpt-4o", "gpt-3.5-turbo")
    
    Returns:
        str: Generated response content from the model
    
    Raises:
        KeyError: If OpenAI API key is not configured
    """
    if not CLIENT.api_key:
        raise KeyError("Need to provide OpenAI API key in environment variable `OPENAI_API_KEY`.")

    # Convert string prompt to message format if needed
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    # Call OpenAI chat completions API
    output = CLIENT.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=200,
        temperature=temperature,
    )
    response = output.choices[0].message.content
    logging.info(f"Model response (model={model}, temp={temperature}): {response}")
    return response


def md5hash(string):
    """
    Compute MD5 hash of a string and convert to integer.
    
    Args:
        string (str): Input string to hash
    
    Returns:
        int: Integer representation of MD5 hash value
    """
    return int(hashlib.md5(string.encode("utf-8")).hexdigest(), 16)
import os
import hashlib
from tenacity import retry, wait_random_exponential, retry_if_not_exception_type
import logging

# Configure logging to INFO level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from openai import OpenAI


# Initialize OpenAI client with proxy configuration
CLIENT = OpenAI(api_key=os.environ['OPENAI_API_KEY'])


class KeyError(Exception):
    pass


@retry(retry=retry_if_not_exception_type(KeyError), wait=wait_random_exponential(min=1, max=10))
def predict(prompt, temperature=1.0, model="gpt-4o"):
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
    return int(hashlib.md5(string.encode("utf-8")).hexdigest(), 16)

"""Implement HuggingfaceModel models."""

import copy
import logging
from collections import Counter
import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown,nvmlDeviceGetCount
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.models.base_model import BaseModel
from src.models.base_model import STOP_SEQUENCES


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops, tokenizer, match_on="text", initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        
        # Convert stop patterns to token tensors if using token matching mode
        if self.match_on == "tokens":
            self.stops = [
                torch.tensor(self.tokenizer.encode(stop_str)).to("cuda") 
                for stop_str in self.stops
            ]
            # Debug print for token conversion (original functionality preserved)
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
      
        del scores  # Unused parameter required by StoppingCriteria interface
        
        for stop in self.stops:
            if self.match_on == "text":
                # Decode generated tokens (excluding initial input) and check for string match
                generation = self.tokenizer.decode(
                    input_ids[0][self.initial_length:], 
                    skip_special_tokens=False
                )
                match = stop in generation
            elif self.match_on == "tokens":
                # Check if stop token sequence appears at the end of current input IDs
                # Note: Token matching can be ambiguous due to tokenizer behavior
                match = stop in input_ids[0][-len(stop):]
            else:
                raise ValueError(f"Unsupported matching mode: {self.match_on}")
            
            if match:
                return True
        return False


def remove_split_layer(device_map_in):

    device_map = copy.deepcopy(device_map_in)
    destinations = list(device_map.keys())
    
    # Count occurrences of top-level layer groups (e.g., "model.layers.0" -> "model.layers")
    layer_groups = [".".join(name.split(".")[:2]) for name in destinations]
    group_counts = Counter(layer_groups)

    found_split = False
    for layer_group, count in group_counts.items():
        # Skip groups with only one layer
        if count == 1:
            continue
        
        # Error if multiple split layers found (original validation preserved)
        if found_split:
            raise ValueError(
                "Multiple split layers detected.\n"
                f"Current layer group: {layer_group}\n"
                f"Original map: {device_map_in}\n"
                f"Adjusted map: {device_map}"
            )

        logging.info(f"Detected split layer group: {layer_group}")
        
        # Consolidate all layers in the group to a single device
        group_devices = []
        for name in list(device_map.keys()):
            if name.startswith(layer_group):
                group_devices.append(device_map.pop(name))
        
        # Use the last device in the group (original behavior preserved)
        device_map[layer_group] = group_devices[-1]
        found_split = True

    return device_map


def get_gpu_memory_info():
    nvmlInit()
    try:
        device_count = nvmlDeviceGetCount()
        gpu_indices = list(range(device_count))
        memory_info = {}
        
        for i in gpu_indices:
            handle = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(handle)
            # Convert free memory from bytes to GB with 2 decimal places
            free_memory_gb = round(info.free / (1024 ** 3), 2)
            memory_info[i] = float(f"{free_memory_gb:.2f}")
        
        return memory_info
    finally:
        nvmlShutdown()  # Ensure NVML cleanup even if errors occur


def set_max_memory(memory_info):
    return {card: f"{memory}GIB" for card, memory in memory_info.items()}

class HuggingfaceModel(BaseModel):

    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None):
        
        if max_new_tokens is None:
            raise ValueError("max_new_tokens must be specified")
        self.max_new_tokens = max_new_tokens

        # Use default stop sequences if specified
        if stop_sequences == "default":
            stop_sequences = STOP_SEQUENCES

        # Get GPU memory information and convert to model-compatible format
        gpu_free_memory = get_gpu_memory_info()
        max_memory = set_max_memory(gpu_free_memory)
        print(f"GPU memory allocation dict: {max_memory}")

        # Model loading logic for different architectures
        kwargs = {}  # Common arguments container for model initialization

        # --- Llama family models ---
        if "llama" in model_name.lower():
            # Handle quantization suffixes
            if model_name.endswith("-8bit"):
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                model_name = model_name[: -len("-8bit")]  # Remove quantization suffix
            elif model_name.endswith("-4bit"):
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                model_name = model_name[: -len("-4bit")]
            
            model_id = f"meta-llama/{model_name}"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                device_map="auto",
                token_type_ids=None
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                max_memory=max_memory,
                **kwargs
                # llm_int8_enable_fp32_cpu_offload=True  # Commented out as optional
            )

        # --- Falcon family models ---
        elif "falcon" in model_name.lower():
            print(f"Loading model: {model_name}")
            # Handle quantization suffixes
            if model_name.endswith("-8bit"):
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                model_name = model_name[: -len("-8bit")]
            if model_name.endswith("-4bit"):
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                model_name = model_name[: -len("-4bit")]
            
            model_id = f"tiiuae/{model_name}"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                device_map="auto",
                token_type_ids=None,
                clean_up_tokenization_spaces=False
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map="auto",
                max_memory=max_memory,
                **kwargs
            )

        # --- Phi family models ---
        elif "phi" in model_name.lower():
            print(f"Loading model: {model_name}")
            # Handle quantization suffixes
            if model_name.endswith("-8bit"):
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                model_name = model_name[: -len("-8bit")]
            elif model_name.endswith("-4bit"):
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                model_name = model_name[: -len("-4bit")]
            
            model_id = f"microsoft/{model_name}"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                device_map="auto",
                token_type_ids=None,
                clean_up_tokenization_spaces=False,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
                **kwargs
            )

        else:
            raise ValueError(f"Unsupported model family: {model_name}")

        # Post-initialization configuration
        self.model_name = model_name
        self.stop_sequences = stop_sequences + [self.tokenizer.eos_token] if stop_sequences else [self.tokenizer.eos_token]
        self.token_limit = 4096  # Default token limit for generation

    def predict(self, input_data, temperature, return_full=False):
        # Tokenize input and move to GPU
        inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")

        # Model-specific token handling
        eos_token_id = None
        pad_token_id = None
        model_families = ["falcon", "phi", "llama"]
        if any(family in self.model_name.lower() for family in model_families):
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]  # Remove deprecated token type IDs
            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = eos_token_id if eos_token_id is not None else None

        # Configure stopping criteria
        stopping_criteria = None
        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList([
                StoppingCriteriaSub(
                    stops=self.stop_sequences,
                    initial_length=len(inputs["input_ids"][0]),
                    tokenizer=self.tokenizer
                )
            ])

        # Generate text with model
        logging.debug(f"Generation temperature: {temperature:.2f}")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                temperature=temperature,
                do_sample=True,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id
            )

        # Validate token limit
        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                f"Generation exceeded token limit: {len(outputs.sequences[0])} > {self.token_limit}"
            )

        # Decode full generated text
        full_answer = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        if return_full:
            return full_answer

        # Extract answer by removing input prefix
        if full_answer.startswith(input_data):
            input_offset = len(input_data)
        else:
            input_offset = full_answer.find(input_data)
            if input_offset == -1:
                logging.warning("Input data not found in full answer, using entire text")
                input_offset = 0
        answer = full_answer[input_offset:]

        # Remove trailing stop sequences
        stop_at = len(answer)
        sliced_answer = answer
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                if answer.endswith(stop):
                    stop_at = len(answer) - len(stop)
                    sliced_answer = answer[:stop_at]
                    break
            # Verify stop sequence removal
            if not all(stop not in sliced_answer for stop in self.stop_sequences):
                logging.error(
                    f"Stop sequence removal incomplete!\n"
                    f"Original answer: {answer}\n"
                    f"Sliced answer: {sliced_answer}"
                )

        # Clean up whitespace
        sliced_answer = sliced_answer.strip()

        # Calculate generation statistics
        token_stop_index = self.tokenizer(
            full_answer[:input_offset + stop_at], return_tensors="pt"
        )["input_ids"].shape[1]
        n_input_token = len(inputs["input_ids"][0])
        n_generated = token_stop_index - n_input_token

        # Handle edge case with zero generated tokens
        if n_generated == 0:
            logging.warning("No new tokens generated, using stop sequence as fallback")
            n_generated = 1

        # Extract hidden states for embedding calculation
        hidden = outputs.decoder_hidden_states if "decoder_hidden_states" in outputs else outputs.hidden_states
        if len(hidden) == 1:
            logging.warning("Only one hidden state available, using first state")
            last_input = hidden[0]
        elif (n_generated - 1) >= len(hidden):
            logging.error("Generated tokens exceed hidden states, using last state")
            last_input = hidden[-1]
        else:
            last_input = hidden[n_generated - 1]

        # Extract last token embedding
        last_layer = last_input[-1]
        last_token_embedding = last_layer[:, -1, :].cpu()

        # Calculate log likelihoods
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        log_likelihoods = [score.item() for score in transition_scores[0]]
        if len(log_likelihoods) == 1:
            logging.warning("Only one log likelihood value available")
        else:
            log_likelihoods = log_likelihoods[:n_generated]

        # Handle max token limit interruption
        if len(log_likelihoods) == self.max_new_tokens:
            logging.warning("Generation stopped by max_new_tokens limit")

        if not log_likelihoods:
            raise ValueError("No log likelihoods calculated")

        return sliced_answer, log_likelihoods, last_token_embedding

    def get_p_true(self, input_data):
        # Prepare prompt with 'A' suffix
        input_data += " A"
        tokenized_prompt = self.tokenizer(input_data, return_tensors="pt").to("cuda")["input_ids"]
        
        # Mask all tokens except the last one for loss calculation
        target_ids = tokenized_prompt.clone()
        target_ids[0, :-1] = -100  # -100 is ignored in loss calculation

        # Calculate loss for 'A' completion
        with torch.no_grad():
            model_output = self.model(tokenized_prompt, labels=target_ids)
        
        return -model_output.loss.item()

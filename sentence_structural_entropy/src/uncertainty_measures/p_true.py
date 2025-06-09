import logging

def construct_few_shot_prompt(
    *, 
    model, 
    dataset, 
    indices, 
    prompt: str, 
    brief: bool, 
    brief_always: bool, 
    make_prompt, 
    num_generations: int, 
    metric
) -> tuple[str, dict, int]:
    few_shot_prompt = []
    all_responses = {}
    
    for example_idx, dataset_idx in enumerate(indices):
        current_prompt_segments = []
        example = dataset[dataset_idx]
        question = example["question"]
        context = example["context"]

        # Add separator between examples (except first)
        if example_idx != 0:
            current_prompt_segments.append('\n')

        # Add question section
        current_prompt_segments.append(f"Question: {question}")
        current_prompt_segments.append("\nBrainstormed Answers: ")

        # Construct and log current question prompt
        current_question = make_prompt(context, question, None, brief, brief_always)
        local_prompt = prompt + current_question
        logging.info("P_TRUE >> Current Prompt: %s", prompt.ljust(25))
        logging.info("P_TRUE >> Current Question: %s", current_question.ljust(25))

        responses = []
        most_likely_response = ""
        is_correct = False

        # Generate multiple responses with different temperatures
        for gen_idx in range(num_generations + 1):
            temperature = 0.1 if gen_idx == 0 else 1.0  # First gen: low temp, others: high temp
            response, _, _ = model.predict(local_prompt, temperature)
            logging.info("P_TRUE >> Current Response: %s", response.ljust(25))

            responses.append(response)
            current_prompt_segments.append(f"{response.strip()} \n")

            # Handle most likely response (first generation)
            if gen_idx == 0:
                most_likely_response = response
                is_correct = metric(response, example, model)
                true_answers = example['answers']['text']
                logging.info("P_TRUE >> LOW-T >> true answer: %s", str(true_answers).ljust(35))
                logging.info("P_TRUE >> LOW-T >> accuracy: %s", str(is_correct).ljust(35))

        # Store response metadata
        all_responses[dataset_idx] = {
            "responses": responses,
            "most_likely_response": most_likely_response,
            "is_correct": is_correct
        }

        # Add correctness verification section
        current_prompt_segments.extend([
            f"Possible answer: {most_likely_response}\n",
            "Is the possible answer:\n",
            "A) True\n",
            "B) False\n",
            "The possible answer is: ",
            "A" if is_correct else "B"
        ])

        # Check token limit constraints
        full_prompt = ''.join(few_shot_prompt + current_prompt_segments)
        prompt_token_count = len(model.tokenizer.encode(full_prompt))
        max_allowed_tokens = prompt_token_count + num_generations * model.max_new_tokens + 200  # 200 buffer

        if max_allowed_tokens < model.token_limit:
            few_shot_prompt.extend(current_prompt_segments)
        else:
            logging.warning("Truncating p_true prompt at example %d (token limit exceeded)", example_idx)
            break

    return ''.join(few_shot_prompt), all_responses, example_idx


def calculate_p_true(
    model, 
    question: str, 
    most_probable_answer: str, 
    brainstormed_answers: list[str], 
    few_shot_prompt: str = "", 
    hint: bool = False
) -> float:
    # Initialize prompt with few-shot examples if available
    base_prompt = f"{few_shot_prompt}\n" if few_shot_prompt else ""

    # Build question and answer section
    base_prompt += f"Question: {question}\n"
    base_prompt += "Brainstormed Answers: "
    for answer in brainstormed_answers + [most_probable_answer]:
        base_prompt += f"{answer.strip()}\n"
    base_prompt += f"Possible answer: {most_probable_answer}\n"

    # Add verification question
    if not hint:
        base_prompt += (
            "Is the possible answer:\n"
            "A) True\n"
            "B) False\n"
            "The possible answer is:"
        )
    else:
        base_prompt += (
            "Do the brainstormed answers match the possible answer? "
            "Respond with A if they do, B if they do not. Answer:"
        )

    # Get p_true score from model
    log_prob = model.get_p_true(base_prompt)
    return log_prob
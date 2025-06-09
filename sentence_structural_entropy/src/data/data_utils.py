import os
import json
import hashlib
import datasets


def load_ds(dataset_name, seed):
    train_dataset, validation_dataset = None, None

    # --- SQuAD v2 Dataset ---
    if dataset_name == "squad":
        # Load SQuAD v2 dataset with training/validation splits
        dataset = datasets.load_dataset("rajpurkar/squad_v2")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]

    # --- SVAMP Dataset ---
    elif dataset_name == 'svamp':
        # Load SVAMP (Single Variable Algebra Math Problems) dataset
        dataset = datasets.load_dataset('ChilleD/SVAMP')
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

        # Reformat dataset to standard question-answer format
        reformat = lambda x: {
            'question': x['Question'], 
            'context': x['Body'], 
            'type': x['Type'],
            'equation': x['Equation'], 
            'id': x['ID'],
            'answers': {'text': [str(x['Answer'])]}  # Convert answer to string list
        }

        # Apply reformatting to both splits
        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    # --- Natural Questions (NQ) Dataset ---
    elif dataset_name == 'nq':
        # Load NQ Open dataset with training/validation splits
        dataset = datasets.load_dataset("nq_open")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]

        # Helper function to generate unique ID using MD5 hash
        md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))

        # Reformat dataset to standard question-answer format
        reformat = lambda x: {
            'question': x['question'] + '?',  # Ensure question ends with '?'
            'answers': {'text': x['answer']},  # Directly use provided answers
            'context': '',  # NQ has no explicit context field
            'id': md5hash(str(x['question'])),  # Unique ID from question text
        }

        # Apply reformatting to both splits
        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    # --- TriviaQA Dataset ---
    elif dataset_name == "trivia_qa":
        # Load TriviaQA dataset in SQuAD-compatible format
        dataset = datasets.load_dataset('TimoImhof/TriviaQA-in-SQuAD-format')['unmodified']
        # Split into training (80%) and validation (20%) sets using given seed
        dataset = dataset.train_test_split(test_size=0.2, seed=seed)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']

    # --- BioASQ Dataset ---
    elif dataset_name == "bioasq":
        # Construct path to BioASQ training data
        current_working_directory = os.getcwd()
        print("data_utils.py working directory: ", current_working_directory)
        relative_path = 'src/data/bioasq/training11b.json'
        path = os.path.join(current_working_directory, relative_path)
        print("dataset bioasq path:", path)

        # Load and parse JSON data
        with open(path, "rb") as file:
            data = json.load(file)
        questions = data["questions"]

        # Initialize dataset dictionary with standard fields
        dataset_dict = {
            "question": [],
            "answers": [],
            "id": []
        }

        for question in questions:
            # Skip questions without exact answer field
            if "exact_answer" not in question:
                continue

            # Extract question text
            dataset_dict["question"].append(question["body"])

            # Process exact answers (handle nested list cases)
            if isinstance(question['exact_answer'], list):
                exact_answers = [
                    ans[0] if isinstance(ans, list) else ans  # Flatten nested lists
                    for ans in question['exact_answer']
                ]
            else:
                exact_answers = [question['exact_answer']]

            # Add answers with dummy answer_start positions
            dataset_dict["answers"].append({
                "text": exact_answers,
                "answer_start": [0] * len(question["exact_answer"])
            })

            # Add question ID and dummy context
            dataset_dict["id"].append(question["id"])
            dataset_dict["context"] = [None] * len(dataset_dict["id"])  # No context

        # Convert dict to Hugging Face Dataset and split
        dataset = datasets.Dataset.from_dict(dataset_dict)
        dataset = dataset.train_test_split(test_size=0.8, seed=seed)  # 20% training, 80% validation
        train_dataset = dataset['train']
        validation_dataset = dataset['test']

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_dataset, validation_dataset
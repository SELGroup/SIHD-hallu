from datetime import timedelta
import os
import json
import re
import time
from matplotlib.streamplot import InvalidIndexError
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
import logging
from tenacity import (
    retry,
    wait_random_exponential,
    before_sleep_log,
)
from typing import List
from DeepSeek_R1_data import DS_data
from o1_data import o1data
from HCSE import main
import utils
import json
import numpy as np
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MAX_TOKENS_BIO = 600
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])


class EmptyResponseError(Exception):
    pass


class InvalidAnswerError(Exception):
    pass


class BipartiteProcessor:

    def __init__(self, num_generations, data: List, output_path: str):
        self.num_generations = num_generations
        self.data = data
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.master_data = {
            "metadata": {
                "version": "2.1",
                "total_data": 21,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "data_items": [],
        }

    @retry(
        wait=wait_random_exponential(min=1, max=15),
        before_sleep=before_sleep_log(logging, logging.WARNING),
    )
    def llm_predict(self, prompt, mode):
        if not isinstance(prompt, (str, list)):
            raise ValueError(
                f"Invalid prompt type: {type(prompt)}, expected str or list"
            )
        if isinstance(prompt, str):
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [
                {"role": msg["role"], "content": str(msg["content"])}
                for msg in prompt
                if isinstance(msg, dict) and "role" in msg and "content" in msg
            ]
        logging.info(f"{messages}")

        if mode == "generate":
            output = client.chat.completions.create(
                model="deepseek-reasoner",  # o1
                messages=messages,
                max_tokens=MAX_TOKENS_BIO,
            )
        elif mode == "inference":
            output = client.chat.completions.create(
                model="deepseek-reasoner",  # o1
                messages=messages,
                temperature=0,
                seed=666,
            )
        else:
            raise InvalidIndexError(
                "An invalid 'mode' parameter. It should be either 'generate' or 'inference'."
            )

        response = output.choices[0].message.content

        if mode == "generate":
            if not response.strip():
                raise EmptyResponseError()

        elif mode == "inference":
            clean_res = response.strip().lower()[:3]
            if not (clean_res.startswith("yes") or clean_res.startswith("no")):
                raise InvalidAnswerError()

        logging.info(f"{mode} API response: {response}")
        if mode == "generate":
            return response
        elif mode == "inference":
            return response.strip().lower()[:3]

    def _generate_responses(self, question: str, num_generation) -> List[str]:
        question_prompt = [
            {
                "role": "system",
                "content": "Generate biographies that user ask about.",
            },
            {"role": "user", "content": question},
        ]

        return [
            self.llm_predict(question_prompt.copy(), "generate")
            for _ in range(num_generation)
        ]

    def _build_adjacency_matrix(
        self, responses: List[str], claims: List[str], num_generation
    ) -> np.ndarray:
        adj_matrix = np.zeros((len(claims), num_generation), dtype=np.uint8)

        for claim_idx, claim in enumerate(claims):
            for resp_idx, response in enumerate(
                tqdm(responses, desc="Building Adjacency", leave=False)
            ):
                interface_prompt = f"Context: {response}\nClaim: {claim}\nIs the Claim supported by the Context above? Answer start with Yes or No:"
                answer = self.llm_predict(interface_prompt, "inference")
                adj_matrix[claim_idx, resp_idx] = int(answer.startswith("yes"))

        return adj_matrix

    def _matrix_to_b64(self, matrix: np.ndarray) -> str:
        return matrix.tobytes().decode("latin-1")

    def _extract_name_from_question(self, question: str) -> str:
        match = re.search(r"Who is ([A-Za-z\s]+)\?", question)
        if match:
            return match.group(1).strip()
        return None

    def process_all_data(self, num_datum):
        for datum_idx, datum in enumerate(tqdm(self.data, desc="Processing Data")):

            if datum_idx > num_datum - 1:
                break

            question = datum[1]
            claims = datum[4]
            claim_labels = datum[5]

            responses = self._generate_responses(question, self.num_generations)

            adj_matrix = self._build_adjacency_matrix(
                responses, claims, self.num_generations
            )

            data_item = {
                "datum_id": datum_idx,
                "question": question,
                "metadata": {
                    "num_responses": len(responses),
                    "num_claims": len(claims),
                    "matrix_shape": adj_matrix.shape,
                },
                "claims": claims,
                "claim_labels": claim_labels,
                "responses": responses,
                "adjacency_matrix": adj_matrix.tobytes().decode("latin-1"),
            }

            self.master_data["data_items"].append(data_item)

        self._save_as_json()

    def _save_as_json(self):
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.master_data, f, ensure_ascii=False, indent=2)


class DataLoader:

    def __init__(self, file_path: str = "run_recoed\\graphs.json"):
        self.file_path = Path(file_path)
        self.metadata = {}
        self.data_items = []

    def load_data(self) -> bool:
        if not self.file_path.exists():
            raise FileNotFoundError(f"File {self.file_path} does not exist")

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"{str(e)}")

        self.metadata = raw_data.get("metadata", {})

        self.data_items = []
        for item in raw_data.get("data_items", []):
            processed = self._process_item(item)
            self.data_items.append(processed)

        return True

    def _format_with_commas(self, matrix: np.ndarray) -> str:
        rows = []
        for row in matrix.tolist():
            formatted_row = ",".join(map(str, row))
            rows.append(f"[{formatted_row}]")

        return "[\n" + ",\n".join(rows) + "\n]"

    def _process_item(self, item: dict) -> dict:

        matrix_str = item.get("adjacency_matrix", "")
        matrix_bytes = matrix_str.encode("latin-1")

        matrix_shape = item.get("metadata", {}).get("matrix_shape")

        matrix = np.frombuffer(matrix_bytes, dtype=np.uint8).reshape(matrix_shape)

        return {
            "datum_id": item.get("datum_id", -1),
            "question": item.get("question", ""),
            "metadata": item.get("metadata", {}),
            "claims": item.get("claims", []),
            "claim_labels": item.get("claim_labels", []),
            "responses": item.get("responses", []),
            "adjacency_matrix": matrix,
        }

    @property
    def num_items(self) -> int:

        return len(self.data_items)

    def get_item(self, index: int) -> dict:

        if 0 <= index < self.num_items:
            return self.data_items[index]
        raise IndexError(f"The index is out of range (0-{self.num_items-1})")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def calculate_result(loader):
    adjacency_matrix_COLUMNS = loader.get_item(0)["adjacency_matrix"].shape[1]
    logging.info(f"adjacency_matrix_COLUMNS: {adjacency_matrix_COLUMNS}")

    all_labels, all_uncertainties = [], []
    for i in range(loader.num_items):
        sample = loader.get_item(i)
        all_labels += sample["claim_labels"]

        matrix = sample["adjacency_matrix"][:, :5]
        logging.info(f"Sub-matrix shape: {matrix.shape}")
        statement_entropy = main(matrix)

        for stmt, entropy in statement_entropy.items():
            logging.info(f"Statement {stmt}: {entropy:.6f}")
            all_uncertainties.append(entropy)
    logging.info(f"all labels: {all_labels} {len(all_labels)}")
    logging.info(f"all uncertainties: {all_uncertainties}  {len(all_uncertainties)}")

    metrics = utils.get_metrics(all_labels, all_uncertainties)
    metrics_file_path = os.getcwd() + "/run_record/metrics.json"

    with open(metrics_file_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)


if __name__ == "__main__":
    dir_path = os.path.join(os.getcwd() + "/run_record")

    time_start = time.time()
    num_generations = 5

    time_start = time.time()

    processor = BipartiteProcessor(num_generations, DS_data, dir_path + f"/graphs.json")
    processor.process_all_data(num_datum=len(DS_data))

    loader = DataLoader(dir_path + f"/graphs.json")

    try:
        if loader.load_data():
            logging.info(f"Successfully loaded {loader.num_items} data items.")
            calculate_result(loader)

    except Exception as e:
        logging.info(f"error: {str(e)}")

    time_end = time.time()
    time_diff_seconds = time_end - time_start
    time_diff = timedelta(seconds=time_diff_seconds)
    logging.info(f"Total time taken {time_diff}")

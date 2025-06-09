import gc
import logging
import random
from tqdm import tqdm
import numpy as np
import torch
from src.data.data_utils import load_ds
from src.utils import utils
from src.uncertainty_measures import p_true as p_true_utils
from uncertainty_quantification import main as main_compute
import time
from typing import Dict, List, Tuple, Optional
runid = time.strftime("%Y%m%d_%H%M%S")

class ExperimentRunner:
    """Main class for running QA answer generation experiments."""

    def __init__(self, args):
        self.args = args
        self.logger = self._setup_logger()
        self.model = None
        self.experiment_details = {"args": args}

    def _setup_logger(self) -> logging.Logger:
        """Configure logging system."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("experiment.log"), logging.StreamHandler()],
        )
        return logging.getLogger()


    def _load_datasets(self) -> Tuple[List, List]:
        if args.dataset == "svamp":
            if not args.use_context:
                logging.info("Forcing `use_context=True` for svamp dataset.")
                args.use_context = True
        elif args.dataset == "squad":
            if not args.answerable_only:
                logging.info("Forcing `answerable_only=True` for squad dataset.")
                args.answerable_only = True
        """Load and prepare datasets based on configuration."""
        train, val = load_ds(
            self.args.dataset,
            add_options=self.args.use_mc_options,
            seed=self.args.random_seed,
        )

        if self.args.ood_train_dataset:
            self.logger.warning(f"Using OOD dataset {self.args.ood_train_dataset}")
            train, _ = load_ds(
                self.args.ood_train_dataset,
                add_options=self.args.use_mc_options,
                seed=self.args.random_seed,
            )

        return train, val

    def _generate_responses(
        self, prompt: str, example: Dict, num_generations: int
    ) -> Tuple[Dict, List]:
        """Generate model responses for a single example."""
        
        full_responses = []
        most_likely_answer = None
        metric = utils.get_metric(self.args.metric)

        for i in range(num_generations):
            temp = 0 if i == 0 else self.args.temperature
            answer, logprobs, emb = self.model.predict(prompt, temp)
            emb = emb.cpu() if emb is not None else None
            self.logger.info(f"temp:{temp},Answer: {answer}")
            acc = (
                metric(answer, example, self.model)
                if (
                    example["answers"]["text"]
                    and (self.args.compute_accuracy_at_all_temps or i == 0)
                )
                else 0.0
            )
            if i == 0:
                logging.info(f"acc: %f",acc)
            if i == 0:
                most_likely_answer = {
                    "response": answer,
                    "token_log_likelihoods": logprobs,
                    "embedding": emb,
                    "accuracy": acc,
                }
            else:
                full_responses.append((answer, logprobs, emb, acc))

        return most_likely_answer, full_responses

    def _construct_prompt(self, example):
        """根据示例构建prompt"""
        question, context = example["question"], example['context']
        make_prompt = utils.get_make_prompt(args)
        current_input = make_prompt(
                context, question, None, utils.BRIEF_PROMPTS['chat'], args.brief_always and args.enable_brief)
        local_prompt = utils.BRIEF_PROMPTS['chat'] + current_input

        logging.info('Current input: '.ljust(15) + current_input)
        return local_prompt

    def _process_dataset_split(
        self, dataset: List, split_name: str, p_true_prompt: Optional[str] = None
    ) -> Dict:
        """Process a single dataset split (train/validation)."""
        self.logger.info(f"Processing {split_name} split")
        results = {
            "accuracies": [],
            "generations": {},
            "p_trues": [] if split_name == "validation" else None,
        }

        indices = random.sample(
            range(len(dataset)), min(self.args.num_samples, len(dataset))
        )

        for idx in tqdm(indices):
            example = dataset[idx]
            prompt = self._construct_prompt(example)

            num_gens = (
                1
                if (
                    split_name == "train"
                    and self.args.get_training_set_generations_most_likely_only
                )
                else self.args.num_generations + 1
            )

            ml_answer, responses = self._generate_responses(prompt, example, num_gens)

            results["generations"][example["id"]] = {
                "question": example["question"],
                "context": example["context"],
                "most_likely_answer": ml_answer,
                "reference": utils.get_reference(example),
                "responses": responses,
            }

            results["accuracies"].append(ml_answer["accuracy"])

            if split_name == "validation" and p_true_prompt:
                p_true = p_true_utils.calculate_p_true(
                    self.model,
                    example["question"],
                    ml_answer["response"],
                    [r[0] for r in responses],
                    p_true_prompt,
                    hint=self.args.p_true_hint,
                )
                results["p_trues"].append(p_true)

            if (idx + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        return results

    def run(self) -> None:
        """Main execution method for the experiment."""
        self.logger.info("Starting experiment run")
        try:
            self.model = utils.init_model(self.args)

            train, val = self._load_datasets()
            p_true_prompt = self._prepare_p_true_prompt(train)

            # Process both datasets
            for split_name, dataset in [("train", train), ("validation", val)]:
                results = self._process_dataset_split(
                    dataset,
                    split_name,
                    p_true_prompt if split_name == "validation" else None,
                )

                self._save_results(split_name, results)
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}", exc_info=True)
            raise
        finally:
            self.logger.info("sample completed, cleaning up resources")
            self._cleanup()

    def _prepare_p_true_prompt(self, dataset: List) -> str:
        """Prepare few-shot prompt for p_true calculation."""
        answerable_indices, _ = utils.split_dataset(dataset)
        p_true_indices = random.sample(answerable_indices, self.args.p_true_num_fewshot)

        prompt, responses, len_p = p_true_utils.construct_few_shot_prompt(
            model=self.model,
            dataset=dataset,
            indices=p_true_indices,
            prompt=utils.BRIEF_PROMPTS['chat'],
            brief=utils.BRIEF_PROMPTS['chat'],
            brief_always=self.args.brief_always and self.args.enable_brief,
            make_prompt=utils.get_make_prompt(self.args),
            num_generations=self.args.num_generations,
            metric=utils.get_metric(self.args.metric),
        )
        
        return prompt

    def _save_results(self, split_name: str, results: Dict) -> None:
        """Save results locally"""
        utils.save(results['generations'], runid,f'{split_name}_generations.pkl')
        
        accuracy = np.mean(results['accuracies'])
        self.logger.info(f"Overall {split_name} split accuracy: {accuracy}")
        
        if split_name == 'validation':
            uncertainty = {
                'p_false': [1 - p for p in results['p_trues']],
                'p_true': [1 - np.exp(p) for p in results['p_trues']]
            }
            utils.save({'uncertainty_measures': uncertainty},runid, 'uncertainty_measures.pkl')

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self.model:
            del self.model
        gc.collect()
        torch.cuda.empty_cache()


def main(args):
    runner = ExperimentRunner(args)
    runner.run()


if __name__ == "__main__":
    time_start = time.time()
    parser = utils.get_parser()
    args, unknown = parser.parse_known_args()

    if unknown:
        raise ValueError(f"Unknown args: {unknown}")

    logging.info("Starting experiment with args: %s", args)
    main(args)

    if args.compute_uncertainties:
        gc.collect()
        torch.cuda.empty_cache()
        logging.info("Starting uncertainty computation")
        main_compute(args,runid)
        logging.info("Finished uncertainty computation")

    time_end = time.time()
    hours, minutes = divmod(int(time_end - time_start) // 60, 60)
    logging.info(f"{args.model_name} total run time: {hours}-{minutes} (hours-minutes)")



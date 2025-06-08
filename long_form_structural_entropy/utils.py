"""Utility functions."""


import functools
import logging
import numpy as np
from eval_utils import (
    bootstrap,
    compatible_bootstrap,
    auroc,
    accuracy_at_quantile,
    area_under_thresholded_accuracy,
)


MAJOR = "Major False"
MINOR = "Minor False"

def set_logging():
    """Set up logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

set_logging()
def get_metrics(all_labels, all_uncertainties):
    all_labels = np.array(all_labels)
    all_uncertainties = np.array(all_uncertainties)

    rng = np.random.default_rng(41)
    # Base accuracy of GPT-4 propositions correctly.
    out = dict(uncertainties=all_uncertainties)

    for wrongness in ["major_only", "minor_and_major"]:
        out[wrongness] = dict()

        if wrongness == "major_only":
            # Only MAJOR becomes False.
            labels = [True if lab != MAJOR else False for lab in all_labels]
        elif wrongness == "minor_and_major":
            # Only True is True. Both MINOR and MAJOR map to false.
            labels = [True if lab == "True" else False for lab in all_labels]
        else:
            raise ValueError
        labels = np.array(labels)
        assert set(labels) in [
            {True},
            {False},
            {True, False},
        ], f"labels set is {set(labels)}"

        out[wrongness]["per_question"] = dict(labels=labels)
        out[wrongness]["performance"] = dict(
            mean=np.mean(labels), 
            bootstrap=bootstrap(np.mean, rng)(labels),
        )

        eval_metrics = dict(
            zip(
                ["auroc", "area_under_thresholded_accuracy", "mean_uncertainty"],
                list(
                    zip(
                        [auroc, area_under_thresholded_accuracy, np.mean],
                        [compatible_bootstrap, compatible_bootstrap, bootstrap],
                    )
                ),
            )
        )

        answer_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        for answer_fraction in answer_fractions:
            key = f"accuracy_at_{answer_fraction}_answer_fraction"
           

            eval_metrics[key] = [
                functools.partial(accuracy_at_quantile, quantile=answer_fraction),
                compatible_bootstrap,
            ]

        fargs = {
            "auroc": [labels, all_uncertainties],  # pos_label=0
            "area_under_thresholded_accuracy": [labels, all_uncertainties],
            "mean_uncertainty": [all_uncertainties],
        }

    
        for answer_fraction in answer_fractions:
            fargs[f"accuracy_at_{answer_fraction}_answer_fraction"] = [
                labels,
                all_uncertainties,
            ]

        out[wrongness]["uncertainty"] = {}
        for fname, (function, bs_function) in eval_metrics.items():
            metric_i = function(*fargs[fname])
            logging.info("%s: %f", fname, metric_i)
            out[wrongness]["uncertainty"][fname] = {
                "mean": metric_i,
                "bootstrap": bs_function(function, rng)(*fargs[fname]),
            }
    return out








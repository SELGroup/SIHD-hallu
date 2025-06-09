import functools
import logging
import numpy as np
import scipy
from sklearn import metrics


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




def bootstrap(function, rng, n_resamples=1000):
    def inner(data):
        bs = scipy.stats.bootstrap(
            (data, ), function, n_resamples=n_resamples, confidence_level=0.95,
            random_state=rng)
        return {
            'std_err': bs.standard_error,
            'low': bs.confidence_interval.low,
            'high': bs.confidence_interval.high
        }
    return inner


def auroc(y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score,pos_label=0)
    del thresholds
    return metrics.auc(fpr, tpr)


def accuracy_at_quantile(accuracies, uncertainties, quantile):
    cutoff = np.quantile(uncertainties, quantile)
    select = uncertainties <= cutoff 
    return np.mean(accuracies[select])


def area_under_thresholded_accuracy(accuracies, uncertainties):
    quantiles = np.linspace(0.1, 1, 20)
    select_accuracies = np.array([accuracy_at_quantile(accuracies, uncertainties, q) for q in quantiles])
    dx = quantiles[1] - quantiles[0]
    area = (select_accuracies * dx).sum()
    return area


def compatible_bootstrap(func, rng):
    def helper(y_true_y_score):
        # this function is called in the bootstrap
        y_true = np.array([i['y_true'] for i in y_true_y_score])
        y_score = np.array([i['y_score'] for i in y_true_y_score])
        out = func(y_true, y_score)
        return out

    def wrap_inputs(y_true, y_score):
        return [{'y_true': i, 'y_score': j} for i, j in zip(y_true, y_score)]

    def converted_func(y_true, y_score):
        y_true_y_score = wrap_inputs(y_true, y_score)
        return bootstrap(helper, rng=rng)(y_true_y_score)
    return converted_func



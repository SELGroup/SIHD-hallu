import json
import numpy as np
import argparse
import os
import json
from sklearn import metrics
import pickle


def load_data(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def calculate_auroc(y_true, y_score):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
    auroc_value = metrics.auc(fpr, tpr)
    return fpr, tpr, auroc_value


def AUROC(uncertainty_measures_json_path):
    data = load_data(uncertainty_measures_json_path)
    y_true = data["validation_is_false"]  # 1 = false
    metrics_list = [
        "p_ik",
        "p_true",
        "length_normalized_entropy",
        "semantic_entropy",
        "structural_entropy",
    ]

    AUROC_values = {}
    fpr_dict = {}
    tpr_dict = {}

    for metric in metrics_list:
        y_score = data["uncertainty_measures"][metric]
        # if metric == "structrual_entropy":
        #     y_score = [-val for val in y_score]
        fpr, tpr, auroc_value = calculate_auroc(y_true, y_score)
        AUROC_values[metric] = auroc_value
        fpr_dict[metric] = fpr
        tpr_dict[metric] = tpr

    for metric, auroc_value in AUROC_values.items():
        print(f"{metric} AUROC: {auroc_value:.4f}")
    return AUROC_values


def calculate_accuracy_at_quantile(accuracies, uncertainties, quantile):
    cutoff = np.quantile(uncertainties, quantile)
    select = uncertainties <= cutoff
    selected_indices = np.where(select)[0]
    return (
        np.mean(np.array(accuracies)[selected_indices])
        if selected_indices.size > 0
        else 0
    )


def calculate_aurac(accuracies, uncertainties, quantiles):
    accuracies_at_quantiles = [
        calculate_accuracy_at_quantile(accuracies, uncertainties, q) for q in quantiles
    ]
    dx = quantiles[1] - quantiles[0]
    area = (np.array(accuracies_at_quantiles) * dx).sum()
    return area


def AURAC(uncertainty_measures_json_path):
    data = load_data(uncertainty_measures_json_path)
    accuracies = [1 - is_false for is_false in data["validation_is_false"]]
    metrics_list = [
        "p_ik",
        "p_true",
        "length_normalized_entropy",
        "semantic_entropy",
        "structural_entropy",
    ]
    AURAC_values = {}
    quantiles = np.linspace(0.1, 1, 20)

    for metric in metrics_list:
        uncertainties = data["uncertainty_measures"][metric]
        aurac_value = calculate_aurac(
            accuracies, uncertainties, quantiles
        )
        AURAC_values[metric] = aurac_value

    for metric, aurac_value in AURAC_values.items():
        print(f"{metric} AURAC: {aurac_value:.4f}")
    return AURAC_values


def convert_pkl(semantic_cluster_path, semantic_cluster_json_path):
    # Read the pickle file
    with open(semantic_cluster_path, "rb") as f:
        data = pickle.load(f)
    
    # Custom encoder to handle numpy arrays
    class NumpyArrayEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    # Convert data to JSON format
    json_data = json.dumps(data, cls=NumpyArrayEncoder, ensure_ascii=False, indent=4)

    # Write to JSON file
    with open(semantic_cluster_json_path, "w", encoding="utf-8") as f:
        f.write(json_data)
    print(f"uncertainty_measures.pkl convert successfully.")


def analyze_run(runid):
    dir_path = os.path.join(
        os.getcwd(),
        f"run_record/{runid}/",
    )
    uncertainty_measures_path = dir_path + "uncertainty_measures.pkl"
    uncertainty_measures_json_path = dir_path + "uncertainty_measures.json"
    convert_pkl(uncertainty_measures_path, uncertainty_measures_json_path)

    AUROC_values = AUROC(uncertainty_measures_json_path)
    AURAC_values = AURAC(uncertainty_measures_json_path)
    all_results = {"AUROC": AUROC_values, "AURAC": AURAC_values}
    with open(dir_path + "all_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute metrics")
    parser.add_argument("--runid", type=str, help="Please enter a valid file directory")
    args = parser.parse_args()
    analyze_run(runid=args.runid)
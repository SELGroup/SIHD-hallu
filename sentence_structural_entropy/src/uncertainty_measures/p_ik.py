import logging
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def get_p_ik(train_embeddings, is_false, eval_embeddings=None, eval_is_false=None):
    # Validate input dimensions
    assert len(train_embeddings) > 0, "Empty training embeddings list"
    embedding_dim = train_embeddings[0].shape[-1]
    for emb in train_embeddings:
        assert emb.shape[-1] == embedding_dim, "Inconsistent embedding dimensions"

    # Log base task accuracy
    logging.info(f"Base model task accuracy: {1 - torch.tensor(is_false).mean():.4f}")  # pylint: disable=no-member

    # Convert training embeddings to numpy array
    train_embeddings_tensor = torch.cat(train_embeddings, dim=0)  # Combine list of tensors [N, D]
    embeddings_array = train_embeddings_tensor.cpu().float().numpy()  # Move to CPU and convert to numpy

    # Split training data into train/test subsets
    X_train, X_test, y_train, y_test = train_test_split(  # pylint: disable=invalid-name
        embeddings_array, is_false, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # pylint: disable=invalid-name
    X_test_scaled = scaler.transform(X_test)  # pylint: disable=invalid-name

    # Train initial classifier on split training data
    base_model = LogisticRegression(max_iter=1000)
    base_model.fit(X_train_scaled, y_train)

    # Prepare evaluation data if provided
    metrics = {}
    y_preds_proba = {}
    
    if eval_embeddings is not None and eval_is_false is not None:
        # Validate evaluation inputs
        assert len(eval_embeddings) > 0, "Empty evaluation embeddings list"
        for emb in eval_embeddings:
            assert emb.shape[-1] == embedding_dim, "Evaluation embedding dimension mismatch"

        # Process evaluation embeddings
        eval_embeddings_tensor = torch.cat(eval_embeddings, dim=0)  # Combine evaluation tensors
        X_eval = eval_embeddings_tensor.cpu().float().numpy()  # Convert to numpy
        X_eval_scaled = scaler.transform(X_eval)  # Apply training scaler

        # Train final classifier on full training data for evaluation
        final_model = LogisticRegression(max_iter=1000)
        final_model.fit(scaler.transform(embeddings_array), is_false)  # Use full training data
        
        # Track convergence status
        convergence_info = {
            "n_iter": final_model.n_iter_[0],
            "converged": final_model.n_iter_[0] < final_model.max_iter
        }
        logging.debug(f"Classifier convergence: {convergence_info}")

        # Evaluate on all splits
        evaluation_splits = [
            ("train_train", X_train_scaled, y_train),
            ("train_test", X_test_scaled, y_test),
            ("eval", X_eval_scaled, eval_is_false)
        ]

        for suffix, X, y_true in evaluation_splits:  # pylint: disable=invalid-name
            y_pred = final_model.predict(X) if suffix == "eval" else base_model.predict(X)
            y_proba = final_model.predict_proba(X) if suffix == "eval" else base_model.predict_proba(X)
            y_preds_proba[suffix] = y_proba[:, 1]  # Store positive class probabilities
            
            # Calculate metrics
            acc = accuracy_score(y_true, y_pred)
            auroc = roc_auc_score(y_true, y_proba[:, 1])
            metrics[f"acc_p_ik_{suffix}"] = round(acc, 4)
            metrics[f"auroc_p_ik_{suffix}"] = round(auroc, 4)

        logging.info(f"p_ik classifier metrics: {metrics}")
        return y_preds_proba["eval"]

    # Handle case without evaluation data
    logging.warning("No evaluation data provided - returning empty probabilities")
    return np.array([])
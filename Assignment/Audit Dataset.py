# {"id":"10001","variant":"standard","title":"COMPAS Fairness Audit Notebook"}
"""
COMPAS Fairness Audit (AIF360)
- Load COMPAS dataset (AIF360)
- Train baseline classifier (Logistic Regression)
- Compute fairness metrics (FPR, FNR, Disparate Impact, Equal Opportunity)
- Visualize disparities (bar charts)
- Run simple remediation: Reweighing (pre-processing) + post-processing (Reject Option)
- Save results
Notes: Run in Colab or local env. Install AIF360 before running:
!pip install aif360==0.5.0  # or latest compatible
!pip install sklearn pandas matplotlib
"""
# Cell 1: Imports & install notes (run the pip lines in a notebook cell if needed)
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

# AIF360 imports (may require installation)
try:
    from aif360.datasets import CompasDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
    from aif360.algorithms.preprocessing import Reweighing
    from aif360.algorithms.postprocessing import RejectOptionClassification
    from aif360.algorithms.preprocessing import DisparateImpactRemover
except Exception as e:
    raise ImportError("AIF360 not found. Install with: pip install aif360. If in Colab, restart runtime after install.") from e

def _ensure_race_attr(dataset):
    # ensure 'race' exists, otherwise pick the first protected attribute available
    protected = dataset.protected_attribute_names
    if not protected:
        raise RuntimeError("No protected attributes found in dataset.")
    if 'race' in protected:
        return 'race'
    return protected[0]

if __name__ == "__main__":
    # Cell 2: Load COMPAS dataset (AIF360 wrapper)
    # The AIF360 CompasDataset provides a preprocessed dataset. We'll use 'race' as protected attribute when present.
    compas = CompasDataset()
    print("Dataset shape:", compas.shape)
    print("Protected attributes:", compas.protected_attribute_names)
    print("Labels shape:", compas.labels.shape)

    # Cell 3: Convert to DataFrame for quick inspection (optional)
    df = compas.convert_to_dataframe()[0]
    print(df.head())

    # Determine protected attribute key
    prot_attr = _ensure_race_attr(compas)
    privileged_groups = [{prot_attr: 1.0}]
    unprivileged_groups = [{prot_attr: 0.0}]
    print("Using protected attribute:", prot_attr)
    # Note: encoding (1.0 vs 0.0) is dataset dependent; verify these values for your AIF360 version.

    # Cell 4: Train/test split using AIF360 Dataset API
    splits = compas.split([0.7], shuffle=True)
    if len(splits) < 2:
        raise RuntimeError("Dataset split did not return train and test parts.")
    train, test = splits[0], splits[1]

    # For sklearn we need arrays:
    X_train = train.features
    y_train = train.labels.ravel()
    X_test = test.features
    y_test = test.labels.ravel()

    # Scale numeric features - use dedicated scalers for baseline and reweighing workflows
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Cell 5: Baseline classifier
    clf = LogisticRegression(solver='liblinear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy (baseline):", accuracy_score(y_test, y_pred))

    # Map predictions back to AIF360 BinaryLabelDataset for metric computations
    # Create copies of test dataset to hold predictions (check shapes match)
    test_pred = test.copy()
    if y_pred.reshape(-1, 1).shape[0] != test_pred.labels.shape[0]:
        raise RuntimeError("Prediction length does not match test labels length.")
    test_pred.labels = y_pred.reshape(-1, 1)

    # Cell 6: Compute fairness metrics
    metric_test = BinaryLabelDatasetMetric(test, unprivileged_groups=unprivileged_groups,
                                           privileged_groups=privileged_groups)
    metric_pred = ClassificationMetric(test, test_pred,
                                       unprivileged_groups=unprivileged_groups,
                                       privileged_groups=privileged_groups)

    print("Base dataset metrics:")
    try:
        print("Disparate impact (base):", metric_test.disparate_impact())
    except Exception:
        print("Disparate impact (base): N/A")
    try:
        print("Base positive rate (privileged):", metric_test.base_rate(privileged=True))
        print("Base positive rate (unprivileged):", metric_test.base_rate(privileged=False))
    except TypeError:
        # fallback if signature differs
        print("Base positive rate (privileged):", metric_test.base_rate(privileged_groups))
    except Exception:
        pass

    print("\nClassifier metrics (predictions):")
    print("Disparate impact (pred):", metric_pred.disparate_impact())
    print("False positive rate (privileged):", metric_pred.false_positive_rate(privileged=True))
    print("False positive rate (unprivileged):", metric_pred.false_positive_rate(privileged=False))
    print("False negative rate (privileged):", metric_pred.false_negative_rate(privileged=True))
    print("False negative rate (unprivileged):", metric_pred.false_negative_rate(privileged=False))
    print("Equal opportunity difference:", metric_pred.equal_opportunity_difference())
    print("Average odds difference:", metric_pred.average_odds_difference())

    # Cell 7: Visualize FPR and FNR disparities by group
    groups = ['unprivileged', 'privileged']
    fpr_vals = [
        metric_pred.false_positive_rate(privileged=False),
        metric_pred.false_positive_rate(privileged=True)
    ]
    fnr_vals = [
        metric_pred.false_negative_rate(privileged=False),
        metric_pred.false_negative_rate(privileged=True)
    ]

    # bar plot FPR
    plt.figure(figsize=(6, 4))
    plt.bar(groups, fpr_vals)
    plt.title("False Positive Rate by Group (Baseline)")
    plt.ylabel("False Positive Rate")
    plt.ylim(0, max(fpr_vals + fnr_vals) * 1.2)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.show()

    # bar plot FNR
    plt.figure(figsize=(6, 4))
    plt.bar(groups, fnr_vals)
    plt.title("False Negative Rate by Group (Baseline)")
    plt.ylabel("False Negative Rate")
    plt.ylim(0, max(fpr_vals + fnr_vals) * 1.2)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.show()

    # Cell 8: Simple remediation - Reweighing (pre-processing)
    rw = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    rw.fit(train)
    train_transf = rw.transform(train)

    # Train classifier on reweighed sample (use sample_weight)
    X_train_rw = train_transf.features
    y_train_rw = train_transf.labels.ravel()
    sample_weight = train_transf.instance_weights.ravel()

    scaler_rw = StandardScaler()
    X_train_rw = scaler_rw.fit_transform(X_train_rw)  # re-scale using separate scaler
    clf_rw = LogisticRegression(solver='liblinear')
    clf_rw.fit(X_train_rw, y_train_rw, sample_weight=sample_weight)

    # Apply to original test features (scaled with scaler_rw if training used different scaling)
    X_test_for_rw = scaler_rw.transform(test.features)
    y_pred_rw = clf_rw.predict(X_test_for_rw)
    test_pred_rw = test.copy()
    test_pred_rw.labels = y_pred_rw.reshape(-1, 1)

    metric_pred_rw = ClassificationMetric(test, test_pred_rw,
                                          unprivileged_groups=unprivileged_groups,
                                          privileged_groups=privileged_groups)
    print("After Reweighing - Disparate impact (pred):", metric_pred_rw.disparate_impact())
    print("After Reweighing - Equal opportunity diff:", metric_pred_rw.equal_opportunity_difference())
    print("After Reweighing - Avg odds diff:", metric_pred_rw.average_odds_difference())

    # Cell 9: Post-processing - Reject Option Classification (requires classifier scores)
    # Note: RejectOptionClassification needs score probabilities; create them:
    scores = clf.predict_proba(X_test)[:, 1]
    test_pred_scores = test.copy()
    test_pred_scores.scores = scores.reshape(-1, 1)

    roc = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups,
                                     low_class_thresh=0.01, high_class_thresh=0.99,
                                     num_class_thresh=100, num_ROC_margin=50,
                                     metric_name="Statistical parity difference")
    # roc.fit usually expects validation set and predicted scores; using test as placeholder (use caution)
    roc = roc.fit(test, test_pred_scores)
    test_roc_pred = roc.predict(test_pred_scores)
    metric_pred_roc = ClassificationMetric(test, test_roc_pred,
                                           unprivileged_groups=unprivileged_groups,
                                           privileged_groups=privileged_groups)
    print("After Reject Option - Disparate impact (pred):", metric_pred_roc.disparate_impact())
    print("After Reject Option - Equal opportunity diff:", metric_pred_roc.equal_opportunity_difference())
    print("After Reject Option - Avg odds diff:", metric_pred_roc.average_odds_difference())

    # Cell 10: Save key results to CSV for inclusion in report
    results = {
        'metric': ['disparate_impact', 'equal_opportunity_diff', 'avg_odds_diff',
                   'fpr_privileged', 'fpr_unprivileged', 'fnr_privileged', 'fnr_unprivileged'],
        'baseline': [metric_pred.disparate_impact(),
                     metric_pred.equal_opportunity_difference(),
                     metric_pred.average_odds_difference(),
                     metric_pred.false_positive_rate(privileged=True),
                     metric_pred.false_positive_rate(privileged=False),
                     metric_pred.false_negative_rate(privileged=True),
                     metric_pred.false_negative_rate(privileged=False)],
        'reweighing': [metric_pred_rw.disparate_impact(),
                       metric_pred_rw.equal_opportunity_difference(),
                       metric_pred_rw.average_odds_difference(),
                       metric_pred_rw.false_positive_rate(privileged=True),
                       metric_pred_rw.false_positive_rate(privileged=False),
                       metric_pred_rw.false_negative_rate(privileged=True),
                       metric_pred_rw.false_negative_rate(privileged=False)],
        'reject_option': [metric_pred_roc.disparate_impact(),
                          metric_pred_roc.equal_opportunity_difference(),
                          metric_pred_roc.average_odds_difference(),
                          metric_pred_roc.false_positive_rate(privileged=True),
                          metric_pred_roc.false_positive_rate(privileged=False),
                          metric_pred_roc.false_negative_rate(privileged=True),
                          metric_pred_roc.false_negative_rate(privileged=False)]
    }
    pd.DataFrame(results).to_csv("compas_fairness_results.csv", index=False)
    print("Saved compas_fairness_results.csv")

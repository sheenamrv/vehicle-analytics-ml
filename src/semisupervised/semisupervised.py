import pandas as pd
from sklearn.base import clone
from sklearn.semi_supervised import SelfTrainingClassifier

# Get the numeric feature columns
def get_num_feature_columns(df, label):
    features = df.select_dtypes(include="number").columns.tolist()

    if label in features:
        features.remove(label)

    return features

# Convert unlabeled rows with a -1 (unlabeled) label for SelfTrainingClassifier
def convert_ssl_labels(df, label):
    return df[label].copy().fillna(-1)

# Main training
def train_ssl(df, features, label, pretrained_model, threshold=0.90, max_iter=10):
    X = df[features].fillna(df[features].median())
    y = convert_ssl_labels(df, label)
    
    # Track original number of unlabeled labels
    original_unlabeled_mask = (y == -1)
    original_unlabeled_count = original_unlabeled_mask.sum()

    # Takes a new copy of the pretrained model, keeps track of progress through verbose parameter
    ssl_model = SelfTrainingClassifier(estimator=clone(pretrained_model), threshold=threshold, max_iter=max_iter, verbose=True)

    ssl_model.fit(X, y)

    # Count only rows that were originally unlabeled and later received a label
    new_pseudo_labeled_mask = original_unlabeled_mask & (ssl_model.transduction_ != -1)

    pseudo_labeled_count = new_pseudo_labeled_mask.sum()
    unlabeled_remaining_count = (ssl_model.transduction_ == -1).sum()

    if original_unlabeled_count > 0:
        pseudo_labeled_percentage = (pseudo_labeled_count/ original_unlabeled_count)*100
        unlabeled_remaining_percentage = (unlabeled_remaining_count/ original_unlabeled_count)*100
    else:
        pseudo_labeled_percentage = 0
        unlabeled_remaining_percentage = 0
        
    progress = pd.DataFrame({"status":["Originally unlabeled", "Pseudo-labeled", "Remaining unlabeled"], "count":[original_unlabeled_count, pseudo_labeled_count, unlabeled_remaining_count], 
                             "percentage":[100.0, pseudo_labeled_percentage, unlabeled_remaining_percentage]})
    
    result = {"model": ssl_model, "transduction": ssl_model.transduction_, "labeled_iter": ssl_model.labeled_iter_, "pseudo_labeled_count": pseudo_labeled_count, "unlabeled_remaining": unlabeled_remaining_count, "progress_df": progress}

    return result

def get_ssl_training_progress(train_ssl_result):
    labeled_iter = train_ssl_result["labeled_iter"]

    rows = []
    total_count = len(labeled_iter)
    
    sorted_labels = sorted(set(labeled_iter))

    for iteration in sorted_labels:
        count = (labeled_iter == iteration).sum()

        if total_count > 0:
            percentage = (count/ total_count)*100
        else:
            percentage = 0

        if iteration == -1:
            label = "Never labeled"
        elif iteration == 0:
            label = "Originally labeled"
        else:
            label = f"Pseudo-labeled on iteration {iteration}"

        rows.append({"iteration": iteration, "description": label, "count": count, "percentage": percentage})

    return pd.DataFrame(rows) # Returning a df for easier plotting

# TO DO: ssl workflow function

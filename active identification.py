from __future__ import print_function
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import matplotlib
matplotlib.use('Agg')
from utilis import *


def compute_mean_radius_sq(features: torch.Tensor,
                           labels: torch.Tensor,
                           num_classes: int) -> np.ndarray:
    """
    Compute class-wise R_c^2 = mean squared distance to class center.
    features: Tensor[N, D]
    labels:   LongTensor[N]
    """
    radii2 = []
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() == 0:
            continue
        fc = features[mask]                  # [Nc, D]
        mu = fc.mean(dim=0, keepdim=True)    # [1, D]
        # squared distances, mean over class
        radii2.append(((fc - mu)**2).sum(dim=1).mean().item())
    return np.array(radii2)

def extract_features_single_batch(loader, classifier, device, transform_amplify):
    """
    Extracts features from all layers for a single batch of data.
    """
    classifier.eval()

    # Get a single batch of data from the loader
    data, _ = next(iter(loader))
    data = data.to(device)

    with torch.no_grad():
        # The classifier returns the entire list of features from all blocks
        feat_all, prediction = classifier(transform_amplify(data), release=True)

        # Reshape and move to CPU, but do not concatenate
        all_features_by_block = [f.view(f.size(0), -1).cpu() for f in feat_all]
        preds = prediction.argmax(dim=1).cpu()

    return all_features_by_block, preds

def run_mean_radius(args, classifier, train_loader, transform_amplify, device):
    print("Running mean squared radius analysis for specified layers...")

    save_dir = f"t-SNE/{args.dataset}/{args.target_model}_all_layers/"
    os.makedirs(save_dir, exist_ok=True)

    # Define the specific layers to analyze
    # target_indices = [2, 7, 24, 39, 47, 48, 49]
    target_indices = [13, 17, 26, 30, 39, 43, 52]

    # Step 1: Extract features from ALL layers for a single batch
    all_features, pred_labels = extract_features_single_batch(train_loader, classifier, device, transform_amplify)

    # Step 2: Select and process only the features for the target indices
    selected_avg_radii2_oh_list = []
    for idx in target_indices:
        if idx < len(all_features):
            features = all_features[idx]
            radius_sq = compute_mean_radius_sq(features, pred_labels, args.nz)
            selected_avg_radii2_oh_list.append(radius_sq.mean().item())
        else:
            print(f"Warning: Layer index {idx} is out of bounds. Skipping.")

    print(f"Average radii^2 for specified layers: {selected_avg_radii2_oh_list}")

    # Step 3: Calculate the percentage decrease between consecutive layers (using raw values)
    percentage_decrease = []
    for i in range(1, len(selected_avg_radii2_oh_list)):
        current_val = selected_avg_radii2_oh_list[i]
        previous_val = selected_avg_radii2_oh_list[i - 1]
        if previous_val != 0:
            percentage = (previous_val - current_val) / previous_val
            percentage_decrease.append(percentage)
        else:
            percentage_decrease.append(0.0)

    # Print the percentage decrease along with the corresponding layer indices
    print("Percentage decrease between specified layers:")
    for i, p in enumerate(percentage_decrease):
        from_layer = target_indices[i]
        to_layer = target_indices[i + 1]
        print(f"  {from_layer}->{to_layer}: {p:.4f}")

    # Step 4: Find the largest percentage decrease
    if not percentage_decrease:
        print("Not enough specified layers to calculate percentage decrease.")
        return None

    sorted_indices_relative = sorted(range(len(percentage_decrease)), key=lambda k: percentage_decrease[k],
                                     reverse=True)

    first_idx_relative = sorted_indices_relative[0]
    first_idx = target_indices[first_idx_relative + 1]
    print(f"Largest percentage decrease found between layers {target_indices[first_idx_relative]} and {first_idx}")

    # Step 5: Check for a second significant decrease and return two indices
    if len(sorted_indices_relative) > 1:
        second_idx_relative = sorted_indices_relative[1]
        second_idx = target_indices[second_idx_relative + 1]
        print(
            f"Second largest percentage decrease found between layers {target_indices[second_idx_relative]} and {second_idx}")

        # Step 6: Check the condition for all intermediate layers
        condition_met = True
        start_idx = min(first_idx_relative, second_idx_relative) + 1
        end_idx = max(first_idx_relative, second_idx_relative)

        for i in range(start_idx, end_idx):
            if abs(percentage_decrease[i]) >= 0.2:
                condition_met = False
                break

        if condition_met:
            result = [first_idx]
            print(f"Condition met (all intermediate gaps are small), returning a single index: {result}")
            return result
        else:
            result = [first_idx, second_idx]
            print(f"Condition not met, returning two indices: {result}")
            return result
    else:
        result = [first_idx]
        print(f"Only one significant gap found, returning a single index: {result}")
        return result
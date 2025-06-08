import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List, Any, Optional, Union
import pandas as pd
from collections import Counter
import random
import tensorflow as tf
import keras
from tensorflow.keras.callbacks import Callback

import os
import logging

def silence_tensorflow():
    """Silence every unnecessary warning from tensorflow."""
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    os.environ["KMP_AFFINITY"] = "noverbose"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # print("SUPRESSING TF OUTPUT")

    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(3)
    except ModuleNotFoundError:
        pass


def compute_seed(base_seed, params):
    param_hash = hash(frozenset(params.items())) & 0xFFFFFFFF
    return (base_seed + param_hash) & 0xFFFFFFFF


# Add this method before _make_LSTM
class LabeledDataMetricsCallback(Callback):
    """Custom callback to track metrics only on labeled data"""
    
    def __init__(self, validation_data, unlabeled_value):
        super().__init__()
        self.val_data = validation_data  # (x_val, y_val)
        self.unlabeled_value = unlabeled_value
        self.history = {'labeled_accuracy': [], 'labeled_loss': []}
        
    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.val_data
        # Get predictions on validation data
        y_pred = self.model.predict(x_val, verbose=0)
        
        # For OPO mode (sequence of predictions)
        if len(y_val.shape) > 1:
            # Find valid indices (labeled data points)
            valid_mask = y_val != self.unlabeled_value
            
            # Get predicted classes
            pred_classes = np.argmax(y_pred, axis=2)
            
            # Calculate accuracy only on labeled data points
            if np.any(valid_mask):
                labeled_acc = np.mean(pred_classes[valid_mask] == y_val[valid_mask])
                self.history['labeled_accuracy'].append(labeled_acc)
            else:
                self.history['labeled_accuracy'].append(0.0)
                
        # For OPS mode (single prediction per sequence)
        else:
            # Find valid indices (labeled data points)
            valid_mask = y_val != self.unlabeled_value
            
            # Get predicted classes
            pred_classes = np.argmax(y_pred, axis=1)
            
            # Calculate accuracy only on labeled data points
            if np.any(valid_mask):
                labeled_acc = np.mean(pred_classes[valid_mask] == y_val[valid_mask])
                self.history['labeled_accuracy'].append(labeled_acc)
            else:
                self.history['labeled_accuracy'].append(0.0)
                
        # Add the current epoch metrics to our history
        if logs:
            for key, value in logs.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)


@keras.saving.register_keras_serializable()
class MaskedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='masked_accuracy', unlabeled_value=-1, **kwargs):
        super(MaskedAccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.unlabeled_value = unlabeled_value
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Create mask for labeled data
        # This handles both the case with sample weights and the case with unlabeled values
        if sample_weight is not None:
            # If sample weight is provided, use it as primary mask
            weight_mask = tf.cast(sample_weight > 0, tf.bool)
        else:
            # If no sample weight, use all ones
            weight_mask = tf.ones_like(y_true, dtype=tf.bool)
        
        # Also check for unlabeled values in y_true
        value_mask = tf.not_equal(y_true, self.unlabeled_value)
        
        # Combine both masks - an example is valid if it has weight > 0 AND is not unlabeled
        mask = tf.logical_and(weight_mask, value_mask)
        
        # Get predicted classes
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        y_pred_classes = tf.cast(y_pred_classes, y_true.dtype)
        
        # Calculate correct predictions only on labeled data
        correct = tf.cast(tf.equal(y_true, y_pred_classes), tf.float32) * tf.cast(mask, tf.float32)
        
        # Update running totals
        self.total.assign_add(tf.reduce_sum(tf.cast(mask, tf.float32)))
        self.count.assign_add(tf.reduce_sum(correct))
        
    def result(self):
        return self.count / (self.total + tf.keras.backend.epsilon())
        
    def reset_state(self):
        self.total.assign(0)
        self.count.assign(0)

@keras.saving.register_keras_serializable()
class MaskedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, unlabeled_value=-1, name='masked_sparse_categorical_crossentropy', **kwargs):
        super(MaskedSparseCategoricalCrossentropy, self).__init__(name=name, **kwargs)
        self.unlabeled_value = unlabeled_value
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)
    
    def call(self, y_true, y_pred, sample_weight=None):
        # Create mask for labeled data
        mask = tf.not_equal(y_true, self.unlabeled_value)
        mask_float = tf.cast(mask, dtype=tf.float32)
        
        # If sample weight is provided, combine it with our mask
        if sample_weight is not None:
            mask_float = mask_float * sample_weight
            
        # Calculate loss only on labeled data
        per_example_loss = self.loss_fn(y_true, y_pred)
        masked_loss = per_example_loss * mask_float
        
        # Return average loss over valid points
        sum_loss = tf.reduce_sum(masked_loss)
        sum_mask = tf.reduce_sum(mask_float)
        
        # Avoid division by zero
        return sum_loss / (sum_mask + tf.keras.backend.epsilon())



@keras.saving.register_keras_serializable()
class MaskedConv1D(tf.keras.layers.Conv1D):
    def compute_mask(self, inputs, mask=None):
        # preserve the time-step mask (True = valid) 
        return mask
    def call(self, inputs, mask=None):
        out = super().call(inputs)
        if mask is not None:
            # zero out conv outputs at masked positions
            out *= tf.cast(tf.expand_dims(mask, -1), out.dtype)
        return out



def random_validation_split(
    train_X: np.ndarray,
    train_Y: np.ndarray,
    sample_weights: np.ndarray,
    test_size: float = 0.33,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        train_X,
        train_Y,
        sample_weights,
        test_size=test_size,
        shuffle=True,
        random_state=random_state
    )

def get_sequence_ids(train_X: np.ndarray, train_Y: np.ndarray) -> np.ndarray:
    ops_mode = len(train_Y.shape) == 1

    if ops_mode:
        return np.arange(len(train_Y))
    else:
        return np.arange(len(train_Y))

def count_labels_per_sequence(train_Y: np.ndarray, sequence_ids: np.ndarray, unlabeled_value: int = -1) -> Dict[int, int]:
    labeled_counts = {}
    ops_mode = len(train_Y.shape) == 1

    for seq_id in np.unique(sequence_ids):
        if ops_mode:
            labeled_counts[seq_id] = 1 if train_Y[seq_id] != unlabeled_value else 0
        else:
            seq_labels = train_Y[seq_id]
            labeled_counts[seq_id] = np.sum(seq_labels != unlabeled_value)

    return labeled_counts

def count_class_distribution(train_Y: np.ndarray, sequence_ids: np.ndarray, unlabeled_value: int = -1) -> Dict[int, Dict[int, int]]:
    class_distributions = {}
    ops_mode = len(train_Y.shape) == 1

    for seq_id in np.unique(sequence_ids):
        if ops_mode:
            if train_Y[seq_id] != unlabeled_value:
                class_distributions[seq_id] = {train_Y[seq_id]: 1}
            else:
                class_distributions[seq_id] = {}
        else:
            seq_labels = train_Y[seq_id]
            valid_labels = seq_labels[seq_labels != unlabeled_value]
            if len(valid_labels) > 0:
                class_distributions[seq_id] = dict(Counter(valid_labels))
            else:
                class_distributions[seq_id] = {}

    return class_distributions


def sequence_aware_validation_split(
    train_X: np.ndarray, 
    train_Y: np.ndarray, 
    sample_weights: np.ndarray, 
    sequence_ids: np.ndarray,
    unlabeled_value: int = -1,
    test_size: float = 0.33
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits data for validation while preserving sequence integrity.
    Chooses entire sequences for validation based on label counts.
    """
    # Count labels per sequence
    labeled_counts = count_labels_per_sequence(train_Y, sequence_ids, unlabeled_value)
    
    # Sort sequences by label count (descending order)
    sorted_seqs = sorted(labeled_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate total number of labeled data points
    total_labels = sum(labeled_counts.values())
    target_val_labels = total_labels * test_size
    
    # Choose sequences for validation to get close to target percentage
    val_seqs = []
    val_label_count = 0
    
    for seq_id, count in sorted_seqs:
        if val_label_count < target_val_labels:
            val_seqs.append(seq_id)
            val_label_count += count
        
        # Stop once we've reached our target
        if val_label_count >= target_val_labels:
            break
    
    # Remaining sequences go to training
    train_seqs = [seq_id for seq_id, _ in sorted_seqs if seq_id not in val_seqs]
    
    # Split the data using the sequences
    x_train = train_X[train_seqs]
    y_train = train_Y[train_seqs]
    sw_train = sample_weights[train_seqs]
    
    x_valid = train_X[val_seqs]
    y_valid = train_Y[val_seqs]
    sw_valid = sample_weights[val_seqs]
    
    # Calculate actual split percentage for logging
    train_label_count = np.sum(y_train != unlabeled_value)
    val_label_count = np.sum(y_valid != unlabeled_value)
    actual_split = val_label_count / (train_label_count + val_label_count) if (train_label_count + val_label_count) > 0 else 0
    
    # print(f"Sequence-aware split - Training: {train_label_count} labels, Validation: {val_label_count} labels")
    # print(f"Actual validation split: {actual_split:.2f} (target: {test_size:.2f})")
    
    return x_train, x_valid, y_train, y_valid, sw_train, sw_valid

def interleaved_validation_split(
    train_X: np.ndarray, 
    train_Y: np.ndarray, 
    sample_weights: np.ndarray, 
    sequence_ids: np.ndarray,
    unlabeled_value: int = -1,
    interleave_factor: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Interleaved sequence splitting - takes every nth sequence for validation.
    """
    # Count labels per sequence
    labeled_counts = count_labels_per_sequence(train_Y, sequence_ids, unlabeled_value)
    
    # Sort sequences by label count (descending)
    sorted_seqs = sorted(labeled_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Take every nth sequence for validation (interleaved approach)
    train_seqs = [seq_id for i, (seq_id, _) in enumerate(sorted_seqs) if i % interleave_factor != 0]
    val_seqs = [seq_id for i, (seq_id, _) in enumerate(sorted_seqs) if i % interleave_factor == 0]
    
    # Split the data using the sequences
    x_train = train_X[train_seqs]
    y_train = train_Y[train_seqs]
    sw_train = sample_weights[train_seqs]
    
    x_valid = train_X[val_seqs]
    y_valid = train_Y[val_seqs]
    sw_valid = sample_weights[val_seqs]
    
    # Calculate actual split percentage for logging
    train_label_count = np.sum(y_train != unlabeled_value)
    val_label_count = np.sum(y_valid != unlabeled_value)
    actual_split = val_label_count / (train_label_count + val_label_count) if (train_label_count + val_label_count) > 0 else 0
    
    # print(f"Interleaved split - Training: {train_label_count} labels, Validation: {val_label_count} labels")
    # print(f"Actual validation split: {actual_split:.2f} (target: {1/interleave_factor:.2f})")
    
    return x_train, x_valid, y_train, y_valid, sw_train, sw_valid

# def stratified_sequence_split(
#     train_X: np.ndarray, 
#     train_Y: np.ndarray, 
#     sample_weights: np.ndarray, 
#     sequence_ids: np.ndarray,
#     unlabeled_value: int = -1,
#     test_size: float = 0.33
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Performs stratified validation split at the sequence level,
#     ensuring similar class distributions in train and validation sets.
#     """
#     from collections import Counter
    
#     # Get overall label distribution (excluding unlabeled)
#     ops_mode = len(train_Y.shape) == 1
    
#     if ops_mode:
#         all_labels = train_Y[train_Y != unlabeled_value]
#     else:
#         all_labels = train_Y[train_Y != unlabeled_value].flatten()
    
#     overall_distribution = Counter(all_labels)
#     total_labels = len(all_labels)
    
#     # Calculate target distribution for validation set
#     target_val_dist = {cls: count * test_size for cls, count in overall_distribution.items()}
    
#     # Get class distribution per sequence
#     sequence_class_dist = count_class_distribution(train_Y, sequence_ids, unlabeled_value)
    
#     # Greedy algorithm to create validation split with distribution similar to overall
#     current_val_dist = Counter()
#     val_seqs = []
    
#     # Sort sequences by total label count (descending)
#     seq_counts = [(seq_id, sum(dist.values())) for seq_id, dist in sequence_class_dist.items() if dist]
#     seq_counts.sort(key=lambda x: x[1], reverse=True)
    
#     for seq_id, _ in seq_counts:
#         # Check if adding this sequence would improve distribution
#         candidate_dist = current_val_dist + Counter(sequence_class_dist[seq_id])
        
#         # Check if we're still under target for any class
#         under_target = False
#         for cls, target in target_val_dist.items():
#             if candidate_dist[cls] <= target:
#                 under_target = True
#                 break
        
#         # Add to validation if it keeps us under target for any class
#         if under_target:
#             current_val_dist = candidate_dist
#             val_seqs.append(seq_id)
            
#             # Check if we've reached or exceeded our target for all classes
#             all_reached = True
#             for cls, target in target_val_dist.items():
#                 if current_val_dist[cls] < target * 0.9:  # Allow 10% under-representation
#                     all_reached = False
#                     break
            
#             if all_reached:
#                 break
    
#     # Remaining sequences go to training
#     train_seqs = [seq_id for seq_id in np.unique(sequence_ids) if seq_id not in val_seqs]
    
#     # Split the data using the sequences
#     x_train = train_X[train_seqs]
#     y_train = train_Y[train_seqs]
#     sw_train = sample_weights[train_seqs]
    
#     x_valid = train_X[val_seqs]
#     y_valid = train_Y[val_seqs]
#     sw_valid = sample_weights[val_seqs]
    
#     # Calculate actual distribution for logging
#     ops_mode = len(y_train.shape) == 1
    
#     if ops_mode:
#         train_dist = Counter(y_train[y_train != unlabeled_value])
#         val_dist = Counter(y_valid[y_valid != unlabeled_value])
#     else:
#         train_dist = Counter(y_train[y_train != unlabeled_value].flatten())
#         val_dist = Counter(y_valid[y_valid != unlabeled_value].flatten())
    
#     train_total = sum(train_dist.values())
#     val_total = sum(val_dist.values())
#     actual_split = val_total / (train_total + val_total) if (train_total + val_total) > 0 else 0
    
#     print(f"Stratified split - Training: {train_total} labels, Validation: {val_total} labels")
#     print(f"Actual validation split: {actual_split:.2f} (target: {test_size:.2f})")
#     print(f"Training distribution: {dict(train_dist)}")
#     print(f"Validation distribution: {dict(val_dist)}")
    
#     return x_train, x_valid, y_train, y_valid, sw_train, sw_valid


def stratified_sequence_split(
    train_X: np.ndarray, 
    train_Y: np.ndarray, 
    sample_weights: np.ndarray, 
    sequence_ids: np.ndarray,
    unlabeled_value: int = -1,
    test_size: float = 0.33
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs stratified validation split at the sequence level,
    ensuring similar class distributions in train and validation sets.
    
    This implementation focuses on:
    1. Maintaining the target validation percentage
    2. Keeping a similar distribution of classes in both splits
    3. Handling sequences with sparse labeling appropriately
    """

    
    # Get overall label distribution (excluding unlabeled)
    ops_mode = len(train_Y.shape) == 1
    
    if ops_mode:
        all_labels = train_Y[train_Y != unlabeled_value]
    else:
        all_labels = train_Y[train_Y != unlabeled_value].flatten()
    
    overall_distribution = Counter(all_labels)
    total_labels = len(all_labels)
    
    # # Print overall class distribution
    # print("Overall class distribution:")
    # for cls, count in sorted(overall_distribution.items()):
    #     print(f"Class {cls}: {count} samples ({count/total_labels*100:.1f}%)")
    
    # Get class distribution per sequence
    sequence_class_dist = count_class_distribution(train_Y, sequence_ids, unlabeled_value)
    
    # Calculate total labels in each sequence
    sequence_label_counts = {seq_id: sum(dist.values()) for seq_id, dist in sequence_class_dist.items()}
    
    # Calculate target number of labels for validation
    target_val_labels = int(total_labels * test_size)
    
    # Calculate relative class proportions for stratification
    class_proportions = {cls: count/total_labels for cls, count in overall_distribution.items()}
    
    # Calculate target number of labels per class in validation
    target_class_counts = {cls: int(count * test_size) for cls, count in overall_distribution.items()}
    
    # print(f"Target validation size: {target_val_labels} labels ({test_size*100:.1f}%)")
    # print("Target class counts in validation:")
    # for cls, count in sorted(target_class_counts.items()):
    #     print(f"Class {cls}: {count} samples ({target_class_counts[cls]/overall_distribution[cls]*100:.1f}%)")
    
    # Approach: Two-phase selection
    # 1. First, try to select sequences that have good class representation
    # 2. Then, adjust to meet target validation size
    
    # Score each sequence based on how well it represents the overall distribution
    sequence_scores = {}
    for seq_id, dist in sequence_class_dist.items():
        if not dist:  # Skip sequences with no labels
            continue
            
        # Calculate the class distribution within this sequence
        total_seq_labels = sum(dist.values())
        seq_proportions = {cls: count/total_seq_labels for cls, count in dist.items()}
        
        # Calculate how well this sequence matches the overall distribution
        # Lower score = better match (0 = perfect match)
        score = 0
        for cls in overall_distribution:
            seq_prop = seq_proportions.get(cls, 0)
            overall_prop = class_proportions[cls]
            score += abs(seq_prop - overall_prop)
        
        # Also consider the number of labels in this sequence
        # We prefer sequences with more labels
        score = score / (total_seq_labels ** 0.5)  # Scale by square root of label count
        
        sequence_scores[seq_id] = score
    
    # Sort sequences by score (best matching sequences first)
    sorted_sequences = sorted(sequence_scores.items(), key=lambda x: x[1])
    
    # Select sequences for validation until we reach our target
    val_seqs = []
    current_val_counts = Counter()
    current_val_total = 0
    
    # First phase: select sequences until we reach our target for each class
    # or until we've gone through all sequences
    for seq_id, _ in sorted_sequences:
        # Skip if we've already reached our overall target size
        if current_val_total >= target_val_labels:
            break
            
        # Check if adding this sequence would help meet our class targets
        seq_dist = sequence_class_dist[seq_id]
        would_help = False
        
        for cls, count in seq_dist.items():
            if current_val_counts[cls] < target_class_counts.get(cls, 0):
                would_help = True
                break
        
        # If this sequence would help, add it to validation
        if would_help:
            val_seqs.append(seq_id)
            for cls, count in seq_dist.items():
                current_val_counts[cls] += count
                current_val_total += count
    
    # Second phase: adjust to meet overall target size if needed
    # If we have too many validation sequences, remove some
    if current_val_total > target_val_labels * 1.2:  # Allow up to 20% over target
        # Sort validation sequences by score (worst matching first)
        val_seqs_sorted = sorted(val_seqs, key=lambda x: -sequence_scores[x])
        
        # Remove sequences until we're close to target
        for seq_id in val_seqs_sorted:
            seq_dist = sequence_class_dist[seq_id]
            seq_total = sum(seq_dist.values())
            
            # Don't remove if it would make us go below target
            if current_val_total - seq_total < target_val_labels * 0.9:  # Allow down to 90% of target
                break
                
            # Remove this sequence
            val_seqs.remove(seq_id)
            for cls, count in seq_dist.items():
                current_val_counts[cls] -= count
                current_val_total -= count
    
    # If we have too few validation sequences, add more
    elif current_val_total < target_val_labels * 0.8:  # If below 80% of target
        # Get sequences not already in validation
        remaining_seqs = [seq_id for seq_id, _ in sorted_sequences if seq_id not in val_seqs]
        
        # Add sequences until we're close to target
        for seq_id in remaining_seqs:
            seq_dist = sequence_class_dist[seq_id]
            seq_total = sum(seq_dist.values())
            
            # Add this sequence
            val_seqs.append(seq_id)
            for cls, count in seq_dist.items():
                current_val_counts[cls] += count
                current_val_total += count
            
            # Stop if we've reached our target
            if current_val_total >= target_val_labels * 0.9:  # At least 90% of target
                break
    
    # Remaining sequences go to training
    train_seqs = [seq_id for seq_id in np.unique(sequence_ids) if seq_id not in val_seqs]
    
    # Split the data using the sequences
    x_train = train_X[train_seqs]
    y_train = train_Y[train_seqs]
    sw_train = sample_weights[train_seqs]
    
    x_valid = train_X[val_seqs]
    y_valid = train_Y[val_seqs]
    sw_valid = sample_weights[val_seqs]
    
    # Calculate actual distribution for logging
    if ops_mode:
        train_dist = Counter(y_train[y_train != unlabeled_value])
        val_dist = Counter(y_valid[y_valid != unlabeled_value])
    else:
        train_dist = Counter(y_train[y_train != unlabeled_value].flatten())
        val_dist = Counter(y_valid[y_valid != unlabeled_value].flatten())
    
    train_total = sum(train_dist.values())
    val_total = sum(val_dist.values())
    actual_split = val_total / (train_total + val_total) if (train_total + val_total) > 0 else 0
    
    # print(f"Stratified split - Training: {train_total} labels, Validation: {val_total} labels")
    # print(f"Actual validation split: {actual_split:.2f} (target: {test_size:.2f})")
    
    # Compare target vs actual class distribution
    # print("\nClass distribution comparison:")
    # print(f"{'Class':<10} {'Train Count':<12} {'Train %':<10} {'Val Count':<12} {'Val %':<10} {'Target %':<10}")
    # print("-" * 70)
    
    # for cls in sorted(overall_distribution.keys()):
    #     train_count = train_dist.get(cls, 0)
    #     val_count = val_dist.get(cls, 0)
        
    #     train_pct = train_count / train_total * 100 if train_total > 0 else 0
    #     val_pct = val_count / val_total * 100 if val_total > 0 else 0
    #     target_pct = class_proportions[cls] * 100
        
    #     print(f"{cls:<10} {train_count:<12} {train_pct:>8.1f}% {val_count:<12} {val_pct:>8.1f}% {target_pct:>8.1f}%")
    
    return x_train, x_valid, y_train, y_valid, sw_train, sw_valid





def balanced_class_validation_split(
    train_X: np.ndarray, 
    train_Y: np.ndarray, 
    sample_weights: np.ndarray, 
    sequence_ids: np.ndarray,
    unlabeled_value: int = -1,
    test_size: float = 0.33,
    class_counts: Optional[Dict[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits data while balancing different behavior classes between train/validation.
    
    Parameters:
    -----------
    train_X : numpy.ndarray
        Feature data for training
    train_Y : numpy.ndarray
        Label data for training
    sample_weights : numpy.ndarray
        Sample weights that account for class imbalance and missing labels
    sequence_ids : numpy.ndarray
        Array identifying which sequence each data point belongs to
    unlabeled_value : int, default=-1
        Value used to indicate unlabeled data
    test_size : float, default=0.33
        Target proportion of labeled data to include in validation
    class_counts : dict, optional
        Dictionary mapping class IDs to their counts in the dataset
        
    Returns:
    --------
    x_train : numpy.ndarray
        Features for training
    x_valid : numpy.ndarray
        Features for validation
    y_train : numpy.ndarray
        Labels for training
    y_valid : numpy.ndarray
        Labels for validation
    sw_train : numpy.ndarray
        Sample weights for training
    sw_valid : numpy.ndarray
        Sample weights for validation
    """
    # Determine if we're in OPS mode (one prediction per sequence) or OPO mode (one prediction per observation)
    ops_mode = len(train_Y.shape) == 1
    
    # Get class distribution per sequence (excluding unlabeled)
    sequence_class_dist = {}
    
    for seq_id in np.unique(sequence_ids):
        if ops_mode:
            # For OPS, sequence_id is the index in train_Y
            if train_Y[seq_id] != unlabeled_value:
                # Only include labeled data
                label = train_Y[seq_id]
                sequence_class_dist[seq_id] = {label: 1}
            else:
                sequence_class_dist[seq_id] = {}
        else:
            # For OPO, count each labeled class in the sequence
            seq_labels = train_Y[seq_id]
            valid_labels = seq_labels[seq_labels != unlabeled_value]
            if len(valid_labels) > 0:
                counts = {}
                for label in valid_labels:
                    if label in counts:
                        counts[label] += 1
                    else:
                        counts[label] = 1
                sequence_class_dist[seq_id] = counts
            else:
                sequence_class_dist[seq_id] = {}
    
    # Determine actual class counts (excluding unlabeled)
    if class_counts is None:
        class_counts = {}
        
        # Get unique classes from all labeled data
        if ops_mode:
            labeled_data = train_Y[train_Y != unlabeled_value]
        else:
            labeled_data = train_Y[train_Y != unlabeled_value].flatten()
        
        unique_classes = np.unique(labeled_data)
        
        for cls in unique_classes:
            class_counts[cls] = np.sum(train_Y == cls)
    
    # Skip sequences with no labels
    sequences_with_labels = [seq_id for seq_id, counts in sequence_class_dist.items() if counts]
    
    # Group sequences by dominant label
    dominant_class_sequences = {cls: [] for cls in class_counts.keys()}
    mixed_sequences = []
    
    for seq_id in sequences_with_labels:
        counts = sequence_class_dist[seq_id]
        
        # Determine dominant label
        if counts:  # Skip empty counts
            max_label = max(counts.items(), key=lambda x: x[1])[0]
            max_count = counts[max_label]
            total = sum(counts.values())
            
            if max_count / total > 0.6:  # If one label is dominant (>60%)
                dominant_class_sequences[max_label].append(seq_id)
            else:
                mixed_sequences.append(seq_id)
    
    # Take validation sequences from each group
    val_seqs = []
    
    # First, add from each dominant class group
    for cls, seqs in dominant_class_sequences.items():
        if seqs:
            # Take approximately test_size proportion from each class group
            num_to_take = max(1, int(len(seqs) * test_size))
            val_seqs.extend(seqs[:num_to_take])
            
    # Then add from mixed group
    if mixed_sequences:
        num_to_take = max(1, int(len(mixed_sequences) * test_size))
        val_seqs.extend(mixed_sequences[:num_to_take])
    
    # Remaining sequences with labels go to training
    train_seqs = [seq_id for seq_id in sequences_with_labels if seq_id not in val_seqs]
    
    # Split the data using the sequences
    x_train = train_X[train_seqs]
    y_train = train_Y[train_seqs]
    sw_train = sample_weights[train_seqs]
    
    x_valid = train_X[val_seqs]
    y_valid = train_Y[val_seqs]
    sw_valid = sample_weights[val_seqs]
    
    # Calculate class distribution in train and validation sets (excluding unlabeled)
    train_class_counts = {}
    val_class_counts = {}
    
    for cls in class_counts.keys():
        train_class_counts[cls] = np.sum(y_train == cls)
        val_class_counts[cls] = np.sum(y_valid == cls)
    
    # Calculate total labeled data in each set
    train_labeled_count = sum(train_class_counts.values())
    val_labeled_count = sum(val_class_counts.values())
    total_labeled = train_labeled_count + val_labeled_count
    
    # Calculate actual split percentage (considering only labeled data)
    actual_split = val_labeled_count / total_labeled if total_labeled > 0 else 0
    
    # print(f"Class-balanced split - Training: {train_labeled_count} labeled samples, Validation: {val_labeled_count} labeled samples")
    # print(f"Actual validation split: {actual_split:.2f} (target: {test_size:.2f})")
    
    # Print class distribution as percentages
    # if train_labeled_count > 0:
    #     train_pct = {cls: count/train_labeled_count*100 for cls, count in train_class_counts.items()}
    # else:
    #     train_pct = {cls: 0 for cls in train_class_counts}
        
    # if val_labeled_count > 0:
    #     val_pct = {cls: count/val_labeled_count*100 for cls, count in val_class_counts.items()}
    # else:
    #     val_pct = {cls: 0 for cls in val_class_counts}
    
    # print("Training class distribution (counts):", train_class_counts)
    # print("Training class distribution (%):", {cls: f"{pct:.1f}%" for cls, pct in train_pct.items()})
    # print("Validation class distribution (counts):", val_class_counts)
    # print("Validation class distribution (%):", {cls: f"{pct:.1f}%" for cls, pct in val_pct.items()})
    
    return x_train, x_valid, y_train, y_valid, sw_train, sw_valid



def manual_chunking(cow_ids, chunk_size):
    """
    Divide cow_ids into chunks of approximately chunk_size.

    Parameters:
    - cow_ids: List of cow IDs to divide
    - chunk_size: Desired size for each chunk

    Returns:
    - List of lists, where each inner list contains cow IDs for one chunk
    """
    # 1) Shuffle a copy (so we don't clobber the original)
    cows = cow_ids[:]
    random.shuffle(cows)

    # 2) Compute how many chunks we need
    total_cows = len(cows)
    n_chunks = (total_cows + chunk_size - 1) // chunk_size  # Ceiling division


    # print("TOTAL COWS", total_cows)
    # print("N_CHUNKS", n_chunks)
    # 3) Compute the actual chunk sizes
    k, r = divmod(total_cows, n_chunks)
    # First 'r' chunks get size k+1, the rest get k

    # 4) Create the chunks
    chunks = []
    idx = 0
    for i in range(n_chunks):
        size = k + 1 if i < r else k
        chunks.append(cows[idx:idx + size])
        idx += size

    # 5) Shuffle the chunks
    random.shuffle(chunks)
    # print(chunks)

    return chunks

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout

from sklearn.metrics import precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from collections import Counter
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

def get_callbacks(result_dir):
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,  # Giảm learning rate nếu val_loss không cải thiện sau 3 epoch
        min_lr=1e-6,
        verbose=1
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(result_dir, 'best_weights.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    return [reduce_lr, checkpoint]


def create_keras_model(X_train, y_train):
    model = Sequential()

    # Input layer
    model.add(Dense(512, input_dim=X_train.shape[1], kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))  # Tăng Dropout nhẹ

    # Hidden layer 1
    model.add(Dense(256, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    # Hidden layer 2
    model.add(Dense(128, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))  # Tăng từ 0.2 → 0.3

    # Hidden layer 3
    model.add(Dense(64, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(y_train.shape[1], activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(
    loss='categorical_crossentropy',  # hoặc 'binary_crossentropy' nếu output là 1 node sigmoid
    optimizer=optimizer,
    metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    return model


def normalize_data(train_data, test_data):
    """
    Normalize the data using StandardScaler.
    """
    scaler = StandardScaler()
    train_data_normalized = scaler.fit_transform(train_data)
    test_data_normalized = scaler.transform(test_data)
    return train_data_normalized, test_data_normalized


def plot_combined_metrics(metric_collection, result_dir):
    """
    Plot combined Precision, Recall, F1-Score, Sensitivity, and Specificity for all models.
    Each batch size will have its own chart.
    """
    df = pd.DataFrame(metric_collection)

    # List of metrics to plot
    metrics = ["Precision", "Recall", "F1 Score", "Sensitivity", "Specificity", 
           "Best Validation Accuracy", "Test Accuracy", "Time Taken"]

    metric_titles = {
        "Precision": "Precision Comparison",
        "Recall": "Recall Comparison",
        "F1 Score": "F1-Score Comparison",
        "Sensitivity": "Sensitivity Comparison",
        "Specificity": "Specificity Comparison",
        "Best Validation Accuracy": "Validation Accuracy Comparison",  # Sửa đổi ở đây
        "Test Accuracy": "Test Accuracy Comparison",
        "Time Taken": "Training Time Comparison"
    }

    
    # Define colors and patterns for bars
    colors = plt.cm.tab10.colors  # Use a colormap with distinct colors
    patterns = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']  # Different bar patterns

    # Group by batch size
    batch_sizes = df["Batch Size"].unique()
    for batch_size in batch_sizes:
        df_batch = df[df["Batch Size"] == batch_size]
        
        batch_folder = os.path.join(result_dir, f"batch_size_{batch_size}")
        os.makedirs(batch_folder, exist_ok=True)
        
        for metric in metrics:
            if metric not in df_batch.columns:
                print(f"Metric '{metric}' not found in dataset. Skipping.")
                continue
      
            plt.figure(figsize=(14, 8))

            # Prepare data for plotting
            grouped_data = df_batch.groupby(["Model"])[metric].mean().reset_index()
            models = grouped_data["Model"].unique()

            bar_width = 0.5  # Width of each bar
            x_positions = np.arange(len(models))  # X-axis positions for models

            # Plot bars for each model
            for i, model in enumerate(models):
                model_value = grouped_data[grouped_data["Model"] == model][metric].values[0]
                plt.bar(
                    x_positions[i],
                    model_value,
                    bar_width,
                    label=f'{model}',
                    color=colors[i % len(colors)],
                    hatch=patterns[i % len(patterns)]
                )

                # Add value annotations at the top of each bar
                plt.text(
                    x_positions[i],
                    model_value + 0.01,
                    f'{model_value:.2f}',
                    ha='center',
                    fontsize=10,
                    color='black'
                )
                
            # Remove x-axis tick labels
            plt.xticks(x_positions, [''] * len(models))  # Set empty strings for x-axis ticks
            # Set x-axis labels and legend
            # plt.xticks(x_positions, models, rotation=45, ha='right')  # Rotate model names for readability
            plt.ylabel(metric)
            # plt.title(f'{metric_titles[metric]} (Batch Size: {batch_size})')
            plt.legend(loc='upper left', title="Models", fontsize=10)
            plt.tight_layout()

            # Save the plot
            plt.savefig(os.path.join(batch_folder, f'{metric.lower().replace(" ", "_")}_batch_size_{batch_size}_comparison.png'))
            plt.close()

    print("All combined metric comparison plots saved.")


def plot_epoch_based_metrics(all_histories, result_dir):
    """
    Vẽ biểu đồ timeline của Train Loss, Validation Loss, Train Accuracy, Validation Accuracy
    theo các giá trị batch_size.
    """
    # Convert `all_histories` dictionary to a DataFrame
    metrics_list = []
    for model_name, model_histories in all_histories.items():
        for history_entry in model_histories:
            batch_size = history_entry["batch_size"]
            epoch = history_entry["epoch"]
            history = history_entry["history"]
            
            for epoch_idx, (train_loss, val_loss, train_acc, val_acc) in enumerate(
                zip(history["loss"], history["val_loss"], history["accuracy"], history["val_accuracy"])
            ):
                metrics_list.append({
                    "Model": model_name,
                    "Batch Size": batch_size,
                    "Epoch": epoch_idx + 1,
                    "Train Loss": train_loss,
                    "Validation Loss": val_loss,
                    "Train Accuracy": train_acc,
                    "Validation Accuracy": val_acc,
                })

    # Convert to DataFrame
    df = pd.DataFrame(metrics_list)

    # Metrics cần vẽ
    metrics = ["Train Loss", "Validation Loss", "Train Accuracy", "Validation Accuracy"]

    # Lặp qua từng batch size
    batch_sizes = df["Batch Size"].unique()
    for batch_size in batch_sizes:
        batch_folder = os.path.join(result_dir, f"batch_size_{batch_size}")
        os.makedirs(batch_folder, exist_ok=True)

        for metric in metrics:
            plt.figure(figsize=(14, 8))
            batch_df = df[df["Batch Size"] == batch_size]
            # for model_name, model_df in batch_df.groupby("Model"):
            #     epochs = model_df["Epoch"].values
            #     metric_values = model_df[metric].values

            #     # Vẽ đường timeline cho mỗi mô hình
            #     plt.plot(epochs, metric_values, label=model_name, marker='o', linestyle='-')
            
            for model_name, model_df in batch_df.groupby("Model"):
                grouped = model_df.groupby("Epoch")[metric].mean().reset_index()
                epochs = grouped["Epoch"]
                metric_values = grouped[metric]
                plt.plot(epochs, metric_values, label=model_name, marker='o', linestyle='-')
            

            plt.xlabel("Epochs", fontsize=12)
            plt.ylabel(metric, fontsize=12)
            # plt.title(f"{metric} Timeline Comparison Across Models (Batch Size: {batch_size})", fontsize=14)
            plt.grid(alpha=0.3)
            plt.legend(title="Models", loc="best", fontsize=10)
            plt.tight_layout()

            # Lưu biểu đồ
            plot_path = os.path.join(batch_folder, f"{metric.lower().replace(' ', '_')}_batch_size_{batch_size}_timeline_comparison.png")
            plt.savefig(plot_path)
            plt.close()

    print(f"Epoch-based timeline comparison plots saved.")
      
# def plot_all_figures(batch_size, epoch, history, y_true_labels, y_pred_labels, y_pred_probs, categories, result_out, model_name):
def plot_all_figures(batch_size, epoch, history, y_true_labels, y_pred_labels, y_pred_probs, categories, result_out, model_name):
    """
    Plots Accuracy, Loss, Confusion Matrix, ROC Curve, and Accuracy vs. Recall plots.
    """
    # 1. Accuracy Plot
    plt.figure()
    plt.plot(history.history['accuracy'], linestyle='--')
    plt.plot(history.history['val_accuracy'], linestyle=':')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(result_out, model_name + f'_bs{batch_size}_ep{epoch}_accuracy_plot.png'))
    plt.close()

    # 2. Loss Plot
    plt.figure()
    plt.plot(history.history['loss'], linestyle='--')
    plt.plot(history.history['val_loss'], linestyle=':')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(os.path.join(result_out, model_name + f'_bs{batch_size}_ep{epoch}_loss_plot.png'))
    plt.close()

    # 3. Confusion Matrix Plot with Float Numbers
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(os.path.join(result_out, model_name + f'_bs{batch_size}_ep{epoch}_confusion_matrix_normalized.png'))
    plt.close()

    # Encode the true labels to binary format
    label_encoder = LabelEncoder()
    y_true_binary = label_encoder.fit_transform(y_true_labels)

    # 4. ROC Curve Plot for each class in a one-vs-rest fashion
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10.colors  # Simplified colors
    line_styles = ['-', '--', '-.', ':']  # Updated line styles
    line_width = 1.5  # Reduced line thickness

    # Ensure y_true_labels and y_pred_labels are NumPy arrays and encode labels if they are not integers
    label_encoder = LabelEncoder()
    if isinstance(y_true_labels[0], str) or isinstance(y_true_labels[0], bool):
        y_true_labels = label_encoder.fit_transform(y_true_labels)
    else:
        y_true_labels = np.array(y_true_labels)

    if isinstance(y_pred_labels[0], str) or isinstance(y_pred_labels[0], bool):
        y_pred_labels = label_encoder.transform(y_pred_labels)
    else:
        y_pred_labels = np.array(y_pred_labels)

    if len(categories) == 2:  # Binary classification case
        # Plotting for the positive class (1)
        y_true_binary = (y_true_labels == 1).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[1], linestyle=line_styles[0], linewidth=line_width, label=f'{categories[1]} (AUC = {roc_auc:.4f})')
        
        # Plotting for the negative class (0)
        y_true_binary = (y_true_labels == 0).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, 0])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[0], linestyle=line_styles[1], linewidth=line_width, label=f'{categories[0]} (AUC = {roc_auc:.4f})')
        
    else:  # Multi-class case
        for i, class_name in enumerate(categories):
            y_true_binary = (y_true_labels == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr, tpr,
                color=colors[i % len(colors)],
                linestyle=line_styles[i % len(line_styles)],
                linewidth=line_width,
                label=f'{class_name} (AUC = {roc_auc:.4f})'
            )

    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.0, label="Chance (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multiple Classes')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_out, model_name + f'_bs{batch_size}_ep{epoch}_roc_curve.png'))
    plt.close()

    # 5. Accuracy vs. Recall Plot
    report = classification_report(y_true_labels, y_pred_labels, target_names=categories, output_dict=True)
    accuracy = [report[category]['precision'] for category in categories]
    recall = [report[category]['recall'] for category in categories]

    plt.figure()
    plt.plot(categories, accuracy, marker='o', linestyle='--', color='b', label='Accuracy')
    plt.plot(categories, recall, marker='o', linestyle='-', color='g', label='Recall')
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.legend(loc='best')
    plt.savefig(os.path.join(result_out, model_name + f'_bs{batch_size}_ep{epoch}_accuracy_vs_recall.png'))
    plt.close()

    print(f"All plots saved to {result_out}")

    # 6. Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    if len(categories) == 2:  # Binary classification case
        # Plotting for the positive class (1)
        y_true_binary = (y_true_labels == 1).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_probs[:, 1])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, color=colors[1], linestyle=line_styles[0], linewidth=line_width, 
                 label=f'{categories[1]} (PR AUC = {pr_auc:.4f})')

        # Plotting for the negative class (0)
        y_true_binary = (y_true_labels == 0).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_probs[:, 0])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, color=colors[0], linestyle=line_styles[1], linewidth=line_width, 
                 label=f'{categories[0]} (PR AUC = {pr_auc:.4f})')
    else:  # Multi-class case
        for i, class_name in enumerate(categories):
            y_true_binary = (y_true_labels == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_probs[:, i])
            pr_auc = auc(recall, precision)
            plt.plot(
                recall, precision,
                color=colors[i % len(colors)],
                linestyle=line_styles[i % len(line_styles)],
                linewidth=line_width,
                label=f'{class_name} (PR AUC = {pr_auc:.4f})'
            )

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_out, model_name + f'_bs{batch_size}_ep{epoch}_precision_recall_curve.png'))
    plt.close()
    
    
from sklearn.metrics import roc_auc_score

def run_layer_experiment(epoch_values, batch_size_list, metric_collection,
                         base_feature_dir, handcrafted_folders,
                         layers, result_dir, categories):

    performance_metrics = []
    all_histories = {}
    best_model_info = None
    best_score = -1

    for layer_name in layers:
        print(f"Running experiment for layer: {layer_name}")
        all_histories[layer_name] = []

        layer_result_out = os.path.join(result_dir, layer_name)
        os.makedirs(layer_result_out, exist_ok=True)

        label_log_path = os.path.join(layer_result_out, f"{layer_name}_label_distribution.txt")
        label_log_file = open(label_log_path, "w")

        X_train, y_train = load_with_cache(
            base_feature_dir=base_feature_dir,
            handcrafted_folders=handcrafted_folders,
            categories=categories,
            result_dir=result_dir,
            split='train',
            layer_name=layer_name
        )

        if len(X_train) == 0:
            print(f"[ERROR] No training data available for layer {layer_name}. Skipping...")
            continue

        val_dir = os.path.join(base_feature_dir, 'val')
        if os.path.exists(val_dir):
            X_val, y_val = load_with_cache(
                base_feature_dir=base_feature_dir,
                handcrafted_folders=handcrafted_folders,
                categories=categories,
                result_dir=result_dir,
                split='val',
                layer_name=layer_name
            )
            X_all = np.concatenate([X_train, X_val], axis=0)
            y_all = np.concatenate([y_train, y_val], axis=0)
        else:
            print(f"[INFO] Validation set not found. Using training set only for CV...")
            X_all, y_all = X_train, y_train

        nan_mask = ~np.isnan(X_all).any(axis=1)
        if not nan_mask.all():
            removed_count = (~nan_mask).sum()
            print(f"[⚠️ WARN] Removed {removed_count} samples with NaN values.")
            X_all = X_all[nan_mask]
            y_all = y_all[nan_mask]

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_all)
        num_classes = len(np.unique(y_encoded))

        for batch_size in batch_size_list:
            for epoch in epoch_values:
                rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
                fold = 1
                fold_metrics = []

                for train_val_index, pseudo_test_index in rkf.split(X_all, y_encoded):
                    fold_result_dir = os.path.join(layer_result_out, f"fold_{fold}")
                    os.makedirs(fold_result_dir, exist_ok=True)

                    X_pseudo_test = X_all[pseudo_test_index]
                    y_pseudo_test = y_encoded[pseudo_test_index]

                    X_train_val = X_all[train_val_index]
                    y_train_val = y_encoded[train_val_index]

                    train_val_counter = Counter(label_encoder.inverse_transform(y_train_val))
                    pseudo_test_counter = Counter(label_encoder.inverse_transform(y_pseudo_test))

                    label_log_file.write(f"=== Fold {fold} ===\n")
                    label_log_file.write(f"Train+Val distribution:\n")
                    for cls, count in train_val_counter.items():
                        label_log_file.write(f"  {cls}: {count}\n")
                    label_log_file.write(f"Pseudo-Test distribution:\n")
                    for cls, count in pseudo_test_counter.items():
                        label_log_file.write(f"  {cls}: {count}\n")
                    label_log_file.write("\n")

                    X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
                        X_train_val, y_train_val, test_size=0.15, random_state=42, stratify=y_train_val
                    )

                    X_train_input, X_val = normalize_data(
                        X_train_fold.reshape(X_train_fold.shape[0], -1),
                        X_val_fold.reshape(X_val_fold.shape[0], -1)
                    )
                    _, X_pseudo_test = normalize_data(X_train_input, X_pseudo_test.reshape(X_pseudo_test.shape[0], -1))

                    y_train_cat = to_categorical(y_train_fold, num_classes)
                    y_val_cat = to_categorical(y_val_fold, num_classes)

                    class_weights_array = compute_class_weight(
                        class_weight='balanced',
                        classes=np.unique(y_train_fold),
                        y=y_train_fold
                    )
                    class_weights = dict(enumerate(class_weights_array))

                    model = create_keras_model(X_train_input, y_train_cat)
                    callbacks = get_callbacks(layer_result_out)

                    start_time = time.time()
                    history = model.fit(
                        X_train_input, y_train_cat,
                        validation_data=(X_val, y_val_cat),
                        batch_size=batch_size,
                        epochs=epoch,
                        verbose=0,
                        class_weight=class_weights
                    )
                    end_time = time.time()

                    best_epoch = np.argmax(history.history['val_accuracy']) + 1
                    best_val_accuracy = history.history['val_accuracy'][best_epoch - 1]

                    y_pred_probs = model.predict(X_pseudo_test)
                    y_pred_labels = np.argmax(y_pred_probs, axis=1)
                    y_true_labels = y_pseudo_test

                    # === Tính AUC ===
                    if num_classes == 2:
                        auc_value = roc_auc_score(y_true_labels, y_pred_probs[:, 1])
                    else:
                        auc_value = roc_auc_score(y_true_labels, y_pred_probs, multi_class='ovr', average='macro')

                    cm = confusion_matrix(y_true_labels, y_pred_labels)
                    test_accuracy = accuracy_score(y_true_labels, y_pred_labels)
                    precision = precision_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)
                    recall = recall_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)
                    f1 = f1_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)

                    report = classification_report(y_true_labels, y_pred_labels, target_names=categories, output_dict=True)

                    fold_metrics.append({
                        "F1": f1,
                        "Recall": recall,
                        "Precision": precision,
                        "Accuracy": test_accuracy,
                        "Macro F1": report["macro avg"]["f1-score"],
                        "Macro Recall": report["macro avg"]["recall"],
                        "Macro Precision": report["macro avg"]["precision"],
                        "Macro AUC": auc_value,
                        "Best Epoch": best_epoch,
                        "Best Validation Accuracy": best_val_accuracy,
                        "Time Taken": end_time - start_time,
                    })

                    all_histories[layer_name].append({
                        "batch_size": batch_size,
                        "epoch": epoch,
                        "history": history.history
                    })

                    plot_all_figures(
                        batch_size=batch_size,
                        epoch=epoch,
                        history=history,
                        y_true_labels=y_true_labels,
                        y_pred_labels=y_pred_labels,
                        y_pred_probs=y_pred_probs,
                        categories=categories,
                        result_out=fold_result_dir,
                        model_name=layer_name
                    )

                    fold += 1

                df_fold = pd.DataFrame(fold_metrics)

                metric_summary = {
                    "Model": layer_name,
                    "Layer": layer_name,
                    "Batch Size": batch_size,
                    "Epoch": epoch,
                    "Best Epoch": int(df_fold["Best Epoch"].mean()),
                    "Best Validation Accuracy": df_fold["Best Validation Accuracy"].mean(),
                    "Test Accuracy": df_fold["Accuracy"].mean(),
                    "Precision": df_fold["Precision"].mean(),
                    "Recall": df_fold["Recall"].mean(),
                    "F1 Score": df_fold["F1"].mean(),
                    "Macro F1": df_fold["Macro F1"].mean(),
                    "Macro Precision": df_fold["Macro Precision"].mean(),
                    "Macro Recall": df_fold["Macro Recall"].mean(),
                    "Macro AUC": df_fold["Macro AUC"].mean(),
                    "Time Taken": df_fold["Time Taken"].mean(),
                    "Std F1": df_fold["F1"].std(),
                    "Std Recall": df_fold["Recall"].std()
                }

                performance_metrics.append(metric_summary)

                score = (
                    metric_summary["Macro Recall"] +
                    metric_summary["Macro F1"] +
                    metric_summary["Macro Precision"] +
                    metric_summary["Macro AUC"] -
                    (metric_summary["Std F1"] + metric_summary["Std Recall"])
                )

                if score > best_score:
                    best_score = score
                    best_model_info = {
                        "layer_name": layer_name,
                        "batch_size": batch_size,
                        "epoch": epoch,
                        "label_encoder": label_encoder,
                        "train_images": X_all,
                        "train_labels": y_encoded,
                        "num_classes": num_classes
                    }

        label_log_file.close()

    if best_model_info:
        print("\nRetraining best model on full data and saving it...")

        X_resampled = best_model_info["train_images"].reshape(best_model_info["train_images"].shape[0], -1)
        X_resampled, _ = normalize_data(X_resampled, X_resampled)
        y_cat = to_categorical(best_model_info["train_labels"], best_model_info["num_classes"])

        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(best_model_info["train_labels"]),
            y=best_model_info["train_labels"]
        )
        class_weights = dict(enumerate(class_weights_array))

        model = create_keras_model(X_resampled, y_cat)
        model.fit(X_resampled, y_cat, batch_size=best_model_info["batch_size"],
                  epochs=best_model_info["epoch"], verbose=0, class_weight=class_weights)

        model.save(os.path.join(result_dir, "best_model.h5"))
        print(f"✅ Best model saved to {os.path.join(result_dir, 'best_model.h5')}")

        import pickle
        with open(os.path.join(result_dir, "label_encoder.pkl"), "wb") as f:
            pickle.dump(best_model_info["label_encoder"], f)

    metric_collection.extend(performance_metrics)
    performance_df = pd.DataFrame(performance_metrics)
    performance_df.to_csv(os.path.join(result_dir, 'performance_metrics_repeated_kfold.csv'), index=False)

    print("📊 Performance metrics with Repeated K-Fold CV saved.")
    return all_histories, metric_collection, best_model_info



def extract_prefix(fname):
    """
    Trích tiền tố file, ví dụ: ISIC_123456_feature.npy → ISIC_123456
    """
    return fname.split('_feature')[0]

def load_combined_features(base_dir, split, categories, handcrafted_folders, layer):
    """
    Load và kết hợp đặc trưng từ BlockNet + handcrafted cho một layer cụ thể.

    Args:
        base_dir (str): thư mục gốc chứa đặc trưng
        split (str): 'train', 'val', hoặc 'test'
        categories (list): danh sách nhãn (MEL, NV, ...)
        handcrafted_folders (list): danh sách thư mục handcrafted (vd: ['hsv_histograms', 'lbp', ...])
        layer (str): tên layer (vd: block2_conv1)

    Returns:
        X (np.ndarray): đặc trưng đầu vào
        y (np.ndarray): nhãn tương ứng
    """
    X, y = [], []
    total_handcrafted_count = 0
    total_blocknet_vector_count = 0
    total_combined_success = 0

    for label in categories:
        vgg_dir = os.path.join(base_dir, split, 'blocknet_features', label, layer)
        handcrafted_dirs = [
            os.path.join(base_dir, split, f"{feat}_features", label)
            for feat in handcrafted_folders
        ]

        if not os.path.exists(vgg_dir):
            print(f"[WARN] Missing VGG directory: {vgg_dir}")
            continue

        handcrafted_files_set = set()
        for feat_dir in handcrafted_dirs:
            if os.path.exists(feat_dir):
                handcrafted_files_set.update([
                    extract_prefix(f) for f in os.listdir(feat_dir) if f.endswith('.npy')
                ])
        handcrafted_count = len(handcrafted_files_set)
        total_handcrafted_count += handcrafted_count

        blocknet_files = [f for f in os.listdir(vgg_dir) if f.endswith('.npy')]
        total_blocknet_vector_count += len(blocknet_files)

        num_combined = 0
        for fname in blocknet_files:
            prefix = extract_prefix(fname)
            try:
                vgg_path = os.path.join(vgg_dir, fname)
                vgg_vec = np.load(vgg_path)

                handcrafted_vecs = []
                for feat_dir in handcrafted_dirs:
                    matched_files = [f for f in os.listdir(feat_dir) if f.startswith(prefix + '_') and f.endswith('.npy')]
                    if matched_files:
                        path = os.path.join(feat_dir, matched_files[0])
                        handcrafted_vecs.append(np.load(path))
                    else:
                        raise FileNotFoundError(f"Missing handcrafted feature for: {prefix} in {feat_dir}")

                # ✅ Ghép tất cả đặc trưng
                full_vec = np.concatenate(handcrafted_vecs + [vgg_vec])
                X.append(full_vec)
                y.append(label)
                num_combined += 1

            except Exception as e:
                print(f"[ERROR] Skipping {fname}: {e}")

        total_combined_success += num_combined
        print(f"[INFO] {split.upper()} - {label} - Layer: {layer} → Combined {num_combined} samples.")


    return np.array(X), np.array(y)


def load_with_cache(base_feature_dir, handcrafted_folders, categories, result_dir, split, layer_name):
    """
    Load kết quả ghép đặc trưng từ cache (.npz) nếu đã tồn tại.
    Nếu chưa, gọi load_combined_features và lưu lại cache.

    Args:
        base_feature_dir (str): thư mục gốc chứa đặc trưng.
        handcrafted_folders (list): các thư mục chứa đặc trưng thủ công.
        categories (list): danh sách nhãn.
        result_dir (str): thư mục kết quả (nơi chứa features_cache).
        split (str): 'train', 'val', 'test'
        layer_name (str): tên layer CNN.

    Returns:
        X (np.ndarray): đặc trưng đầu vào.
        y (np.ndarray): nhãn tương ứng.
    """
    handcrafted_str = "_".join(sorted(handcrafted_folders)) if handcrafted_folders else "none"
    cache_dir = os.path.join(result_dir, "features_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = f"{split}_{layer_name}_{handcrafted_str}_combined.npz"
    cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(cache_path):
        print(f"[✅ CACHE] Loaded: {cache_path}")
        data = np.load(cache_path)
        return data['X'], data['y']
    else:
        print(f"[⏳ LOAD] Loading and caching features for {split} - {layer_name} - [{handcrafted_str}]")
        X, y = load_combined_features(
            base_dir=base_feature_dir,
            split=split,
            categories=categories,
            handcrafted_folders=handcrafted_folders,
            layer=layer_name
        )
        np.savez_compressed(cache_path, X=X, y=y)
        print(f"[💾 CACHED] Saved to {cache_path}")
        return X, y


def evaluate_on_unlabeled_test_set(model_path, label_encoder_path, feature_dir, result_dir, best_layer, handcrafted_folders):
    import pickle

    print("🔎 Predicting on the unlabeled test set...")

    # 🔍 Kiểm tra file model và encoder trước
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    if not os.path.exists(label_encoder_path):
        print(f"❌ Label encoder file not found: {label_encoder_path}")
        return

    # 📦 Cache path
    cache_dir = os.path.join(result_dir, "features_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"test_unlabeled_{best_layer}_combined.npz")

    # ✅ Load từ cache nếu có
    if os.path.exists(cache_path):
        print(f"[✅ CACHE] Loaded unlabeled test features from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        X = data['X']
        img_names = data['img_names'].tolist()
    else:
        print("🔄 Extracting features from disk (no cache found)...")
        test_dir = os.path.join(feature_dir, 'test')
        unlabeled_dir = os.path.join(test_dir, 'blocknet_features', 'unlabeled', best_layer)
        handcrafted_dirs = [
            os.path.join(test_dir, f"{feat}_features", "unlabeled")
            for feat in handcrafted_folders
        ]

        X, img_names = [], []

        for fname in sorted(os.listdir(unlabeled_dir)):
            if not fname.endswith('.npy'):
                continue

            prefix = fname.split('_feature')[0]
            try:
                vgg_vec = np.load(os.path.join(unlabeled_dir, fname))
                handcrafted_vecs = []

                for feat_dir in handcrafted_dirs:
                    matched_files = [
                        f for f in os.listdir(feat_dir)
                        if f.startswith(prefix + '_') and f.endswith('.npy')
                    ]
                    if matched_files:
                        path = os.path.join(feat_dir, matched_files[0])
                        handcrafted_vecs.append(np.load(path))
                    else:
                        raise FileNotFoundError(f"Missing handcrafted feature for image {prefix} in {feat_dir}")

                full_vec = np.concatenate(handcrafted_vecs + [vgg_vec])
                X.append(full_vec)
                img_names.append(prefix + '.jpg')

            except Exception as e:
                print(f"[⚠️ WARN] Skipping {fname}: {e}")

        if not X:
            print("[❌ ERROR] No test samples loaded from the unlabeled set.")
            return

        X = np.array(X)
        img_names = np.array(img_names)

        # 💾 Save cache
        np.savez_compressed(cache_path, X=X, img_names=img_names)
        print(f"[💾 CACHED] Saved unlabeled test features to {cache_path}")

    # ✅ Normalize
    X = X.reshape(X.shape[0], -1)
    X, _ = normalize_data(X, X)

    # ✅ Load model and label encoder
    model = tf.keras.models.load_model(model_path)
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    # 🔮 Predict
    probs = model.predict(X)
    preds = np.argmax(probs, axis=1)
    pred_labels = label_encoder.inverse_transform(preds)

    # 📄 Save predictions to CSV
    rows = []
    for img_name, pred_label, pred_idx, prob in zip(img_names, pred_labels, preds, probs):
        row = {
            "Image Name": img_name,
            "Predicted Label": pred_label,
            "Predicted Label (encoded)": pred_idx
        }
        for i, class_name in enumerate(label_encoder.classes_):
            row[f"Prob_{class_name}"] = prob[i]

        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = os.path.join(result_dir, "unlabeled_test_predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Predictions for unlabeled test set saved to: {out_path}")


def evaluate_on_test_set(model_path, label_encoder_path, feature_dir, result_dir, best_layer, categories, handcrafted_folders):
    import pickle

    print("🔍 Predicting and evaluating on labeled test set...")

    # Cache path
    cache_dir = os.path.join(result_dir, "features_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"test_labeled_{best_layer}_combined.npz")

    # Load from cache if available
    if os.path.exists(cache_path):
        print(f"[✅ CACHE] Loaded test set from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        X_test = data['X']
        y_test = data['y']
        prefix_names = data['prefix_names'].tolist()
    else:
        print("🔄 Extracting test features from disk...")
        handcrafted_dirs = [
            os.path.join(feature_dir, 'test', f"{feat}_features") for feat in handcrafted_folders
        ]

        X_test, y_test, prefix_names = [], [], []

        for label in categories:
            vgg_dir = os.path.join(feature_dir, 'test', 'blocknet_features', label, best_layer)
            handcrafted_subdirs = [os.path.join(hd, label) for hd in handcrafted_dirs]

            if not os.path.exists(vgg_dir):
                continue

            for fname in sorted(os.listdir(vgg_dir)):
                if not fname.endswith('.npy'):
                    continue
                try:
                    prefix = fname.split('_feature')[0]
                    vgg_path = os.path.join(vgg_dir, fname)
                    vgg_vec = np.load(vgg_path)

                    handcrafted_vecs = []
                    for feat_dir in handcrafted_subdirs:
                        matched = [f for f in os.listdir(feat_dir) if f.startswith(prefix + '_') and f.endswith('.npy')]
                        if matched:
                            path = os.path.join(feat_dir, matched[0])
                            handcrafted_vecs.append(np.load(path))
                        else:
                            raise FileNotFoundError(f"Missing handcrafted feature for {prefix} in {feat_dir}")

                    full_vec = np.concatenate(handcrafted_vecs + [vgg_vec])
                    X_test.append(full_vec)
                    y_test.append(label)
                    prefix_names.append(prefix + '.jpg')

                except Exception as e:
                    print(f"[WARN] Skipping {fname}: {e}")

        X_test = np.array(X_test)
        y_test = np.array(y_test)
        prefix_names = np.array(prefix_names)

        np.savez_compressed(cache_path, X=X_test, y=y_test, prefix_names=prefix_names)
        print(f"[💾 CACHED] Saved test features to {cache_path}")

    # Preprocessing
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_test, _ = normalize_data(X_test, X_test)

    # Load model and label encoder
    model = tf.keras.models.load_model(model_path)
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    y_test_encoded = label_encoder.transform(y_test)
    y_test_probs = model.predict(X_test)
    y_test_preds = np.argmax(y_test_probs, axis=1)

    # Metrics
    cm = confusion_matrix(y_test_encoded, y_test_preds)
    test_acc = accuracy_score(y_test_encoded, y_test_preds)
    precision = precision_score(y_test_encoded, y_test_preds, average='weighted', zero_division=0)
    recall = recall_score(y_test_encoded, y_test_preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test_encoded, y_test_preds, average='weighted', zero_division=0)
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if cm.shape[0] > 1 and (cm[1, 0] + cm[1, 1]) > 0 else 0
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm.shape[0] > 1 and (cm[0, 0] + cm[0, 1]) > 0 else 0

    # Save report
    report_txt = classification_report(y_test_encoded, y_test_preds, target_names=categories)
    report_dict = classification_report(y_test_encoded, y_test_preds, target_names=categories, output_dict=True)
    with open(os.path.join(result_dir, "final_test_classification_report.txt"), "w") as f:
        f.write(report_txt)

    eval_txt_path = os.path.join(result_dir, "final_test_evaluation_metrics.txt")
    with open(eval_txt_path, "w") as f:
        f.write("📊 Final Test Set Evaluation Metrics\n")
        f.write("=" * 40 + "\n")
        f.write(f"Best Layer     : {best_layer}\n")
        f.write(f"Accuracy       : {test_acc:.4f}\n")
        f.write(f"Precision      : {precision:.4f}\n")
        f.write(f"Recall         : {recall:.4f}\n")
        f.write(f"F1 Score       : {f1:.4f}\n")
        f.write(f"Sensitivity    : {sensitivity:.4f}\n")
        f.write(f"Specificity    : {specificity:.4f}\n\n")
        f.write("🧩 Confusion Matrix (Raw Counts):\n")
        f.write(str(cm))
    print(f"✅ Final evaluation metrics saved to {eval_txt_path}")

    # Save per-image predictions
    rows = []
    for prefix_name, true_label, pred_idx in zip(prefix_names, y_test, y_test_preds):
        true_encoded = label_encoder.transform([true_label])[0]
        pred_decoded = label_encoder.inverse_transform([pred_idx])[0]

        row = {
            "Image Name": prefix_name,
            "True Label": true_label,
            "Predicted Label": pred_decoded,
            "True Label (encoded)": true_encoded,
            "Predicted Label (encoded)": pred_idx,
            "Accuracy": test_acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Sensitivity": sensitivity,
            "Specificity": specificity
        }

        for cls in categories:
            if cls in report_dict:
                row[f"{cls} Precision"] = report_dict[cls]["precision"]
                row[f"{cls} Recall"] = report_dict[cls]["recall"]
                row[f"{cls} F1-Score"] = report_dict[cls]["f1-score"]

        rows.append(row)

    df_result = pd.DataFrame(rows)
    df_result.to_csv(os.path.join(result_dir, "final_test_predictions.csv"), index=False)
    print(f"✅ Full predictions with per-class metrics saved to final_test_predictions.csv")


def main():
    # Directory for the BlockNet dataset
    base_dir = os.getcwd()
    home_dir = os.path.join(base_dir, 'data9') 
    feature_dir = os.path.join(home_dir, 'data9_SOTA_and_handcrafts_and_BlookNet_optimal_entropy_features_v3')
    result_dir = os.path.join(home_dir, 'training_data9_SOTA_and_handcrafts_and_BlookNet_optimal_entropy_features_v3_v10_2')
    os.makedirs(result_dir, exist_ok=True)

    # Define categories and layers
    # categories = ['MEL',	'NV',	'BCC',	'AK',	'BKL',	'DF',	'VASC',	'SCC']
    # categories = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
    categories = ['benign', 'malignant']
    num_classes = len(categories)

    # batch_size_list = [8, 16, 32, 64, 128]
    # epoch_values = [200]
    
    batch_size_list = [32]
    epoch_values = [100]
    
    metric_collection = []

    print("Generating paths for layers and categories...")

    layers = ['block2_conv1', 'block1_conv2', 'block1_conv1', 'block3_conv2']
    handcrafted_folders = ['hsv_histograms', 'color_histograms', 'fractal']
    
    print("Starting experiments for each layer...")
    # ✅ Huấn luyện và chọn mô hình tốt nhất
    # ✅ Huấn luyện với Repeated K-Fold + chọn mô hình tốt nhất
    all_histories, metric_collection, best_model_info = run_layer_experiment(
        epoch_values=epoch_values,
        batch_size_list=batch_size_list,
        metric_collection=metric_collection,
        base_feature_dir=feature_dir,
        handcrafted_folders=handcrafted_folders,
        layers=layers,
        result_dir=result_dir,
        categories=categories
    )

    # ✅ Ghi file tổng hợp kết quả
    df_metrics = pd.DataFrame(metric_collection)
    df_metrics["F1 (±)"] = df_metrics.apply(lambda row: f"{row['Macro F1']:.4f} ± {row['Std F1']:.4f}", axis=1)
    df_metrics["Recall (±)"] = df_metrics.apply(lambda row: f"{row['Macro Recall']:.4f} ± {row['Std Recall']:.4f}", axis=1)

    df_metrics.to_csv(os.path.join(result_dir, 'overall_performance_metrics.csv'), index=False)
    print(f"📊 Saved macro-level metrics to overall_performance_metrics.csv")

    # ✅ Lưu riêng chỉ số của best model
    if best_model_info:
        best_layer = best_model_info["layer_name"]
        best_batch_size = best_model_info["batch_size"]
        best_epoch = best_model_info["epoch"]

        best_row = df_metrics[
            (df_metrics["Layer"] == best_layer) &
            (df_metrics["Batch Size"] == best_batch_size) &
            (df_metrics["Epoch"] == best_epoch)
        ]
        
        if not best_row.empty:
            best_row.to_csv(os.path.join(result_dir, "best_model_performance.csv"), index=False)
            print(f"🏅 Best model performance saved to best_model_performance.csv")
        else:
            print(f"[⚠️] Best model info not found in metrics DataFrame.")

    # ✅ Vẽ biểu đồ
    plot_combined_metrics(metric_collection, result_dir)
    plot_epoch_based_metrics(all_histories, result_dir)

    # Đường dẫn tới model và encoder
    best_model_path = os.path.join(result_dir, "best_model.h5")
    label_encoder_path = os.path.join(result_dir, "label_encoder.pkl")

    # ✅ Đánh giá tập test có nhãn (bật nếu cần)
    # evaluate_on_test_set(
    #     model_path=best_model_path,
    #     label_encoder_path=label_encoder_path,
    #     feature_dir=feature_dir,
    #     result_dir=result_dir,
    #     best_layer=best_model_info["layer_name"],
    #     categories=categories,
    #     handcrafted_folders=handcrafted_folders
    # )

    # ✅ Dự đoán tập test không gán nhãn
    evaluate_on_unlabeled_test_set(
        model_path=best_model_path,
        label_encoder_path=label_encoder_path,
        feature_dir=feature_dir,
        result_dir=result_dir,
        best_layer=best_model_info["layer_name"],
        handcrafted_folders=handcrafted_folders
    )

    print("✅ All tasks completed successfully.")


if __name__ == "__main__":
    main()


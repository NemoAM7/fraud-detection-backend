import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Define the sampling methods (based on the server.py file and the available CSVs)
sampling_methods = [
    "No_Sampling",
    "Random_Undersampling", 
    "Random_Oversampling", 
    "SMOTE",
    "ADASYN",
    "SMOTEENN",
    "SMOTETomek"
]

# Define models to plot (focusing on KNN and Gradient Boosting as requested)
models = ["KNN", "Gradient_Boosting"]

# Function to read confusion matrix from CSV
def read_confusion_matrix(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path, index_col=0)
        # Convert to numpy array
        cm = df.values
        return cm
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Function to calculate metrics from confusion matrix
def calculate_metrics(cm):
    if cm is None:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "specificity": 0}
    
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # also known as sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1
    }

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title, ax):
    if cm is None:
        ax.text(0.5, 0.5, "Data not available", horizontalalignment='center', verticalalignment='center')
        ax.set_title(title)
        return {}
    
    # Calculate metrics
    metrics = calculate_metrics(cm)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    # Add metrics to title
    metrics_text = f"Acc: {metrics['accuracy']:.3f}, Prec: {metrics['precision']:.3f}, Rec: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}"
    ax.set_title(f"{title}\n{metrics_text}")
    
    # Set x and y tick labels
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])
    
    return metrics

# Create a directory for the output plots
output_dir = "confusion_matrix_plots"
os.makedirs(output_dir, exist_ok=True)

# Store metrics for later comparison
all_metrics = {}

# Plot KNN confusion matrices
fig, axes = plt.subplots(1, len(sampling_methods), figsize=(24, 4))
fig.suptitle('KNN Confusion Matrices with Different Sampling Methods', fontsize=16)

knn_metrics = {}
for i, sampling_method in enumerate(sampling_methods):
    file_path = f"confusion_matrices/KNN_{sampling_method}_cm.csv"
    cm = read_confusion_matrix(file_path)
    metrics = plot_confusion_matrix(cm, f"KNN with {sampling_method.replace('_', ' ')}", axes[i])
    knn_metrics[sampling_method] = metrics

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{output_dir}/knn_confusion_matrices.png", dpi=300, bbox_inches='tight')
all_metrics["KNN"] = knn_metrics

# Plot Gradient Boosting confusion matrices
fig, axes = plt.subplots(1, len(sampling_methods), figsize=(24, 4))
fig.suptitle('Gradient Boosting Confusion Matrices with Different Sampling Methods', fontsize=16)

gb_metrics = {}
for i, sampling_method in enumerate(sampling_methods):
    file_path = f"confusion_matrices/Gradient_Boosting_{sampling_method}_cm.csv"
    cm = read_confusion_matrix(file_path)
    metrics = plot_confusion_matrix(cm, f"Gradient Boosting with {sampling_method.replace('_', ' ')}", axes[i])
    gb_metrics[sampling_method] = metrics

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{output_dir}/gradient_boosting_confusion_matrices.png", dpi=300, bbox_inches='tight')
all_metrics["Gradient_Boosting"] = gb_metrics

# Plot combined figure with just the top sampling methods for both models
# Let's choose 3 sampling methods: Random_Undersampling, SMOTE, and SMOTETomek
top_sampling_methods = ["Random_Undersampling", "SMOTE", "SMOTETomek"]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Comparison of KNN and Gradient Boosting with Different Sampling Methods', fontsize=16)

for i, model in enumerate(models):
    for j, sampling_method in enumerate(top_sampling_methods):
        file_path = f"confusion_matrices/{model}_{sampling_method}_cm.csv"
        cm = read_confusion_matrix(file_path)
        plot_confusion_matrix(cm, f"{model.replace('_', ' ')} with {sampling_method.replace('_', ' ')}", axes[i, j])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')

# Create performance comparison bar charts
metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
for metric in metrics_to_plot:
    plt.figure(figsize=(12, 6))
    
    # Prepare data for bar chart
    model_data = []
    for model in models:
        values = []
        for sampling in sampling_methods:
            if model in all_metrics and sampling in all_metrics[model]:
                values.append(all_metrics[model][sampling].get(metric, 0))
            else:
                values.append(0)
        model_data.append(values)
    
    # Create bar chart
    x = np.arange(len(sampling_methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    rects1 = ax.bar(x - width/2, model_data[0], width, label=models[0])
    rects2 = ax.bar(x + width/2, model_data[1], width, label=models[1])
    
    # Add labels and title
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()} Comparison by Model and Sampling Method')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ') for s in sampling_methods], rotation=45, ha='right')
    ax.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90)
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig(f"{output_dir}/{metric}_comparison.png", dpi=300, bbox_inches='tight')

# Create a summary table of all metrics
summary_data = []
for model in models:
    for sampling in sampling_methods:
        if model in all_metrics and sampling in all_metrics[model]:
            metrics = all_metrics[model][sampling]
            summary_data.append({
                "Model": model.replace('_', ' '),
                "Sampling": sampling.replace('_', ' '),
                "Accuracy": metrics.get("accuracy", 0),
                "Precision": metrics.get("precision", 0),
                "Recall": metrics.get("recall", 0),
                "Specificity": metrics.get("specificity", 0),
                "F1 Score": metrics.get("f1", 0)
            })

# Convert to DataFrame and save to CSV
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(f"{output_dir}/metrics_summary.csv", index=False)

print(f"Plots saved to {output_dir} directory")
print(f"Summary metrics saved to {output_dir}/metrics_summary.csv") 
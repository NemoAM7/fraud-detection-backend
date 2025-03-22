import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Define the sampling methods (limited to the 3 requested methods)
sampling_methods = [
    "Random_Undersampling", 
    "Random_Oversampling", 
    "SMOTE"
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

# Function to plot individual confusion matrix and save as separate file
def plot_individual_confusion_matrix(model, sampling_method):
    file_path = f"confusion_matrices/{model}_{sampling_method}_cm.csv"
    cm = read_confusion_matrix(file_path)
    
    if cm is None:
        print(f"Unable to plot {model} with {sampling_method} (missing data)")
        return None
    
    # Calculate metrics
    metrics = calculate_metrics(cm)
    
    # Create new figure for individual matrix
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    # Add metrics to title
    metrics_text = f"Accuracy: {metrics['accuracy']:.3f}\nPrecision: {metrics['precision']:.3f}\nRecall: {metrics['recall']:.3f}\nF1: {metrics['f1']:.3f}\nSpecificity: {metrics['specificity']:.3f}"
    title = f"{model.replace('_', ' ')} with {sampling_method.replace('_', ' ')}"
    plt.title(f"{title}")
    
    # Add metrics text in the bottom right
    plt.figtext(0.6, 0.01, metrics_text, ha="left", fontsize=10, 
                bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    # Set x and y tick labels
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])
    
    # Save figure
    output_file = f"{output_dir}/{model}_{sampling_method}_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {output_file}")
    return metrics

# Create a directory for the output plots
output_dir = "confusion_matrix_plots"
os.makedirs(output_dir, exist_ok=True)

# Store metrics for later comparison
all_metrics = {}

# Process each model and sampling method combination
for model in models:
    model_metrics = {}
    for sampling_method in sampling_methods:
        metrics = plot_individual_confusion_matrix(model, sampling_method)
        if metrics:
            model_metrics[sampling_method] = metrics
    all_metrics[model] = model_metrics

# Create performance comparison bar charts
metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
for metric in metrics_to_plot:
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
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, model_data[0], width, label=models[0])
    rects2 = ax.bar(x + width/2, model_data[1], width, label=models[1])
    
    # Add labels and title
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()} Comparison by Model and Sampling Method')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ') for s in sampling_methods])
    ax.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig(f"{output_dir}/{metric}_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

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

print(f"All plots saved to {output_dir} directory")
print(f"Summary metrics saved to {output_dir}/metrics_summary.csv") 
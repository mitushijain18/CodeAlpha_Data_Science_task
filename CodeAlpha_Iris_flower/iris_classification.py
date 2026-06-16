import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import Scikit-learn utilities for data splitting, training, and metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_iris_pipeline():
    print("==================================================")
    print("      CODEALPHA IRIS CLASSIFICATION ENGINE        ")
    print("==================================================")
    
    print("[SYSTEM] Loading Iris flower dataset repository...")
    raw_data = load_iris()
    
    # Structure features and target components into a unified Pandas framework
    df = pd.DataFrame(data=raw_data.data, columns=raw_data.feature_names)
    df['species_label'] = raw_data.target
    
    # Map raw integer indices to actual string class names
    target_names_map = {i: name for i, name in enumerate(raw_data.target_names)}
    df['species_name'] = df['species_label'].map(target_names_map)
    
    print(f"✔ Dataset structural rows parsed: {df.shape[0]} records loaded successfully.")
    print("✔ Class distributions identified: Setosa, Versicolor, Virginica.")
    
    # Separate input configurations from target prediction values
    X = df[raw_data.feature_names]
    y = df['species_label']
    
    # Split data structures into an 80/20 test split to evaluate predictive performance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"[SYSTEM] Splits mapped -> Training rows: {X_train.shape[0]}, Evaluation test rows: {X_test.shape[0]}")
    
    # Initialize and fit the machine learning model estimator
    print("[SYSTEM] Injecting data matrices into Random Forest Classifier core...")
    classifier_model = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier_model.fit(X_train, y_train)
    print("✔ Machine learning model compilation step complete.")
    
    # Execute performance predictions across test evaluation files
    y_predictions = classifier_model.predict(X_test)
    
    # Compute accuracy score performance markers
    system_accuracy = accuracy_score(y_test, y_predictions)
    
    print("\n==================================================")
    print("            MODEL EVALUATION REPORT               ")
    print("==================================================")
    print(f"Overall Model Accuracy Score: {system_accuracy:.4f} ({system_accuracy * 100:.1f}%)")
    
    print("\n--- Detailed Classification Matrix Report ---")
    print(classification_report(y_test, y_predictions, target_names=raw_data.target_names))
    
    # Generate and plot confusion matrix visualization to prove classification metrics
    print("[SYSTEM] Rendering metric confusion matrix plot file...")
    conf_matrix = confusion_matrix(y_test, y_predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=raw_data.target_names, yticklabels=raw_data.target_names)
    plt.title('Iris Classification Confusion Matrix Profile', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label Class')
    plt.ylabel('True Observation Class')
    plt.tight_layout()
    
    output_filename = "iris_model_confusion_matrix.png"
    plt.savefig(output_filename, dpi=150)
    print(f"✔ Performance metric visualization saved -> '{output_filename}'")
    print("==================================================")

if __name__ == "__main__":
    run_iris_pipeline()
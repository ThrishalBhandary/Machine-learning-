import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.datasets import make_classification

# --- 1. Helper Function for Model Training and Evaluation ---
def evaluate_model(majority_ratio):
    """Generates imbalanced data, trains a model, and returns metrics."""

    N_samples = 1000
    # Calculate the number of minority samples based on the ratio (e.g., 9:1 ratio means 100 minority, 900 majority)
    N_minority = int(N_samples / (1 + majority_ratio))
    N_majority = N_samples - N_minority

    # Define weights for make_classification to generate the desired imbalance
    weights = [N_majority / N_samples, N_minority / N_samples] # [Class 0 (Majority), Class 1 (Minority)]

    # Generate a synthetic dataset
    X, y = make_classification(
        n_samples=N_samples,
        n_features=5,
        n_redundant=0,
        n_informative=3,
        n_clusters_per_class=1,
        weights=weights,
        flip_y=0,
        random_state=42
    )

    # Split Data (stratify ensures both sets maintain the imbalance ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train a simple Logistic Regression model
    model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate Metrics (F1, Precision, Recall are calculated for the MINORITY class (1) by default)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, f1, precision, recall, cm, y_test.sum(), len(y_test) - y_test.sum()

# --- 2. Streamlit App Layout ---
def main():
    st.title('Imbalanced Data Evaluation: Why Accuracy Fails')

    st.markdown("""
    Use the slider to create a highly **imbalanced dataset** and observe how the **Accuracy** metric remains high even when the **F1-Score** reveals poor performance on the **Minority Class**.
    """)

    st.sidebar.header('Configuration')
    
    # Slider for user input
    majority_ratio = st.sidebar.slider(
        'Majority Class to Minority Class Ratio (Class 0 : Class 1)',
        min_value=1.0, # Balanced
        max_value=15.0, # Extreme Imbalance
        value=9.0,
        step=1.0,
        format='%.1f : 1'
    )
    
    # Execute the evaluation function
    accuracy, f1, precision, recall, cm, minority_count, majority_count = evaluate_model(majority_ratio)
    
    # --- Results Display ---
    st.header('Dataset and Metrics')

    st.markdown(f"""
    * **Imbalance Ratio:** **{majority_ratio:.1f} : 1**
    * **Test Samples:** 300
    * **Majority Class (0) Count:** {majority_count}
    * **Minority Class (1) Count:** {minority_count}
    """)
    
    st.subheader('Key Metric Comparison')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Overall Accuracy", value=f"{accuracy:.4f}")
        
    with col2:
        st.metric(label="F1-Score (Minority Class 1)", value=f"{f1:.4f}")
        
    with col3:
        st.metric(label="Recall (Sensitivity)", value=f"{recall:.4f}")
        
    with col4:
        st.metric(label="Precision", value=f"{precision:.4f}")
        
    st.markdown("---")

    # --- Confusion Matrix Visualization ---
    st.subheader('Confusion Matrix: Where the Errors Happen')
    
    cm_df = pd.DataFrame(
        cm,
        index=['Actual Class 0 (Majority)', 'Actual Class 1 (Minority)'],
        columns=['Predicted Class 0', 'Predicted Class 1']
    )
    st.table(cm_df)
    
    st.markdown("""
    #### Why Accuracy Fails:
    * **High Accuracy:** When the ratio is 9:1, a model that simply predicts **Class 0 (Majority)** all the time gets 90% accuracy!
    * **The Problem:** The model fails the **Minority Class**. This is seen by a high number in the **False Negatives (FN)** cell (Actual 1, Predicted 0).
    
    
    #### Why F1-Score Succeeds:
    * **F1-Score** is the harmonic mean of **Precision** and **Recall**.
    $$ F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $$
    * A high **FN** count leads to **low Recall**, and a low number of **True Positives (TP)** leads to low **Precision**. Both pull the F1-Score down, correctly indicating a poor model, even if accuracy is high.
    """)


if __name__ == '__main__':
    main()
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score
)

def calculate_metrics(model, X_test, y_test):
    preds = model.predict(X_test)
    # weighted handles class imbalance better
    return {
        'accuracy': accuracy_score(y_test, preds),
        'f1': f1_score(y_test, preds, average='weighted', zero_division=0),
        'precision': precision_score(y_test, preds, average='weighted', zero_division=0),
        'recall': recall_score(y_test, preds, average='weighted', zero_division=0)
    }
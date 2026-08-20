import os
import pandas as pd
from typing import Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from app.models.dataset import Dataset
from app.services.dataset_service import load_split_data
from app.ml.feature_importance import plot_shap_summary

def evaluate_model_pipeline() -> Dict[str, Any]:
    """
    Menjalankan pipeline pengujian model:
    1. Ambil data dari database.
    2. Split 80:20.
    3. GridSearchCV untuk mencari parameter terbaik.
    4. Evaluasi Akurasi dan Confusion Matrix.
    5. Hitung Feature Importance.
    """
    
    # 1. Ambil dataset dari DB dan split 80:20 (via dataset_service — single source of truth)
    try:
        X_train, X_test, y_train, y_test, X, y = load_split_data()
    except ValueError as exc:
        return {"success": False, "message": str(exc)}
    
    # 3. GridSearchCV
    base_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [7, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # 4. Evaluasi (Prediksi di Data Test)
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    classes = best_model.classes_.tolist()
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    report = classification_report(y_test, y_pred, output_dict=True, labels=classes)
    
    # Format Confusion Matrix untuk frontend (per kelas)
    formatted_cm = []
    for i, true_label in enumerate(classes):
        row_data = {"true_class": true_label, "predictions": {}}
        for j, pred_label in enumerate(classes):
            row_data["predictions"][pred_label] = int(cm[i][j])
        formatted_cm.append(row_data)
        
    # Format Metrik per kelas
    class_metrics = []
    for c in classes:
        if c in report:
            class_metrics.append({
                "class": c,
                "precision": round(report[c]["precision"] * 100, 2),
                "recall": round(report[c]["recall"] * 100, 2),
                "f1_score": round(report[c]["f1-score"] * 100, 2)
            })
        
    # 5. Generate SHAP Summary Plot
    # Convert X_test to DataFrame for SHAP
    X_test_df = pd.DataFrame(X_test, columns=Dataset.FEATURE_COLUMNS)
    output_dir = os.path.join(os.path.dirname(__file__), "..", "ml", "models")
    os.makedirs(output_dir, exist_ok=True)
    plot_shap_summary(best_model, X_test_df, Dataset.FEATURE_COLUMNS, output_dir)
    
    return {
        "success": True,
        "n_total": len(X),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "best_parameters": best_params,
        "accuracy": round(acc * 100, 2),
        "classes": classes,
        "confusion_matrix": formatted_cm,
        "class_metrics": class_metrics
    }

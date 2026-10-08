import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


def main():
    # 1. Locate and load dataset
    dataset_paths = [
        os.path.join(os.getcwd(), 'student+performance', 'student', 'student-mat.csv'),
        os.path.join(os.getcwd(), 'student-mat.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'student+performance', 'student', 'student-mat.csv')
    ]
    
    data_path = None
    for p in dataset_paths:
        if os.path.exists(p):
            data_path = os.path.abspath(p)
            break
            
    if not data_path:
        raise FileNotFoundError("Could not find student-mat.csv in expected locations.")

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path, sep=';')
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Separate features (X) and target (y)
    target_col = 'G3'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Automatically identify categorical and numeric features
    cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    num_cols = X.select_dtypes(include=['number']).columns.tolist()

    print(f"Identified {len(cat_cols)} categorical columns: {cat_cols}")
    print(f"Identified {len(num_cols)} numerical columns: {num_cols}")

    # 3. Build Preprocessor and Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('num', 'passthrough', num_cols)
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_depth=None
    )

    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ]
    )

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

    # 5. Train Model
    print("Training RandomForestRegressor pipeline...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # 6. Evaluate Model on Test Set
    y_test_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_test_pred)
    try:
        rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    except TypeError:
        # For newer scikit-learn root_mean_squared_error
        from sklearn.metrics import root_mean_squared_error
        rmse = root_mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    actual_mean_test = np.mean(y_test)
    pred_mean_test = np.mean(y_test_pred)
    pred_min_test = np.min(y_test_pred)
    pred_max_test = np.max(y_test_pred)

    print("\n" + "="*40)
    print("===== MODEL PERFORMANCE (TEST SET) =====")
    print("="*40)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
    print(f"\nActual Mean G3    : {actual_mean_test:.4f}")
    print(f"Predicted Mean G3 : {pred_mean_test:.4f}")
    print(f"Predicted Min     : {pred_min_test:.4f}")
    print(f"Predicted Max     : {pred_max_test:.4f}")
    print("="*40)

    # 7. Generate Predictions on the Entire Dataset (395 students)
    full_preds = pipeline.predict(X)
    full_preds_rounded = np.round(full_preds, 2)

    # Assign Sequential Student IDs
    student_ids = [f"STU{i+1:03d}" for i in range(len(df))]

    # Risk Classification Rules
    def assign_risk_level(pred_grade):
        if pred_grade < 10.0:
            return "High Risk"
        elif pred_grade <= 13.0:
            return "Medium Risk"
        else:
            return "Low Risk"

    risk_levels = [assign_risk_level(p) for p in full_preds_rounded]

    # Risk Score Calculation: 100 - 5 * predicted_grade
    risk_scores = np.round(100.0 - (5.0 * full_preds_rounded), 2)

    # Support Demand Hours Mapping
    support_demand_mapping = {
        "High Risk": 5,
        "Medium Risk": 2,
        "Low Risk": 0
    }
    support_demand_hours = [support_demand_mapping[lvl] for lvl in risk_levels]

    # Construct Predictions DataFrame (including all student features for downstream context)
    predictions_df = pd.DataFrame({
        'student_id': student_ids,
        'actual_grade': df['G3'],
        'predicted_grade': full_preds_rounded,
        'risk_level': risk_levels,
        'risk_score': risk_scores,
        'support_demand_hours': support_demand_hours
    })

    # Combine with original features
    for col in X.columns:
        predictions_df[col] = df[col]

    # 8. Save Model and Predictions
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    predictions_path = os.path.join('outputs', 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\nSaved predictions to: {predictions_path}")

    model_path = os.path.join('models', 'student_performance_rf.joblib')
    joblib.dump(pipeline, model_path)
    print(f"Saved trained pipeline to: {model_path}")

    # 9. Extract and Save Feature Importance
    rf_regressor = pipeline.named_steps['regressor']
    fitted_preprocessor = pipeline.named_steps['preprocessor']
    
    # Get feature names after one-hot encoding
    cat_encoder = fitted_preprocessor.named_transformers_['cat']
    encoded_cat_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    all_feature_names = encoded_cat_names + num_cols
    
    importances = rf_regressor.feature_importances_
    feat_imp_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)

    feat_imp_path = os.path.join('outputs', 'feature_importance.csv')
    feat_imp_df.to_csv(feat_imp_path, index=False)
    print(f"Saved feature importances to: {feat_imp_path}")

    # 10. Summary Statistics
    print("\n" + "="*40)
    print("===== SUMMARY STATISTICS =====")
    print("="*40)
    high_count = sum(1 for r in risk_levels if r == "High Risk")
    med_count = sum(1 for r in risk_levels if r == "Medium Risk")
    low_count = sum(1 for r in risk_levels if r == "Low Risk")
    
    print("### Risk Distribution:")
    print(f"High Risk    : {high_count} students ({high_count/len(df)*100:.2f}%)")
    print(f"Medium Risk  : {med_count} students ({med_count/len(df)*100:.2f}%)")
    print(f"Low Risk     : {low_count} students ({low_count/len(df)*100:.2f}%)")

    high_hours = high_count * 5
    med_hours = med_count * 2
    low_hours = low_count * 0
    total_hours = sum(support_demand_hours)

    print("\n### Support Demand:")
    print(f"High Risk Support Hours   : {high_hours} hours")
    print(f"Medium Risk Support Hours : {med_hours} hours")
    print(f"Low Risk Support Hours    : {low_hours} hours")
    print(f"Total Support Demand      : {total_hours} hours")

    print("\n### Full Dataset Prediction Stats:")
    print(f"Average Predicted Grade   : {np.mean(full_preds_rounded):.2f}")
    print(f"Minimum Predicted Grade   : {np.min(full_preds_rounded):.2f}")
    print(f"Maximum Predicted Grade   : {np.max(full_preds_rounded):.2f}")

    print("\n### Top 10 Important Features:")
    for idx, row in feat_imp_df.head(10).iterrows():
        print(f"{idx+1:2d}. {row['feature']:<25} : {row['importance']:.4f} ({row['importance']*100:.2f}%)")

    # 11. Automated Validation Checks
    print("\n" + "="*40)
    print("===== RUNNING VALIDATION CHECKS =====")
    print("="*40)
    checks_passed = True

    # 1. Dataset contains 395 students
    assert len(df) == 395, f"Check 1 Failed: Expected 395 rows, got {len(df)}"
    print("[PASS] Check 1: Dataset contains exactly 395 students.")

    # 2. Prediction count is 395
    assert len(predictions_df) == 395, f"Check 2 Failed: Expected 395 predictions, got {len(predictions_df)}"
    print("[PASS] Check 2: Prediction count is 395.")

    # 3. No missing predicted grades
    assert predictions_df['predicted_grade'].isnull().sum() == 0, "Check 3 Failed: Missing predicted grades."
    print("[PASS] Check 3: No missing predicted grades.")

    # 4. No missing risk levels
    assert predictions_df['risk_level'].isnull().sum() == 0, "Check 4 Failed: Missing risk levels."
    print("[PASS] Check 4: No missing risk levels.")

    # 5. Every risk level is valid
    valid_risk_levels = {"High Risk", "Medium Risk", "Low Risk"}
    actual_risk_levels = set(predictions_df['risk_level'].unique())
    assert actual_risk_levels.issubset(valid_risk_levels), f"Check 5 Failed: Invalid risk levels: {actual_risk_levels}"
    print("[PASS] Check 5: Every risk level is one of {High Risk, Medium Risk, Low Risk}.")

    # 6. Support demand values are only {0, 2, 5}
    valid_support_hours = {0, 2, 5}
    actual_support_hours = set(predictions_df['support_demand_hours'].unique())
    assert actual_support_hours.issubset(valid_support_hours), f"Check 6 Failed: Invalid support demand hours: {actual_support_hours}"
    print("[PASS] Check 6: Support demand values are strictly in {0, 2, 5}.")

    # 7. predicted_grade is numeric
    assert pd.api.types.is_numeric_dtype(predictions_df['predicted_grade']), "Check 7 Failed: predicted_grade is not numeric."
    print("[PASS] Check 7: predicted_grade is numeric.")

    # 8. Model file exists
    assert os.path.exists(model_path), f"Check 8 Failed: Model file {model_path} does not exist."
    print(f"[PASS] Check 8: Model file exists at {model_path}.")

    # 9. predictions.csv exists
    assert os.path.exists(predictions_path), f"Check 9 Failed: Predictions file {predictions_path} does not exist."
    print(f"[PASS] Check 9: predictions.csv exists at {predictions_path}.")

    # 10. feature_importance.csv exists
    assert os.path.exists(feat_imp_path), f"Check 10 Failed: Feature importance file {feat_imp_path} does not exist."
    print(f"[PASS] Check 10: feature_importance.csv exists at {feat_imp_path}.")

    print("="*40)
    print("ALL 10 VALIDATION CHECKS PASSED SUCCESSFULLY!")
    print("="*40)


if __name__ == "__main__":
    main()

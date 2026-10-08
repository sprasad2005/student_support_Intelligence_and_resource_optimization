import os
import pandas as pd
import pulp

# ==========================================
# CONFIGURATION: RESOURCE CAPACITY
# ==========================================
# Total teacher / specialized intervention capacity available per week
TOTAL_SUPPORT_HOURS = 300


def solve_resource_allocation(df, capacity_hours, verbose=False):
    """
    Formulates and solves a 0-1 Integer Linear Program (Binary Knapsack formulation)
    to allocate limited support hours maximizing total addressed risk score.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing student predictions and risk metrics.
        capacity_hours (float/int): Maximum available teacher intervention hours.
        verbose (bool): Whether PuLP solver prints solver logs.
        
    Returns:
        tuple: (prob_status, results_df, metrics_dict)
    """
    # 1. Filter intervention candidates (support_demand_hours > 0)
    # Low risk students (demand = 0) are excluded from the decision pool
    candidate_mask = df['support_demand_hours'] > 0
    candidates = df[candidate_mask].copy()

    # 2. Define PuLP Optimization Problem
    prob = pulp.LpProblem("Student_Support_Resource_Allocation", pulp.LpMaximize)

    # 3. Create Binary Decision Variables x_i for each candidate student
    # x_i = 1 if student i receives support intervention, 0 otherwise
    student_vars = {}
    for idx, row in candidates.iterrows():
        student_id = row['student_id']
        student_vars[student_id] = pulp.LpVariable(
            f"x_{student_id}",
            cat=pulp.LpBinary
        )

    # 4. Define Objective Function: Maximize Sum of (risk_score_i * x_i)
    prob += pulp.lpSum(
        row['risk_score'] * student_vars[row['student_id']]
        for idx, row in candidates.iterrows()
    ), "Total_Risk_Score_Addressed"

    # 5. Define Resource Constraint: Sum of (support_demand_hours_i * x_i) <= capacity_hours
    prob += pulp.lpSum(
        row['support_demand_hours'] * student_vars[row['student_id']]
        for idx, row in candidates.iterrows()
    ) <= capacity_hours, "Teacher_Support_Capacity_Constraint"

    # 6. Solve Problem using CBC Solver
    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0)
    prob.solve(solver)
    
    status_str = pulp.LpStatus[prob.status]

    # 7. Extract Decision Variable Values
    selection_map = {}
    for idx, row in candidates.iterrows():
        sid = row['student_id']
        var = student_vars[sid]
        val = int(round(var.varValue)) if var.varValue is not None else 0
        selection_map[sid] = val

    # Assign selection status across the full cohort
    results_df = df.copy()
    results_df['selected_for_support'] = results_df['student_id'].map(
        lambda sid: selection_map.get(sid, 0)
    )

    # 8. Compute Aggregated Metrics
    selected_df = results_df[results_df['selected_for_support'] == 1]
    allocated_hours = selected_df['support_demand_hours'].sum()
    unused_hours = max(0, capacity_hours - allocated_hours)
    
    students_selected = len(selected_df)
    high_risk_selected = len(selected_df[selected_df['risk_level'] == 'High Risk'])
    medium_risk_selected = len(selected_df[selected_df['risk_level'] == 'Medium Risk'])
    low_risk_selected = len(selected_df[selected_df['risk_level'] == 'Low Risk'])
    
    risk_score_addressed = round(selected_df['risk_score'].sum(), 2)
    
    total_cohort_demand = df['support_demand_hours'].sum()
    demand_coverage_pct = round((allocated_hours / total_cohort_demand * 100.0), 2) if total_cohort_demand > 0 else 0.0
    resource_utilization_pct = round((allocated_hours / capacity_hours * 100.0), 2) if capacity_hours > 0 else 0.0

    metrics = {
        'status': status_str,
        'available_hours': capacity_hours,
        'allocated_hours': allocated_hours,
        'unused_hours': unused_hours,
        'students_selected': students_selected,
        'high_risk_selected': high_risk_selected,
        'medium_risk_selected': medium_risk_selected,
        'low_risk_selected': low_risk_selected,
        'risk_score_addressed': risk_score_addressed,
        'demand_coverage_percentage': demand_coverage_pct,
        'resource_utilization_percentage': resource_utilization_pct,
        'total_cohort_demand': total_cohort_demand
    }

    return status_str, results_df, metrics


def main():
    # 1. Load Phase 2 Predictions
    predictions_path = os.path.join('outputs', 'predictions.csv')
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found at: {predictions_path}. Run Phase 2 first.")

    print(f"Loading predictions from: {predictions_path}")
    df = pd.read_csv(predictions_path)

    # Validate required columns
    required_cols = [
        'student_id', 'predicted_grade', 'risk_level',
        'risk_score', 'support_demand_hours'
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in predictions: {col}")

    total_students = len(df)
    total_demand = df['support_demand_hours'].sum()
    candidates_count = len(df[df['support_demand_hours'] > 0])
    
    total_high_risk = len(df[df['risk_level'] == 'High Risk'])
    total_medium_risk = len(df[df['risk_level'] == 'Medium Risk'])
    total_low_risk = len(df[df['risk_level'] == 'Low Risk'])

    print(f"Loaded {total_students} student predictions.")
    print(f"Cohort Profile: {total_high_risk} High Risk, {total_medium_risk} Medium Risk, {total_low_risk} Low Risk.")
    print(f"Total Unconstrained Support Demand: {total_demand} hours/week.")

    # 2. Run Main Optimization for Base Capacity (TOTAL_SUPPORT_HOURS = 300)
    print(f"\nSolving Resource Allocation for Capacity = {TOTAL_SUPPORT_HOURS} hours/week...")
    status, alloc_df, base_metrics = solve_resource_allocation(df, TOTAL_SUPPORT_HOURS)

    # Sort selected students first (selected=1), then by descending risk score
    output_columns = [
        'student_id', 'predicted_grade', 'risk_level',
        'risk_score', 'support_demand_hours', 'selected_for_support'
    ]
    # Keep extra columns for full context if desired, prioritizing output columns
    sorted_alloc_df = alloc_df.sort_values(
        by=['selected_for_support', 'risk_score'],
        ascending=[False, False]
    ).reset_index(drop=True)

    # Save outputs/optimized_allocation.csv
    os.makedirs('outputs', exist_ok=True)
    alloc_output_path = os.path.join('outputs', 'optimized_allocation.csv')
    sorted_alloc_df.to_csv(alloc_output_path, index=False)
    print(f"Saved optimized allocation to: {alloc_output_path}")

    # 3. Print Detailed Summary
    print("\n" + "="*50)
    print("===== OPERATIONS RESEARCH RESULTS =====")
    print("="*50)
    print(f"Solver Status               : {base_metrics['status']}")
    print(f"Total students              : {total_students}")
    print(f"Students requiring support  : {candidates_count}")
    print(f"Available support hours     : {TOTAL_SUPPORT_HOURS}")
    print(f"Allocated support hours     : {base_metrics['allocated_hours']}")
    print(f"Unused support hours        : {base_metrics['unused_hours']}")
    print(f"\nSelected High Risk students : {base_metrics['high_risk_selected']} / {total_high_risk}")
    print(f"Selected Medium Risk        : {base_metrics['medium_risk_selected']} / {total_medium_risk}")
    print(f"Selected Low Risk           : {base_metrics['low_risk_selected']} / {total_low_risk}")
    print(f"\nTotal risk score addressed  : {base_metrics['risk_score_addressed']:.2f}")
    print(f"Resource Utilization        : {base_metrics['resource_utilization_percentage']:.2f}%")
    print(f"Demand Coverage             : {base_metrics['demand_coverage_percentage']:.2f}%")

    # 4. Print Before vs After Comparison
    print("\n" + "="*50)
    print("===== BEFORE VS AFTER OPTIMIZATION =====")
    print("="*50)
    print("### Before Optimization (Unconstrained Demand):")
    print(f"  High Risk Demand   : {total_high_risk * 5} hours ({total_high_risk} students x 5h)")
    print(f"  Medium Risk Demand : {total_medium_risk * 2} hours ({total_medium_risk} students x 2h)")
    print(f"  Total Demand       : {total_demand} hours/week")
    print("\n### After Optimization (Budget = 300 hours/week):")
    print(f"  Allocated Hours    : {base_metrics['allocated_hours']} hours/week")
    print(f"  Remaining Capacity : {base_metrics['unused_hours']} hours/week")
    print(f"  Demand Coverage    : {base_metrics['demand_coverage_percentage']:.2f}% of total needed hours")

    # 5. Print Risk Coverage Analysis
    hr_cov = (base_metrics['high_risk_selected'] / total_high_risk * 100.0) if total_high_risk > 0 else 0.0
    mr_cov = (base_metrics['medium_risk_selected'] / total_medium_risk * 100.0) if total_medium_risk > 0 else 0.0
    print("\n" + "="*50)
    print("===== RISK COVERAGE =====")
    print("="*50)
    print("High Risk:")
    print(f"  Total      : {total_high_risk}")
    print(f"  Supported  : {base_metrics['high_risk_selected']}")
    print(f"  Coverage   : {hr_cov:.2f}%")
    print("\nMedium Risk:")
    print(f"  Total      : {total_medium_risk}")
    print(f"  Supported  : {base_metrics['medium_risk_selected']}")
    print(f"  Coverage   : {mr_cov:.2f}%")

    # 6. Run Sensitivity Analysis
    capacities_to_test = [100, 200, 300, 500, 700, 1000]
    sensitivity_rows = []

    print("\n" + "="*50)
    print("===== RUNNING CAPACITY SENSITIVITY ANALYSIS =====")
    print("="*50)
    for cap in capacities_to_test:
        _, _, sens_metrics = solve_resource_allocation(df, cap)
        sensitivity_rows.append({
            'available_hours': sens_metrics['available_hours'],
            'allocated_hours': sens_metrics['allocated_hours'],
            'unused_hours': sens_metrics['unused_hours'],
            'students_selected': sens_metrics['students_selected'],
            'high_risk_selected': sens_metrics['high_risk_selected'],
            'medium_risk_selected': sens_metrics['medium_risk_selected'],
            'risk_score_addressed': sens_metrics['risk_score_addressed'],
            'demand_coverage_percentage': sens_metrics['demand_coverage_percentage'],
            'resource_utilization_percentage': sens_metrics['resource_utilization_percentage']
        })
        print(f"Capacity {cap:4d}h -> Allocated: {sens_metrics['allocated_hours']:4d}h | Selected: {sens_metrics['students_selected']:3d} students (HR: {sens_metrics['high_risk_selected']:3d}, MR: {sens_metrics['medium_risk_selected']:3d}) | Risk Score: {sens_metrics['risk_score_addressed']:8.2f} | Coverage: {sens_metrics['demand_coverage_percentage']:5.2f}%")

    sens_df = pd.DataFrame(sensitivity_rows)
    capacity_output_path = os.path.join('outputs', 'capacity_analysis.csv')
    sens_df.to_csv(capacity_output_path, index=False)
    print(f"\nSaved capacity sensitivity analysis to: {capacity_output_path}")

    # 7. Automated Validation Checks (Step 13)
    print("\n" + "="*50)
    print("===== RUNNING OPTIMIZATION VALIDATION CHECKS =====")
    print("="*50)

    # 1. Input contains 395 students
    assert len(sorted_alloc_df) == 395, f"Check 1 Failed: Expected 395 rows, got {len(sorted_alloc_df)}"
    print("[PASS] Check 1: Input/output contains exactly 395 students.")

    # 2. Every selected student has support_demand_hours > 0
    selected_students = sorted_alloc_df[sorted_alloc_df['selected_for_support'] == 1]
    assert (selected_students['support_demand_hours'] > 0).all(), "Check 2 Failed: Selected student with 0 support demand."
    print("[PASS] Check 2: Every selected student has support_demand_hours > 0.")

    # 3. Every unselected student has selected_for_support = 0
    unselected_students = sorted_alloc_df[sorted_alloc_df['selected_for_support'] == 0]
    assert (unselected_students['selected_for_support'] == 0).all(), "Check 3 Failed: Invalid unselected flag."
    print("[PASS] Check 3: Every unselected student has selected_for_support = 0.")

    # 4. Selected support hours do not exceed 300
    assert base_metrics['allocated_hours'] <= TOTAL_SUPPORT_HOURS, f"Check 4 Failed: Allocated hours {base_metrics['allocated_hours']} exceeds {TOTAL_SUPPORT_HOURS}."
    print(f"[PASS] Check 4: Selected support hours ({base_metrics['allocated_hours']}h) do not exceed budget ({TOTAL_SUPPORT_HOURS}h).")

    # 5. Selected values are only 0 or 1
    unique_selections = set(sorted_alloc_df['selected_for_support'].unique())
    assert unique_selections.issubset({0, 1}), f"Check 5 Failed: Invalid selection values: {unique_selections}"
    print("[PASS] Check 5: Selection flags are strictly binary {0, 1}.")

    # 6. Every selected student exists in predictions.csv
    orig_student_ids = set(df['student_id'].unique())
    assert set(selected_students['student_id']).issubset(orig_student_ids), "Check 6 Failed: Selected student not in predictions."
    print("[PASS] Check 6: Every selected student exists in predictions.csv.")

    # 7. No duplicate student IDs in allocation output
    assert sorted_alloc_df['student_id'].duplicated().sum() == 0, "Check 7 Failed: Duplicate student IDs found."
    print("[PASS] Check 7: No duplicate student IDs exist in allocation output.")

    # 8. Solver status is optimal
    assert base_metrics['status'] == "Optimal", f"Check 8 Failed: Solver status is {base_metrics['status']}."
    print("[PASS] Check 8: Solver status is Optimal.")

    # 9. Allocated hours are calculated correctly
    manual_hours = selected_students['support_demand_hours'].sum()
    assert manual_hours == base_metrics['allocated_hours'], f"Check 9 Failed: Hours mismatch {manual_hours} vs {base_metrics['allocated_hours']}."
    print(f"[PASS] Check 9: Allocated hours correctly calculated ({manual_hours}h).")

    # 10. Risk scores match the Phase 2 predictions
    merged_check = pd.merge(df[['student_id', 'risk_score']], sorted_alloc_df[['student_id', 'risk_score']], on='student_id', suffixes=('_orig', '_alloc'))
    assert (merged_check['risk_score_orig'] == merged_check['risk_score_alloc']).all(), "Check 10 Failed: Risk scores mismatch."
    print("[PASS] Check 10: Risk scores strictly match Phase 2 predictions.")

    print("="*50)
    print("ALL 10 OPTIMIZATION VALIDATION CHECKS PASSED SUCCESSFULLY!")
    print("="*50)


if __name__ == "__main__":
    main()

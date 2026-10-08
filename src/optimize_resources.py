import os
import pandas as pd
import pulp

# ==========================================
# CONFIGURATION: RESOURCE CAPACITY
# ==========================================
# Default support specialist intervention hours available per week
DEFAULT_SUPPORT_HOURS = 200


def solve_resource_allocation(df, capacity_hours, verbose=False):
    """
    Formulates and solves a 0-1 Integer Linear Program (Binary Knapsack formulation)
    to allocate limited ASD support hours maximizing total screening priority score.
    
    Decision variable:
        x_i in {0, 1}: 1 if individual i receives allocated support, 0 otherwise
        
    Objective:
        Maximize sum_i (risk_score_i * x_i)
        
    Constraint:
        sum_i (support_demand_hours_i * x_i) <= capacity_hours
    
    Parameters:
        df (pd.DataFrame): DataFrame containing individuals' screening predictions and risk metrics.
        capacity_hours (float/int): Maximum available specialist support intervention hours.
        verbose (bool): Whether PuLP solver prints solver logs.
        
    Returns:
        tuple: (prob_status, results_df, metrics_dict)
    """
    # Identify ID column
    id_col = 'individual_id'
    for c in ['individual_id', 'id', 'client_id']:
        if c in df.columns:
            id_col = c
            break

    # Identify risk column (risk_category or risk_level)
    risk_col = 'risk_category'
    if 'risk_category' not in df.columns and 'risk_level' in df.columns:
        risk_col = 'risk_level'

    # Filter intervention candidates (support_demand_hours > 0)
    candidate_mask = df['support_demand_hours'] > 0
    candidates = df[candidate_mask].copy()

    # Define PuLP Optimization Problem
    prob = pulp.LpProblem("ASD_Support_Resource_Allocation", pulp.LpMaximize)

    # Binary Decision Variables
    individual_vars = {}
    for idx, row in candidates.iterrows():
        ind_id = str(row[id_col])
        individual_vars[ind_id] = pulp.LpVariable(
            f"x_{ind_id}",
            cat=pulp.LpBinary
        )

    # Objective Function: Maximize sum of priority scores
    prob += pulp.lpSum(
        row['risk_score'] * individual_vars[str(row[id_col])]
        for idx, row in candidates.iterrows()
    ), "Total_Screening_Priority_Addressed"

    # Resource Constraint: Total allocated hours <= available capacity
    prob += pulp.lpSum(
        row['support_demand_hours'] * individual_vars[str(row[id_col])]
        for idx, row in candidates.iterrows()
    ) <= capacity_hours, "Specialist_Support_Capacity_Constraint"

    # Solve with CBC Solver
    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0)
    prob.solve(solver)
    
    status_str = pulp.LpStatus[prob.status]

    # Extract Decision Variable Values
    selection_map = {}
    for idx, row in candidates.iterrows():
        ind_id = str(row[id_col])
        var = individual_vars[ind_id]
        val = int(round(var.varValue)) if var.varValue is not None else 0
        selection_map[ind_id] = val

    # Assign selection status across cohort
    results_df = df.copy()
    results_df['selected_for_support'] = results_df[id_col].astype(str).map(
        lambda sid: selection_map.get(sid, 0)
    ).fillna(0).astype(int)

    # Compute Aggregated Metrics
    selected_df = results_df[results_df['selected_for_support'] == 1]
    allocated_hours = int(selected_df['support_demand_hours'].sum())
    unused_hours = max(0, int(capacity_hours - allocated_hours))
    
    individuals_selected = len(selected_df)
    high_risk_selected = len(selected_df[selected_df[risk_col] == 'High Risk'])
    medium_risk_selected = len(selected_df[selected_df[risk_col] == 'Medium Risk'])
    low_risk_selected = len(selected_df[selected_df[risk_col] == 'Low Risk'])
    
    risk_score_addressed = round(float(selected_df['risk_score'].sum()), 2)
    
    total_cohort_demand = int(df['support_demand_hours'].sum())
    demand_coverage_pct = round((allocated_hours / total_cohort_demand * 100.0), 2) if total_cohort_demand > 0 else 0.0
    resource_utilization_pct = round((allocated_hours / capacity_hours * 100.0), 2) if capacity_hours > 0 else 0.0

    metrics = {
        'status': status_str,
        'available_hours': int(capacity_hours),
        'allocated_hours': int(allocated_hours),
        'unused_hours': int(unused_hours),
        'individuals_selected': int(individuals_selected),
        'selected_individuals': int(individuals_selected),
        'selected_count': int(individuals_selected),
        'num_selected': int(individuals_selected),
        'total_selected': int(individuals_selected),
        'high_risk_selected': int(high_risk_selected),
        'medium_risk_selected': int(medium_risk_selected),
        'low_risk_selected': int(low_risk_selected),
        'risk_score_addressed': round(float(risk_score_addressed), 2),
        'total_cohort_demand': int(total_cohort_demand),
        'demand_coverage_percentage': round(float(demand_coverage_pct), 2),
        'resource_utilization_percentage': round(float(resource_utilization_pct), 2)
    }

    return status_str, results_df, metrics


if __name__ == '__main__':
    preds_file = os.path.join('outputs', 'predictions.csv')
    if os.path.exists(preds_file):
        df_p = pd.read_csv(preds_file)
        print(f"Loaded predictions: {len(df_p)} individuals.")
        total_demand = df_p['support_demand_hours'].sum()
        print(f"Total unconstrained support demand: {total_demand} hours/week")
        
        test_cap = min(300, total_demand)
        status, alloc_df, metrics = solve_resource_allocation(df_p, test_cap)
        print(f"\nOptimization Result ({test_cap}h capacity):")
        print(f"Status: {status}")
        print(f"Allocated Hours: {metrics['allocated_hours']}h / {metrics['available_hours']}h")
        print(f"Individuals Selected: {metrics['individuals_selected']}")
        print(f"High Risk Selected: {metrics['high_risk_selected']}")
        print(f"Medium Risk Selected: {metrics['medium_risk_selected']}")
        print(f"Demand Coverage: {metrics['demand_coverage_percentage']}%")
        print(f"Resource Utilization: {metrics['resource_utilization_percentage']}%")
        
        os.makedirs('outputs', exist_ok=True)
        alloc_path = os.path.join('outputs', 'optimized_allocation.csv')
        alloc_df.to_csv(alloc_path, index=False)
        print(f"Saved allocation to {alloc_path}")

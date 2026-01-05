import numpy as np
import pandas as pd

def calculate_eoq(demand, ordering_cost, holding_cost):
    """
    Calculate Economic Order Quantity (EOQ).
    
    Args:
        demand (float): Annual demand units
        ordering_cost (float): Cost per order
        holding_cost (float): Holding cost per unit per year
        
    Returns:
        float: EOQ value
    """
    if holding_cost == 0:
        return 0
    return np.sqrt((2 * demand * ordering_cost) / holding_cost)

def calculate_rop(daily_demand, lead_time, safety_stock=0):
    """
    Calculate Reorder Point (ROP).
    
    Args:
        daily_demand (float): Daily demand units
        lead_time (float): Lead time in days
        safety_stock (float): Safety stock units (default 0)
        
    Returns:
        float: ROP value
    """
    return (daily_demand * lead_time) + safety_stock

def perform_abc_analysis(df, value_col='Annual_Value'):
    """
    Perform ABC Analysis based on Pareto Principle (80/15/5 rule).
    
    Args:
        df (pd.DataFrame): DataFrame containing inventory data
        value_col (str): Column name representing the consumption value (Price * Demand)
        
    Returns:
        pd.DataFrame: DataFrame with 'ABC_Category' and 'Cumulative_Percentage' columns
    """
    # Sort by value descending
    df_sorted = df.sort_values(by=value_col, ascending=False).copy()
    
    # Calculate cumulative sum and percentage
    total_value = df_sorted[value_col].sum()
    df_sorted['Cumulative_Value'] = df_sorted[value_col].cumsum()
    df_sorted['Cumulative_Percentage'] = (df_sorted['Cumulative_Value'] / total_value) * 100
    
    # Assign ABC Categories
    def assign_category(pct):
        if pct <= 80:
            return 'A'
        elif pct <= 95:
            return 'B'
        else:
            return 'C'
            
    df_sorted['ABC_Category'] = df_sorted['Cumulative_Percentage'].apply(assign_category)
    
    return df_sorted

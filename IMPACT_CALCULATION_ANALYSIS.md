# Complete Impact Calculation Analysis: Banking vs Non-Financial Dashboard

## Overview
This document provides a detailed comparison of how impact calculations work in the banking earnings drivers (`banking_earnings_drivers.py`) versus our implemented non-financial companies dashboard.

---

## 1. Banking Earnings Drivers - Complete Analysis (`banking_earnings_drivers.py`)

### Core Calculation Philosophy
The banking system uses a **normalization approach** where all scores sum to exactly ±100%, representing the complete decomposition of PBT changes.

### Step-by-Step Calculation Process

#### Step 1: Calculate Core Components
```python
# Core TOI (Total Operating Income) = NII + Fees
df['Core TOI'] = df['Net Interest Income'] + df['Fees Income']

# Core PBT = NII + Fees + OPEX + Provision (OPEX and Provision are negative)
df['Core PBT'] = df['Net Interest Income'] + df['Fees Income'] + 
                 df['OPEX'] + df['Provision expense']

# Non-recurring = Actual PBT - Core PBT
df['Non-recurring income'] = df['PBT'] - df['Core PBT']
```

#### Step 2: Calculate Period Changes
For T12M (Trailing 12 Months) comparison:
```python
# Calculate 4-quarter rolling average, shifted by 1
df['Metric_T12M'] = df.groupby('TICKER')[metric].transform(
    lambda x: x.rolling(window=4, min_periods=4).mean().shift(1)
)

# Calculate changes
df['PBT_Change'] = df['PBT'] - df['PBT_T12M']
df['Core_TOI_Change'] = df['Core TOI'] - df['Core TOI_T12M']
df['OPEX_Change'] = df['OPEX'] - df['OPEX_T12M']
df['Provision_Change'] = df['Provision expense'] - df['Provision expense_T12M']
```

#### Step 3: Handle Small PBT Changes (Critical!)
```python
# Threshold: 50 billion
small_pbt_threshold = 50_000_000_000

# Flag small changes
mask_small_pbt = df['PBT_Change'].abs() < small_pbt_threshold

# Adjust PBT for normalization
df['PBT_Change_Adjusted'] = df['PBT_Change']
df.loc[mask_small_pbt & (df['PBT_Change'] > 0), 'PBT_Change_Adjusted'] = small_pbt_threshold
df.loc[mask_small_pbt & (df['PBT_Change'] < 0), 'PBT_Change_Adjusted'] = -small_pbt_threshold

# Use absolute value for division
df['PBT_Change_Abs_Adjusted'] = df['PBT_Change_Adjusted'].abs()
```

#### Step 4: Calculate Base Scores (as % of Absolute PBT Change)
```python
# Main scores - each component as % of absolute PBT change
df['Top_Line_Score'] = (df['Raw_Top_Line'] / df['PBT_Change_Abs_Adjusted']) * 100
df['Cost_Cutting_Score'] = (df['Raw_Cost_Cutting'] / df['PBT_Change_Abs_Adjusted']) * 100
df['Non_Recurring_Score'] = (df['Raw_Non_Recurring'] / df['PBT_Change_Abs_Adjusted']) * 100

# Sub-scores
df['NII_Sub_Score'] = (df['NII_Change'] / df['PBT_Change_Abs_Adjusted']) * 100
df['Fee_Sub_Score'] = (df['Fee_Change'] / df['PBT_Change_Abs_Adjusted']) * 100
df['OPEX_Sub_Score'] = (df['OPEX_Change'] / df['PBT_Change_Abs_Adjusted']) * 100
df['Provision_Sub_Score'] = (df['Provision_Change'] / df['PBT_Change_Abs_Adjusted']) * 100
```

#### Step 5: Special NII Decomposition
```python
# Loan Growth Score = half of loan growth percentage
df['Loan_Growth_Score'] = df['Loan_Growth_%'] / 2

# NIM Score = NII Score minus Loan Growth contribution
df['NIM_Change_Score'] = df['NII_Sub_Score'] - df['Loan_Growth_Score']
```

#### Step 6: Ensure Scores Sum to ±100% (Scaling)
For small PBT changes, scores are scaled proportionally:
```python
if small_pbt_mask.any():
    # Target: +100 for positive PBT, -100 for negative
    df['Target_Total'] = np.where(df['PBT_Change'] > 0, 100, -100)
    
    # Calculate scaling factor
    df['Scale_Factor'] = df['Target_Total'] / df['Total_Score']
    
    # Apply scaling to all scores
    for col in score_columns:
        df[col] = df[col] * df['Scale_Factor']
```

#### Step 7: Calculate Weighted Impact Scores
```python
def calculate_weighted_impacts(df, suffix=''):
    # Calculate PBT Growth %
    df['PBT_Growth_%'] = (df['PBT_Change'] / df['PBT_T12M'].abs()) * 100
    
    # Impact = Score × |PBT_Growth_%| / 100
    for score in ['Top_Line_Score', 'Cost_Cutting_Score', 'Non_Recurring_Score']:
        impact_col = score.replace('_Score', '_Impact')
        df[impact_col] = (df[score] * df['PBT_Growth_%'].abs()) / 100
    
    # Special: Loan Impact is NOT weighted
    df['Loan_Impact'] = df['Loan_Growth_%'] / 2  # Direct, not weighted
    
    # NIM Impact = NII_Impact - Loan_Impact
    df['NIM_Impact'] = df['NII_Impact'] - df['Loan_Impact']
```

### Key Features of Banking Calculation

1. **Normalization to ±100%**: Scores ALWAYS sum to exactly 100% (positive PBT) or -100% (negative PBT)

2. **Small PBT Threshold**: Uses 50 billion threshold to prevent extreme scores

3. **Absolute Value Division**: Divides by absolute PBT change to maintain sign consistency

4. **Proportional Scaling**: Ensures mathematical consistency even with threshold

5. **Multiple Comparison Periods**: T12M (default), QoQ, YoY - all calculated and stored

6. **Score Capping**: Extreme scores capped at ±500% for readability

---

## 2. Mathematical Example - Banking Method

### Scenario: Bank ABC in Q1 2024

**Given:**
- PBT current quarter: 150 billion
- PBT T12M average: 120 billion
- PBT Change: 30 billion
- NII Change: 25 billion
- Fee Change: 10 billion
- OPEX Change: -3 billion (improvement)
- Provision Change: -2 billion (improvement)
- Non-recurring: 0 billion
- Loan Growth: 10%

**Calculations:**

```
Step 1: Raw Contributions
Raw_Top_Line = NII_Change + Fee_Change = 25 + 10 = 35
Raw_Cost_Cutting = OPEX_Change + Provision_Change = -3 + (-2) = -5
Raw_Non_Recurring = 0

Step 2: Check if Small PBT
30 billion < 50 billion threshold? No → Use actual PBT_Change

Step 3: Calculate Scores (% of absolute PBT change)
Top_Line_Score = 35 / 30 × 100 = 116.7%
Cost_Cutting_Score = -5 / 30 × 100 = -16.7%
Non_Recurring_Score = 0 / 30 × 100 = 0%
Total = 100% ✓

Step 4: Sub-scores
NII_Sub_Score = 25 / 30 × 100 = 83.3%
Fee_Sub_Score = 10 / 30 × 100 = 33.3%
OPEX_Sub_Score = -3 / 30 × 100 = -10%
Provision_Sub_Score = -2 / 30 × 100 = -6.7%

Step 5: NII Decomposition
Loan_Growth_Score = 10% / 2 = 5%
NIM_Change_Score = 83.3% - 5% = 78.3%

Step 6: PBT Growth %
PBT_Growth_% = 30 / 120 × 100 = 25%

Step 7: Weighted Impacts
Top_Line_Impact = 116.7 × 25 / 100 = 29.2
Cost_Cutting_Impact = -16.7 × 25 / 100 = -4.2
Loan_Impact = 10 / 2 = 5 (NOT weighted)
NIM_Impact = NII_Impact - Loan_Impact = 20.8 - 5 = 15.8
```

---

## 3. Key Differences: Banking vs Non-Financial Implementation

| Aspect | Banking (`banking_earnings_drivers.py`) | Non-Financial (`data_processor.py`) |
|--------|----------------------------------------|-------------------------------------|
| **Score Normalization** | Always sums to ±100% | No normalization requirement |
| **PBT Threshold** | 50 billion adjustment | No threshold |
| **Division Method** | Absolute PBT change | Direct PBT change |
| **Score Calculation** | `Component_Change / |PBT_Change| × 100` | Various formulas per metric |
| **Impact Formula** | `Score × |PBT_Growth_%| / 100` | `Score × PBT_Growth_% / 100` |
| **Special Cases** | Loan Impact = Loan_Growth/2 (unweighted) | All impacts weighted equally |
| **Score Capping** | ±500% with tracking | ±100% built into formulas |
| **Comparison Periods** | T12M, QoQ, YoY all calculated | YoY, QoQ calculated separately |
| **Data Structure** | Pre-processes all comparisons | Calculates on-demand |

---

## 4. Banking Calculation Flow Diagram

```
Raw Financial Data
    ↓
Calculate Core Components
    ├── Core TOI = NII + Fees
    ├── Core PBT = Core TOI + OPEX + Provisions
    └── Non-recurring = PBT - Core PBT
    ↓
Calculate Period Changes (T12M/QoQ/YoY)
    ↓
Check PBT Change Magnitude
    ├── If < 50B → Use 50B threshold
    └── If ≥ 50B → Use actual value
    ↓
Calculate Scores (% of |PBT Change|)
    ├── Top Line Score
    ├── Cost Cutting Score
    └── Non-Recurring Score
    ↓
Ensure Sum = ±100% (Scale if needed)
    ↓
Calculate Sub-scores
    ├── NII, Fee, OPEX, Provision
    └── Loan Growth, NIM Change
    ↓
Calculate PBT Growth %
    ↓
Calculate Weighted Impacts
    └── Impact = Score × |PBT_Growth_%| / 100
```

---

## 5. Why Banking Uses This Approach

### 1. **Complete Attribution**
- Every basis point of PBT change is attributed to a driver
- Scores sum to exactly 100%, showing complete decomposition

### 2. **Sign Consistency**
- Positive PBT change → Positive total (100%)
- Negative PBT change → Negative total (-100%)
- Components maintain intuitive signs

### 3. **Handling Edge Cases**
- Small PBT changes don't create infinite scores
- Threshold prevents noise amplification
- Scaling maintains mathematical integrity

### 4. **Industry-Specific Insights**
- Loan growth vs NIM change decomposition
- Provision expense tracking (credit quality)
- Fee income separate from interest income

---

## 6. Recommendations for Non-Financial Dashboard Alignment

To make our non-financial dashboard more similar to the banking approach:

### A. Adopt Normalization Approach
```python
def calculate_normalized_scores(df):
    # Calculate raw contributions
    df['Raw_Revenue'] = df['ΔNet_Revenue']
    df['Raw_Margin'] = df['ΔGross_Profit']
    df['Raw_Efficiency'] = df['ΔSG_A'] + df['ΔIE']
    df['Raw_OneOff'] = df['ΔOne_off']
    
    # Apply threshold for small PBT
    threshold = df['PBT'].median() * 0.1  # 10% of median
    df['PBT_Change_Adjusted'] = df['ΔPBT'].where(
        df['ΔPBT'].abs() >= threshold,
        threshold * np.sign(df['ΔPBT'])
    )
    
    # Calculate normalized scores
    df['Revenue_Score'] = (df['Raw_Revenue'] / df['PBT_Change_Adjusted'].abs()) * 100
    df['Margin_Score'] = (df['Raw_Margin'] / df['PBT_Change_Adjusted'].abs()) * 100
    # etc...
    
    # Ensure sum to ±100%
    df['Total_Score'] = df[score_columns].sum(axis=1)
    df['Scale_Factor'] = np.where(df['ΔPBT'] > 0, 100, -100) / df['Total_Score']
    for col in score_columns:
        df[col] *= df['Scale_Factor']
```

### B. Add Sub-Component Analysis
```python
# Break down Revenue into Volume vs Price
df['Volume_Score'] = df['Quantity_Change'] / df['PBT_Change_Abs'] * 100
df['Price_Score'] = df['Price_Change'] / df['PBT_Change_Abs'] * 100

# Break down Costs into Fixed vs Variable
df['Fixed_Cost_Score'] = df['Fixed_Cost_Change'] / df['PBT_Change_Abs'] * 100
df['Variable_Cost_Score'] = df['Variable_Cost_Change'] / df['PBT_Change_Abs'] * 100
```

### C. Implement Multiple Comparison Periods
```python
# Calculate all comparisons upfront
df_ttm = calculate_ttm_scores(df)
df_qoq = calculate_qoq_scores(df)
df_yoy = calculate_yoy_scores(df)

# Merge into single dataframe
df_merged = merge_all_scores(df, df_ttm, df_qoq, df_yoy)
```

---

## Summary

The banking earnings drivers system uses a sophisticated **normalization-based approach** that:
1. Ensures complete attribution (scores sum to ±100%)
2. Handles edge cases with thresholds and scaling
3. Maintains mathematical consistency
4. Provides detailed sub-component analysis
5. Pre-calculates all comparison periods

Our non-financial implementation uses a **simpler weighted scoring approach** that:
1. Calculates independent scores for each metric
2. Weights them to create a composite score
3. Calculates impacts on-demand
4. Offers more flexibility but less mathematical rigor

The banking approach is superior for **complete profit attribution analysis**, while our approach offers more **flexibility for general business metrics**.
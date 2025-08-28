# Banking to Non-Financial Companies Calculation Mapping

## Overview
This document provides a detailed mapping of every calculation used in the banking earnings drivers to the equivalent calculation for non-financial companies, using available data fields.

---

## Available Data Fields

### Banking Data Fields:
- Net Interest Income (NII)
- Fees Income
- OPEX (Operating Expenses)
- Provision expense
- PBT (Profit Before Tax)
- Loan
- NIM (Net Interest Margin)
- TOI (Total Operating Income)

### Non-Financial Data Fields:
- Net_Revenue
- COGS (Cost of Goods Sold)
- SG_A (Selling, General & Administrative)
- IE (Interest Expense)
- PBT (Profit Before Tax)
- EBIT (Earnings Before Interest & Tax)
- NPATMI (Net Profit After Tax Minority Interest)

---

## 1. Core Components Calculation

### Banking:
```python
# Core revenue components
Core TOI = Net Interest Income + Fees Income

# Core profitability
Core PBT = Net Interest Income + Fees Income + OPEX + Provision expense

# Non-recurring identification
Non-recurring income = PBT - Core PBT
```

### Non-Financial Equivalent:
```python
# Core revenue (single component)
Core Revenue = Net_Revenue

# Core operating costs (excluding interest)
Core Operating Costs = COGS + SG_A  # Both are negative

# Core operating income
Core Operating Income = Net_Revenue + COGS + SG_A

# Core PBT (including interest as separate component)
Core PBT = Net_Revenue + COGS + SG_A + IE

# Non-recurring identification
Non_Recurring = PBT - Core PBT

# Verification
PBT = Net_Revenue + COGS + SG_A + IE + Non_Recurring
```

---

## 2. Period-over-Period Changes

### Banking T12M (Trailing 12 Months):
```python
# 4-quarter rolling average, shifted by 1
Metric_T12M = df.groupby('TICKER')['Metric'].transform(
    lambda x: x.rolling(window=4, min_periods=4).mean().shift(1)
)

# Calculate change
Metric_Change = Metric_Current - Metric_T12M
```

### Non-Financial T12M Equivalent:
```python
# Identical calculation for non-financial
Metric_T12M = df.groupby('Ticker')['Metric'].transform(
    lambda x: x.rolling(window=4, min_periods=4).mean().shift(1)
)

# Calculate change
Metric_Change = Metric_Current - Metric_T12M
```

### Banking QoQ:
```python
Metric_QoQ = df.groupby('TICKER')['Metric'].shift(1)
Metric_Change = Metric_Current - Metric_QoQ
```

### Non-Financial QoQ Equivalent:
```python
# Identical
Metric_QoQ = df.groupby('Ticker')['Metric'].shift(1)
Metric_Change = Metric_Current - Metric_QoQ
```

### Banking YoY:
```python
Metric_YoY = df.groupby('TICKER')['Metric'].shift(4)  # 4 quarters back
Metric_Change = Metric_Current - Metric_YoY
```

### Non-Financial YoY Equivalent:
```python
# Identical
Metric_YoY = df.groupby('Ticker')['Metric'].shift(4)
Metric_Change = Metric_Current - Metric_YoY
```

---

## 3. Raw Contributions Calculation

### Banking:
```python
# Revenue contribution
Raw_Top_Line = Core_TOI_Change
# Where Core_TOI_Change = (NII + Fees)_current - (NII + Fees)_previous

# Cost efficiency contribution
Raw_Cost_Cutting = OPEX_Change + Provision_Change

# Non-recurring contribution
Raw_Non_Recurring = Non_Recurring_Change

# Verification
PBT_Change = Raw_Top_Line + Raw_Cost_Cutting + Raw_Non_Recurring
```

### Non-Financial Equivalent:
```python
# Revenue contribution
Raw_Revenue = Net_Revenue_Change

# Operating cost contribution (COGS + SG&A only)
Raw_Operating_Cost = COGS_Change + SGA_Change

# Interest expense contribution (separate)
Raw_Interest = IE_Change

# Non-recurring contribution
Raw_Non_Recurring = Non_Recurring_Change

# Verification
PBT_Change = Raw_Revenue + Raw_Operating_Cost + Raw_Interest + Raw_Non_Recurring
```

---

## 4. Small PBT Threshold Adjustment

### Banking:
```python
# Threshold
small_pbt_threshold = 50_000_000_000  # 50 billion

# Flag small changes
mask_small_pbt = abs(PBT_Change) < small_pbt_threshold

# Adjust for normalization
if mask_small_pbt:
    if PBT_Change > 0:
        PBT_Change_Adjusted = small_pbt_threshold
    else:
        PBT_Change_Adjusted = -small_pbt_threshold
else:
    PBT_Change_Adjusted = PBT_Change

# Use absolute value for division
PBT_Change_Abs_Adjusted = abs(PBT_Change_Adjusted)
```

### Non-Financial Equivalent:
```python
# IDENTICAL CALCULATION
small_pbt_threshold = 50_000_000_000  # Same 50 billion threshold

mask_small_pbt = abs(PBT_Change) < small_pbt_threshold

if mask_small_pbt:
    if PBT_Change > 0:
        PBT_Change_Adjusted = small_pbt_threshold
    else:
        PBT_Change_Adjusted = -small_pbt_threshold
else:
    PBT_Change_Adjusted = PBT_Change

PBT_Change_Abs_Adjusted = abs(PBT_Change_Adjusted)
```

---

## 5. Normalized Score Calculation (% of |PBT Change|)

### Banking Main Scores:
```python
# Three main components
Top_Line_Score = (Raw_Top_Line / PBT_Change_Abs_Adjusted) × 100
Cost_Cutting_Score = (Raw_Cost_Cutting / PBT_Change_Abs_Adjusted) × 100
Non_Recurring_Score = (Raw_Non_Recurring / PBT_Change_Abs_Adjusted) × 100

# Must sum to ±100%
Total_Score = Top_Line_Score + Cost_Cutting_Score + Non_Recurring_Score
```

### Non-Financial Main Scores:
```python
# Four main components (separating interest from operating costs)
Revenue_Score = (Raw_Revenue / PBT_Change_Abs_Adjusted) × 100
Operating_Cost_Score = (Raw_Operating_Cost / PBT_Change_Abs_Adjusted) × 100
Interest_Score = (Raw_Interest / PBT_Change_Abs_Adjusted) × 100
Non_Recurring_Score = (Raw_Non_Recurring / PBT_Change_Abs_Adjusted) × 100

# Must sum to ±100%
Total_Score = Revenue_Score + Operating_Cost_Score + Interest_Score + Non_Recurring_Score
```

---

## 6. Sub-Component Scores

### Banking Sub-Scores:
```python
# Revenue breakdown
NII_Sub_Score = (NII_Change / PBT_Change_Abs_Adjusted) × 100
Fee_Sub_Score = (Fee_Change / PBT_Change_Abs_Adjusted) × 100

# Cost breakdown
OPEX_Sub_Score = (OPEX_Change / PBT_Change_Abs_Adjusted) × 100
Provision_Sub_Score = (Provision_Change / PBT_Change_Abs_Adjusted) × 100
```

### Non-Financial Sub-Scores:
```python
# No revenue breakdown (single revenue line)
# Revenue_Score already calculated

# Operating cost breakdown
COGS_Sub_Score = (COGS_Change / PBT_Change_Abs_Adjusted) × 100
SGA_Sub_Score = (SGA_Change / PBT_Change_Abs_Adjusted) × 100

# Interest as separate component
# Interest_Score already calculated
```

---

## 7. Special Decompositions

### Banking NII Decomposition:
```python
# Loan volume contribution
Loan_Avg = (Loan_Current + Loan_Previous) / 2
Loan_Growth_% = ((Loan_Current - Loan_Previous) / Loan_Avg) × 100

# Simplified assumption: half of loan growth drives NII
Loan_Growth_Score = Loan_Growth_% / 2

# NIM contribution (residual)
NIM_Change_Score = NII_Sub_Score - Loan_Growth_Score
```

### Non-Financial Equivalent:
```python
# No direct equivalent - non-financial companies don't have loan portfolios
# Could potentially decompose revenue into:
# - Volume vs Price (if quantity data available)
# - Product mix vs same-store sales (if segment data available)
# For now: Revenue_Score stands alone without decomposition
```

---

## 8. Proportional Scaling (Ensure ±100% Total)

### Banking:
```python
if Small_PBT_Flag:
    # Target based on PBT change sign
    Target_Total = 100 if PBT_Change > 0 else -100
    
    # Calculate scaling factor
    Scale_Factor = Target_Total / Total_Score
    
    # Apply to all scores
    Top_Line_Score = Top_Line_Score × Scale_Factor
    Cost_Cutting_Score = Cost_Cutting_Score × Scale_Factor
    Non_Recurring_Score = Non_Recurring_Score × Scale_Factor
    
    # Apply to sub-scores
    NII_Sub_Score = NII_Sub_Score × Scale_Factor
    Fee_Sub_Score = Fee_Sub_Score × Scale_Factor
    OPEX_Sub_Score = OPEX_Sub_Score × Scale_Factor
    Provision_Sub_Score = Provision_Sub_Score × Scale_Factor
    Loan_Growth_Score = Loan_Growth_Score × Scale_Factor
    NIM_Change_Score = NIM_Change_Score × Scale_Factor
    
    # Recalculate total
    Total_Score = Top_Line_Score + Cost_Cutting_Score + Non_Recurring_Score
```

### Non-Financial Equivalent:
```python
if Small_PBT_Flag:
    # IDENTICAL LOGIC with 4 components
    Target_Total = 100 if PBT_Change > 0 else -100
    
    Scale_Factor = Target_Total / Total_Score
    
    # Apply to main scores
    Revenue_Score = Revenue_Score × Scale_Factor
    Operating_Cost_Score = Operating_Cost_Score × Scale_Factor
    Interest_Score = Interest_Score × Scale_Factor
    Non_Recurring_Score = Non_Recurring_Score × Scale_Factor
    
    # Apply to sub-scores
    COGS_Sub_Score = COGS_Sub_Score × Scale_Factor
    SGA_Sub_Score = SGA_Sub_Score × Scale_Factor
    
    # Recalculate total
    Total_Score = Revenue_Score + Operating_Cost_Score + Interest_Score + Non_Recurring_Score
```

---

## 9. Score Capping

### Banking:
```python
# Cap extreme scores
score_columns = ['Top_Line_Score', 'Cost_Cutting_Score', 'Non_Recurring_Score',
                'NII_Sub_Score', 'Fee_Sub_Score', 'OPEX_Sub_Score', 'Provision_Sub_Score',
                'Loan_Growth_Score', 'NIM_Change_Score']

Scores_Capped = False
for col in score_columns:
    if abs(score_value) > 500:
        Scores_Capped = True
        score_value = 500 if score_value > 0 else -500
```

### Non-Financial Equivalent:
```python
# IDENTICAL LOGIC with different columns
score_columns = ['Revenue_Score', 'Operating_Cost_Score', 'Interest_Score', 'Non_Recurring_Score',
                'COGS_Sub_Score', 'SGA_Sub_Score']

Scores_Capped = False
for col in score_columns:
    if abs(score_value) > 500:
        Scores_Capped = True
        score_value = 500 if score_value > 0 else -500
```

---

## 10. Growth Rate Calculation

### Banking:
```python
# PBT Growth %
PBT_Growth_% = (PBT_Change / abs(PBT_Previous)) × 100
```

### Non-Financial:
```python
# Use NPATMI Growth as primary metric
NPATMI_Growth_% = (NPATMI_Change / abs(NPATMI_Previous)) × 100

# Also calculate PBT Growth for reference
PBT_Growth_% = (PBT_Change / abs(PBT_Previous)) × 100
```

---

## 11. Weighted Impact Calculation

### Banking:
```python
# Impact = Score × |PBT_Growth_%| / 100
Top_Line_Impact = Top_Line_Score × abs(PBT_Growth_%) / 100
Cost_Cutting_Impact = Cost_Cutting_Score × abs(PBT_Growth_%) / 100
Non_Recurring_Impact = Non_Recurring_Score × abs(PBT_Growth_%) / 100

# Sub-impacts
NII_Impact = NII_Sub_Score × abs(PBT_Growth_%) / 100
Fee_Impact = Fee_Sub_Score × abs(PBT_Growth_%) / 100
OPEX_Impact = OPEX_Sub_Score × abs(PBT_Growth_%) / 100
Provision_Impact = Provision_Sub_Score × abs(PBT_Growth_%) / 100

# Special: Loan impact NOT weighted
Loan_Impact = Loan_Growth_% / 2  # Direct
NIM_Impact = NII_Impact - Loan_Impact

# Total Impact should equal |PBT_Growth_%|
Total_Impact = Top_Line_Impact + Cost_Cutting_Impact + Non_Recurring_Impact
```

### Non-Financial Equivalent:
```python
# Impact = Score × |NPATMI_Growth_%| / 100 (using NPATMI instead of PBT)
Revenue_Impact = Revenue_Score × abs(NPATMI_Growth_%) / 100
Operating_Cost_Impact = Operating_Cost_Score × abs(NPATMI_Growth_%) / 100
Interest_Impact = Interest_Score × abs(NPATMI_Growth_%) / 100
Non_Recurring_Impact = Non_Recurring_Score × abs(NPATMI_Growth_%) / 100

# Sub-impacts
COGS_Impact = COGS_Sub_Score × abs(NPATMI_Growth_%) / 100
SGA_Impact = SGA_Sub_Score × abs(NPATMI_Growth_%) / 100

# Total Impact should equal |NPATMI_Growth_%|
Total_Impact = Revenue_Impact + Operating_Cost_Impact + Interest_Impact + Non_Recurring_Impact
```

---

## 12. Summary Table of Key Differences

| Aspect | Banking | Non-Financial |
|--------|---------|---------------|
| **Main Components** | 3 (Top Line, Cost Cutting, Non-Recurring) | 4 (Revenue, Operating Costs, Interest, Non-Recurring) |
| **Revenue Components** | NII + Fees | Net Revenue (single line) |
| **Cost Components** | OPEX + Provisions | COGS + SG&A (Operating) + IE (separate) |
| **Primary Growth Metric** | PBT Growth % | NPATMI Growth % |
| **Special Decomposition** | Loan Growth vs NIM | None (no equivalent) |
| **Threshold** | 50 billion | 50 billion (same) |
| **Score Cap** | ±500% | ±500% (same) |
| **Total Score** | Always ±100% | Always ±100% |

---

## Implementation Notes

### Key Principles to Maintain:
1. **Complete Attribution**: Scores MUST sum to exactly ±100%
2. **Sign Convention**: Positive changes → +100%, Negative changes → -100%
3. **Threshold Protection**: Use 50B threshold to prevent noise amplification
4. **Mathematical Consistency**: All calculations must satisfy accounting identities

### Verification Checks:
```python
# 1. Raw contributions sum to PBT change
assert abs(Raw_Revenue + Raw_Operating_Cost + Raw_Interest + Raw_Non_Recurring - PBT_Change) < 0.01

# 2. Scores sum to ±100% (after scaling)
assert abs(abs(Total_Score) - 100) < 0.1

# 3. Sub-scores match main scores
assert abs(COGS_Sub_Score + SGA_Sub_Score - Operating_Cost_Score) < 0.1

# 4. Total impact equals growth rate
assert abs(Total_Impact - abs(NPATMI_Growth_%)) < 0.1
```

### Data Requirements:
- **Minimum**: 4 quarters of historical data for T12M calculations
- **Required Fields**: Net_Revenue, COGS, SG_A, IE, PBT, NPATMI
- **Period Identifiers**: Ticker, Year, Quarter

This mapping ensures complete replication of the banking methodology while adapting to the different account structure of non-financial companies.
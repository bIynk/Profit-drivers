## Coding Guidelines for Claude

**IMPORTANT**: This repository contains both banking and non-financial implementations. The banking implementation (banking_earnings_drivers.py, banking_example.py) serves only as a reference and inspiration. The actual application uses the non-financial implementation (data_processor.py, profit_drivers_dashboard.py).

When writing code for this repository:

1. Jupyter/Interactive Style:
   - Use `#%%` cell markers for code organization
   - Assume pandas, numpy, and plotly are already imported
   - Write code that can be run cell-by-cell in Jupyter

2. Calculation Focus:
   - Prioritize mathematical correctness and clarity
   - Use vectorized pandas operations
   - Don't add excessive try/except blocks
   - Assume data exists and is in expected format

3. Data Analysis Patterns:
   ```python
   # Good - direct calculation
   df['metric'] = df['revenue'] / df['assets']
   
   # Avoid - over-engineered
   def calculate_metric(df):
       if 'revenue' not in df.columns:
           raise ValueError("Missing revenue column")
       # ... more checks
   ```

4. Variable Naming:
   - Use descriptive names for financial metrics
   - Keep DataFrame names short (df_q, df_a, etc.)
   - Use standard financial abbreviations (ROE, ROA, NPAT)

5. Output Style:
   - Display DataFrames directly without wrapping
   - Use simple print statements for quick checks
   - Format numbers inline with f-strings when needed

## Dashboard Development Guidelines

### Current Implementation Status (Dec 2024)

1. **Table Display**:
   - Uses Streamlit's native dataframe display
   - Basic sorting available through Streamlit's built-in features
   - No custom HTML/CSS implementation currently

2. **Visual Structure**:
   - Main components: Gross Margin, SG&A, Interest, Non-Recurring
   - Sub-components: Revenue and COGS (under Gross Margin)
   - Standard Streamlit styling applied

3. **Data Display**:
   - Shows raw contributions, scores, and impacts directly in tables
   - All monetary values in billions VND
   - Percentages with 1 decimal place

4. **Color Coding**:
   ```css
   /* Positive values */
   .positive-value { color: #28a745; font-weight: bold; }
   
   /* Negative values */
   .negative-value { color: #dc3545; font-weight: bold; }
   
   /* Main components */
   .main-component { background-color: #1976d2; }
   
   /* Sub-components */
   .sub-component { background-color: #64b5f6; }
   ```

5. **Non-Financial Methodology**:
   - Scores MUST sum to ±100% (complete attribution)
   - Use 50B VND threshold for small PBT changes
   - No score capping in non-financial implementation
   - Four main components: Gross Margin, SG&A, Interest, Non-Recurring

6. **Display Conventions**:
   - All monetary values in billions VND
   - Percentages with 1 decimal place
   - NPATMI growth as primary metric (not PBT growth)
   - Raw contributions, scores, and impacts all displayed in tables

## Version History & Changes

### [3.0.0] - 2024-12-28 - Gross Margin Reorganization

#### PBT Formula Update
The PBT formula has been reorganized to use Gross Margin as a primary component:

**Previous Structure (4 Components):**
```
PBT = Net_Revenue + COGS + SG_A + IE + Non_Recurring
```
- Revenue (Net_Revenue) - positive
- Operating Costs (COGS + SG_A) - negative
- Interest (IE) - negative
- Non-Recurring - calculated residual

**New Structure (4 Components):**
```
PBT = Gross_Margin + SG_A + IE + Non_Recurring
where Gross_Margin = Net_Revenue + COGS
```
- **Gross Margin** (Net_Revenue + COGS) - typically positive
- **SG&A** (SG_A) - negative, now standalone
- **Interest** (IE) - negative
- **Non-Recurring** - calculated residual

#### Key Implementation Changes
1. **Data Processing Updates:**
   - Added Gross_Margin calculation
   - Raw_Gross_Margin and Raw_SGA replace previous components
   - Revenue and COGS are now sub-components of Gross Margin
   - Gross_Margin_Score and SGA_Score replace previous scores

2. **Dashboard Display:**
   - Main components show Gross Margin, SG&A, Interest, Non-Recurring
   - Revenue and COGS shown as sub-components under Gross Margin
   - Updated tooltips and column headers

3. **Score Calculation:**
   - No score capping in non-financial implementation
   - Ensures mathematical accuracy for extreme values
   - Complete attribution (±100%) maintained through proportional scaling
   - Note: Banking reference implementation still caps at ±500%

### [2.0.0] - 2024-12-27 - Planned Features (Not Yet Implemented)

#### Planned but Not Implemented:
- Interactive sorting with dropdown controls
- Enhanced visual hierarchy with tree symbols
- CSS-based hover tooltips
- HTML/CSS table customization

#### Current State:
- Uses standard Streamlit dataframe display
- Basic sorting through Streamlit's built-in features
- All data displayed directly in tables (no tooltips)

### [1.0.0] - 2024-12-25 - Initial Release

#### Core Features
- **Non-Financial Company Analysis:**
  - Complete profit attribution (±100%)
  - Four-component structure (adapted from banking methodology)
  - 50B VND threshold for small PBT changes
  - Three comparison periods (YoY, QoQ, T12M)
- **Sector Analysis:** Automatic exclusion of financial sectors
- **Company Analysis:** Waterfall charts and trend analysis
- **Data Processing:** Automatic period calculations and non-recurring detection

## Implementation Notes

### File Structure
- **Core Application Files:**
  - `data_processor.py`: Main calculation engine for non-financial companies
  - `profit_drivers_dashboard.py`: Streamlit dashboard interface
  - `data.csv`, `sector_map.pkl`: Data files
  
- **Reference Files (Banking - for inspiration only):**
  - `banking_earnings_drivers.py`: Banking methodology reference
  - `banking_example.py`: Banking dashboard example
  
- **Test Files:**
  - `test_gross_margin.py`: Verifies Gross Margin calculations
  - `test_impact_calculation.py`: Validates impact calculations
  - Other test files for various components

### Key Differences from Banking Implementation
1. **No Score Capping:** Non-financial implementation doesn't cap scores
2. **Component Structure:** Uses Gross Margin as primary component (Revenue + COGS)
3. **Dashboard:** Simpler Streamlit implementation without custom HTML/CSS
4. **Sector Focus:** Specifically excludes financial sectors
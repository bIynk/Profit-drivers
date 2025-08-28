# Profit Drivers Dashboard for Non-Financial Companies

## Overview
This dashboard analyzes profit drivers for non-financial companies using banking earnings driver methodology. It provides complete profit attribution where all component scores sum to exactly ±100%, ensuring comprehensive accounting of profit changes.

## Recent Updates (December 2024)

### 1. Enhanced Interactive Table
- **Sortable Columns**: Sort data by any metric (PBT Growth %, Gross Margin, SG&A, Interest, Non-Recurring)
- **Sort Controls**: Dropdown selector with ascending/descending toggle
- **Visual Sort Indicators**: ▼ for descending, ▲ for ascending in column headers

### 2. Visual Hierarchy Improvements
- **Two-tier Headers**: Clear distinction between main components and sub-components
- **Color Coding**:
  - Main components (Gross Margin, SG&A, Interest, Non-Recurring): Dark blue (#1976d2)
  - Sub-components (Revenue Growth, Margin Expansion): Light blue (#64b5f6) with italic styling
  - Metrics (PBT Growth %): Navy blue (#0d47a1)
- **Tree Structure**: Sub-components use ├ and └ symbols to show hierarchy
- **Background Shading**: Sub-component cells have subtle background tint

### 3. Hover Tooltips
- **Always Enabled**: Enhanced table with tooltips is now shown by default (removed toggle)
- **Comprehensive Information**: Each cell shows:
  - Raw contribution amount (billions VND)
  - Score (% of PBT change)
  - Impact (Score × PBT Growth %)
- **Visual Feedback**: Hover effects with smooth transitions

### 4. Simplified Display
- **Primary Metric**: PBT Growth % as the main performance indicator
- **Raw Values in Billions VND**: All monetary values displayed in billions for clarity
- **Score Breakdown**: Complete scores available in expandable section

## Key Features

### Banking Methodology Implementation
- **Complete Attribution**: Scores always sum to ±100%
- **Four Main Components**:
  1. Revenue
  2. Operating Costs (COGS + SG&A)
  3. Interest Expense (separated from operating costs)
  4. Non-Recurring Items
- **Small PBT Threshold**: 50 billion VND threshold prevents extreme scores
- **Score Capping**: ±500% limit for individual components

### Period Comparisons
- **YoY**: Year-over-Year (4 quarters back)
- **QoQ**: Quarter-over-Quarter (1 quarter back)
- **T12M**: Trailing 12 Months (4-quarter rolling average)

## File Structure

```
profit-drivers/
├── .streamlit/
│   └── config.toml                  # Streamlit configuration (light theme)
├── profit_drivers_dashboard.py       # Main Streamlit dashboard
├── data_processor.py                 # Core calculation engine  
├── banking_earnings_drivers.py       # Banking methodology reference
├── banking_example.py                # Banking implementation example
├── data.csv                         # Financial data
├── sector_map.pkl                   # Sector mappings
├── BANKING_TO_NONFINANCIAL_MAPPING.md # Detailed methodology
├── IMPACT_CALCULATION_ANALYSIS.md    # Impact calculation details
├── CLAUDE.md                        # Development guidelines
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
└── .gitignore                       # Version control exclusions
```

## Installation

### Requirements
- Python 3.8 or higher
- pip package manager

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Verify Installation
```bash
python test_installation.py
```

This will check that all required packages and files are properly installed.

## Usage

### Running the Dashboard
```bash
streamlit run profit_drivers_dashboard.py
```

The dashboard will open in your default browser at `http://localhost:8501`

### Configuration
The app is configured to use a light theme by default. Settings can be modified in `.streamlit/config.toml`

### Dashboard Navigation

#### 1. Sidebar Controls
- **Period Selection**: Choose specific quarter (e.g., 2024Q3)
- **Sector Filter**: Filter by sector or view all
- **Comparison Period**: Select YoY, QoQ, or T12M
- **Company Selection**: For detailed analysis tabs

#### 2. Main Table Features
- **Sorting**: Use dropdown to sort by any metric
- **Search**: Filter companies by ticker
- **Hover Information**: Hover over values to see scores and impacts
- **Score Breakdown**: Expand "View Complete Score Breakdown" for all scores

#### 3. Tab Views
- **Raw Contributions**: Main table with sortable columns and tooltips
- **Company Analysis**: Detailed single company view with waterfall chart
- **Sector Comparison**: Compare performance across sectors
- **Trend Analysis**: Track metrics over time for selected companies

## Data Requirements

### Required Fields (data.csv)
- **Identifiers**: Ticker, Year, Quarter
- **Financial Metrics**:
  - Net_Revenue
  - COGS (Cost of Goods Sold)
  - SG_A (Selling, General & Administrative)
  - Interest_Expense (IE)
  - PBT (Profit Before Tax)
  - NPATMI (Net Profit After Tax Minority Interest)
  - EBIT (Earnings Before Interest & Tax)

### Sector Mapping (sector_map.pkl)
- Maps tickers to sectors
- Excludes financial sectors (Banking, Brokerage, Insurance)

## Calculation Methodology

### Score Calculation
```python
Score = (Component_Change / |PBT_Change_Adjusted|) × 100
```

### Impact Calculation
```python
Impact = Score × |PBT_Growth_%| / 100
```

### Verification
```python
Revenue_Δ + Operating_Cost_Δ + Interest_Δ + Non_Recurring_Δ = PBT_Change
```

## Visual Design

### Color Scheme
- **Positive values**: Green (#28a745)
- **Negative values**: Red (#dc3545)
- **Headers**: Blue gradient (navy to light blue)
- **Sub-components**: Light background tint

### Responsive Design
- Table adjusts to container width
- Optimized column widths for readability
- Mobile-friendly sorting controls

## Technical Stack
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, HTML/CSS
- **Methodology**: Banking earnings driver approach adapted for non-financial companies

## Support
For questions or issues, refer to:
- `BANKING_TO_NONFINANCIAL_MAPPING.md` for detailed methodology
- `IMPACT_CALCULATION_ANALYSIS.md` for calculation specifics
- `CLAUDE.md` for coding guidelines

## Version History
- **v3.1** (Dec 2024): Changed primary display metric to PBT Growth %, simplified Summary Statistics
- **v3.0** (Dec 2024): Gross Margin reorganization with Revenue Growth and Margin Expansion decomposition
- **v2.5** (Dec 2024): Repository cleanup, configuration management, optimized code structure
- **v2.0** (Dec 2024): Interactive sorting, enhanced visual hierarchy, always-on tooltips
- **v1.5**: Added hover tooltips with impact scores
- **v1.0**: Initial implementation with banking methodology

## Recent Changes

### v3.1 - PBT Growth Focus (Dec 2024)
- **Display Changes**:
  - Changed primary metric from NPATMI Growth % to PBT Growth %
  - PBT Growth now shown in main table and all visualizations
  - Updated histogram and trend analysis to use PBT Growth
- **Summary Statistics**:
  - Removed Average and Median growth metrics
  - Now shows only count and percentage of positive/negative growth companies
  - Cleaner, more focused summary section
- **Documentation**: Updated all references to reflect PBT Growth as primary metric

### v2.5 - Repository Optimization (Dec 2024)
- Removed test files and old dashboard versions
- Consolidated CSS styling
- Added configuration management
- Created .gitignore for proper version control
- Optimized code performance
- Preserved banking reference implementations
import pandas as pd
import numpy as np
import pickle
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class Config:
    """Configuration constants for the profit driver processor"""
    
    # File paths
    DEFAULT_DATA_FILE = 'data.csv'
    DEFAULT_SECTOR_FILE = 'sector_map.pkl'
    
    # Thresholds
    SMALL_PBT_THRESHOLD = 50_000_000_000  # 50 billion VND
    
    # Sectors to exclude (financial sectors)
    FINANCIAL_SECTORS = ['Banking', 'Bank', 'Brokerage', 'Insurance']
    
    # Default settings
    DEFAULT_COMPARISON = 'T12M'
    
    # Display settings
    BILLION_DIVISOR = 1_000_000_000
    PERCENTAGE_MULTIPLIER = 100
    
    # Score limits (currently not capped in non-financial implementation)
    MAX_SCORE_CAP = None  # No capping applied
    MIN_SCORE_CAP = None  # No capping applied


class ProfitDriverProcessor:
    """
    Process financial data for non-financial companies using exact banking methodology
    Based on BANKING_TO_NONFINANCIAL_MAPPING.md specifications
    
    This processor implements complete profit attribution where all component scores
    sum to exactly ±100%, ensuring comprehensive accounting of profit changes.
    """
    
    def __init__(self, data_file: str = None, sector_file: str = None):
        """
        Initialize the processor with data and sector files.
        
        Args:
            data_file: Path to CSV file with financial data
            sector_file: Path to pickle file with sector mappings
        """
        self.data_file = data_file or Config.DEFAULT_DATA_FILE
        self.sector_file = sector_file or Config.DEFAULT_SECTOR_FILE
        self.financial_sectors = Config.FINANCIAL_SECTORS
        self.small_pbt_threshold = Config.SMALL_PBT_THRESHOLD
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and process raw data"""
        # Load raw data
        data = pd.read_csv(self.data_file, header=None, 
                          names=['Ticker', 'Year', 'Quarter', 'Keycode', 'Value'])
        
        # Load sector mapping
        with open(self.sector_file, 'rb') as f:
            sector_map = pickle.load(f)
        
        # Pivot data to wide format
        pivot_data = data.pivot(index=['Ticker', 'Year', 'Quarter'], 
                               columns=['Keycode'], 
                               values=['Value']).reset_index()
        pivot_data.columns = ['Ticker', 'Year', 'Quarter', 'COGS', 'EBIT', 
                              'Interest_Expense', 'Interest_expense', 
                              'NPATMI', 'Net_Revenue', 'PBT', 'SG_A']
        
        # Combine interest expense columns (take max)
        pivot_data['IE'] = pivot_data[['Interest_Expense', 'Interest_expense']].max(axis=1)
        pivot_data = pivot_data.drop(['Interest_Expense', 'Interest_expense'], axis=1)
        
        # Merge with sector mapping
        pivot_data = pivot_data.merge(sector_map, on='Ticker', how='left')
        
        # Filter out financial sectors
        non_financial = pivot_data[~pivot_data['Sector'].isin(self.financial_sectors)].copy()
        
        # Create date column for easier sorting
        non_financial['Date_Quarter'] = non_financial['Year'].astype(str) + 'Q' + non_financial['Quarter'].astype(str)
        
        # Sort by ticker and date
        non_financial = non_financial.sort_values(['Ticker', 'Year', 'Quarter'])
        
        return non_financial, sector_map
    
    def calculate_core_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Section 1: Core Components Calculation
        Modified to use Gross Margin as primary component
        PBT = Gross_Margin + SG_A + IE + Non_Recurring
        where Gross_Margin = Net_Revenue + COGS
        """
        df = df.copy()
        
        # Calculate Gross Margin (Revenue + COGS)
        df['Gross_Margin'] = df['Net_Revenue'] + df['COGS']  # COGS is negative, so this gives gross margin
        
        # Core components (now reorganized)
        df['Core_Revenue'] = df['Net_Revenue']
        
        # Core operating costs (excluding interest) - both are negative
        df['Core_Operating_Costs'] = df['COGS'] + df['SG_A']
        
        # Core operating income
        df['Core_Operating_Income'] = df['Net_Revenue'] + df['COGS'] + df['SG_A']
        
        # Core PBT (Gross Margin + SG_A + IE)
        df['Core_PBT'] = df['Gross_Margin'] + df['SG_A'] + df['IE']
        
        # Non-recurring identification
        df['Non_Recurring'] = df['PBT'] - df['Core_PBT']
        
        # Verification - using new formula
        df['PBT_Check'] = df['Gross_Margin'] + df['SG_A'] + df['IE'] + df['Non_Recurring']
        
        return df
    
    def calculate_period_changes_t12m(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Section 2: T12M Period-over-Period Changes (default)
        IDENTICAL to banking calculation
        """
        df = df.sort_values(['Ticker', 'Year', 'Quarter']).copy()
        
        # Metrics to calculate T12M for
        metrics = ['Net_Revenue', 'COGS', 'SG_A', 'IE', 'Non_Recurring', 'PBT', 'NPATMI', 
                  'Gross_Margin', 'Core_Operating_Costs', 'Core_Operating_Income', 'Core_PBT']
        
        # 4-quarter rolling average, shifted by 1
        for metric in metrics:
            df[f'{metric}_T12M'] = df.groupby('Ticker')[metric].transform(
                lambda x: x.rolling(window=4, min_periods=4).mean().shift(1)
            )
        
        # Calculate Gross Margin % for base period
        df['Gross_Margin_%_T12M'] = np.where(
            df['Net_Revenue_T12M'] != 0,
            (df['Gross_Margin_T12M'] / df['Net_Revenue_T12M']) * 100,
            0
        )
        
        # Calculate changes vs T12M
        df['Revenue_Change'] = df['Net_Revenue'] - df['Net_Revenue_T12M']
        df['COGS_Change'] = df['COGS'] - df['COGS_T12M']
        df['Gross_Margin_Change'] = df['Gross_Margin'] - df['Gross_Margin_T12M']
        df['SGA_Change'] = df['SG_A'] - df['SG_A_T12M']
        df['IE_Change'] = df['IE'] - df['IE_T12M']
        df['Operating_Cost_Change'] = df['Core_Operating_Costs'] - df['Core_Operating_Costs_T12M']
        df['Non_Recurring_Change'] = df['Non_Recurring'] - df['Non_Recurring_T12M']
        df['PBT_Change'] = df['PBT'] - df['PBT_T12M']
        df['NPATMI_Change'] = df['NPATMI'] - df['NPATMI_T12M']
        
        return df
    
    def calculate_period_changes_qoq(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Section 2: QoQ Period-over-Period Changes
        IDENTICAL to banking calculation
        """
        df = df.sort_values(['Ticker', 'Year', 'Quarter']).copy()
        
        # Metrics to calculate QoQ for
        metrics = ['Net_Revenue', 'COGS', 'SG_A', 'IE', 'Non_Recurring', 'PBT', 'NPATMI',
                  'Gross_Margin', 'Core_Operating_Costs', 'Core_Operating_Income', 'Core_PBT']
        
        # Previous quarter values
        for metric in metrics:
            df[f'{metric}_QoQ'] = df.groupby('Ticker')[metric].shift(1)
        
        # Calculate Gross Margin % for base period
        df['Gross_Margin_%_QoQ'] = np.where(
            df['Net_Revenue_QoQ'] != 0,
            (df['Gross_Margin_QoQ'] / df['Net_Revenue_QoQ']) * 100,
            0
        )
        
        # Calculate changes vs previous quarter
        df['Revenue_Change_QoQ'] = df['Net_Revenue'] - df['Net_Revenue_QoQ']
        df['COGS_Change_QoQ'] = df['COGS'] - df['COGS_QoQ']
        df['Gross_Margin_Change_QoQ'] = df['Gross_Margin'] - df['Gross_Margin_QoQ']
        df['SGA_Change_QoQ'] = df['SG_A'] - df['SG_A_QoQ']
        df['IE_Change_QoQ'] = df['IE'] - df['IE_QoQ']
        df['Operating_Cost_Change_QoQ'] = df['Core_Operating_Costs'] - df['Core_Operating_Costs_QoQ']
        df['Non_Recurring_Change_QoQ'] = df['Non_Recurring'] - df['Non_Recurring_QoQ']
        df['PBT_Change_QoQ'] = df['PBT'] - df['PBT_QoQ']
        df['NPATMI_Change_QoQ'] = df['NPATMI'] - df['NPATMI_QoQ']
        
        return df
    
    def calculate_period_changes_yoy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Section 2: YoY Period-over-Period Changes
        IDENTICAL to banking calculation
        """
        df = df.sort_values(['Ticker', 'Year', 'Quarter']).copy()
        
        # Metrics to calculate YoY for
        metrics = ['Net_Revenue', 'COGS', 'SG_A', 'IE', 'Non_Recurring', 'PBT', 'NPATMI',
                  'Gross_Margin', 'Core_Operating_Costs', 'Core_Operating_Income', 'Core_PBT']
        
        # Same quarter last year (4 quarters back)
        for metric in metrics:
            df[f'{metric}_YoY'] = df.groupby('Ticker')[metric].shift(4)
        
        # Calculate Gross Margin % for base period  
        df['Gross_Margin_%_YoY'] = np.where(
            df['Net_Revenue_YoY'] != 0,
            (df['Gross_Margin_YoY'] / df['Net_Revenue_YoY']) * 100,
            0
        )
        
        # Calculate changes vs same quarter last year
        df['Revenue_Change_YoY'] = df['Net_Revenue'] - df['Net_Revenue_YoY']
        df['COGS_Change_YoY'] = df['COGS'] - df['COGS_YoY']
        df['Gross_Margin_Change_YoY'] = df['Gross_Margin'] - df['Gross_Margin_YoY']
        df['SGA_Change_YoY'] = df['SG_A'] - df['SG_A_YoY']
        df['IE_Change_YoY'] = df['IE'] - df['IE_YoY']
        df['Operating_Cost_Change_YoY'] = df['Core_Operating_Costs'] - df['Core_Operating_Costs_YoY']
        df['Non_Recurring_Change_YoY'] = df['Non_Recurring'] - df['Non_Recurring_YoY']
        df['PBT_Change_YoY'] = df['PBT'] - df['PBT_YoY']
        df['NPATMI_Change_YoY'] = df['NPATMI'] - df['NPATMI_YoY']
        
        return df
    
    def calculate_raw_contributions(self, df: pd.DataFrame, suffix: str = '') -> pd.DataFrame:
        """
        Section 3: Raw Contributions Calculation
        Modified to decompose Gross Margin into Revenue Growth and Margin Expansion effects
        PBT = Gross_Margin + SG_A + IE + Non_Recurring
        """
        df = df.copy()
        
        # Get appropriate column names based on suffix
        if suffix == '_QoQ':
            base_suffix = '_QoQ'
            gm_pct_base = 'Gross_Margin_%_QoQ'
            rev_base = 'Net_Revenue_QoQ'
        elif suffix == '_YoY':
            base_suffix = '_YoY'
            gm_pct_base = 'Gross_Margin_%_YoY'
            rev_base = 'Net_Revenue_YoY'
        else:  # T12M or default
            base_suffix = '_T12M'
            gm_pct_base = 'Gross_Margin_%_T12M'
            rev_base = 'Net_Revenue_T12M'
            suffix = ''  # No suffix for T12M display
        
        gm_col = f'Gross_Margin_Change{suffix}'
        rev_col = f'Revenue_Change{suffix}'
        sga_col = f'SGA_Change{suffix}'
        ie_col = f'IE_Change{suffix}'
        nr_col = f'Non_Recurring_Change{suffix}'
        
        # Calculate current GM%
        df['Gross_Margin_%_Current'] = np.where(
            df['Net_Revenue'] != 0,
            (df['Gross_Margin'] / df['Net_Revenue']) * 100,
            0
        )
        
        # Decompose Gross Margin change into two effects:
        # 1. Revenue Growth Effect = (Revenue_current - Revenue_base) × GM%_base
        # 2. Margin Expansion Effect = Revenue_current × (GM%_current - GM%_base)
        
        # Revenue Growth Effect (volume/price impact at old margin)
        df[f'Raw_Revenue_Growth_Effect{suffix}'] = np.where(
            pd.notna(df[gm_pct_base]),
            df[rev_col] * (df[gm_pct_base] / 100),  # Convert percentage to decimal
            0
        )
        
        # Margin Expansion Effect (margin improvement on current revenue)
        df[f'Raw_Margin_Expansion_Effect{suffix}'] = np.where(
            pd.notna(df[gm_pct_base]),
            df['Net_Revenue'] * ((df['Gross_Margin_%_Current'] - df[gm_pct_base]) / 100),
            0
        )
        
        # Gross Margin total (should equal sum of the two effects)
        df[f'Raw_Gross_Margin{suffix}'] = df[gm_col]
        
        # Verify the decomposition
        df[f'GM_Decomposition_Check{suffix}'] = (
            df[f'Raw_Revenue_Growth_Effect{suffix}'] + 
            df[f'Raw_Margin_Expansion_Effect{suffix}'] - 
            df[f'Raw_Gross_Margin{suffix}']
        )
        
        # SG&A contribution (separate main component)
        df[f'Raw_SGA{suffix}'] = df[sga_col]
        
        # Interest expense contribution
        df[f'Raw_Interest{suffix}'] = df[ie_col]
        
        # Non-recurring contribution
        df[f'Raw_Non_Recurring{suffix}'] = df[nr_col]
        
        # Verification: should sum to PBT_Change
        df[f'Check_Sum{suffix}'] = (df[f'Raw_Gross_Margin{suffix}'] + 
                                    df[f'Raw_SGA{suffix}'] + 
                                    df[f'Raw_Interest{suffix}'] + 
                                    df[f'Raw_Non_Recurring{suffix}'])
        
        return df
    
    def apply_pbt_threshold(self, df: pd.DataFrame, suffix: str = '') -> pd.DataFrame:
        """
        Section 4: Small PBT Threshold Adjustment
        IDENTICAL to banking calculation
        """
        df = df.copy()
        
        pbt_change_col = f'PBT_Change{suffix}'
        
        # Avoid division by zero
        df[f'PBT_Change_NonZero{suffix}'] = df[pbt_change_col].replace(0, np.nan)
        
        # Flag small changes
        mask_small_pbt = df[pbt_change_col].abs() < self.small_pbt_threshold
        df[f'Small_PBT_Flag{suffix}'] = mask_small_pbt
        
        # Adjust for normalization
        df[f'PBT_Change_Adjusted{suffix}'] = df[f'PBT_Change_NonZero{suffix}'].copy()
        df.loc[mask_small_pbt & (df[pbt_change_col] > 0), f'PBT_Change_Adjusted{suffix}'] = self.small_pbt_threshold
        df.loc[mask_small_pbt & (df[pbt_change_col] < 0), f'PBT_Change_Adjusted{suffix}'] = -self.small_pbt_threshold
        
        # Use absolute value for division
        df[f'PBT_Change_Abs_Adjusted{suffix}'] = df[f'PBT_Change_Adjusted{suffix}'].abs()
        
        return df
    
    def calculate_normalized_scores(self, df: pd.DataFrame, suffix: str = '') -> pd.DataFrame:
        """
        Section 5 & 6: Normalized Score Calculation (% of |PBT Change|)
        Modified to use decomposed Gross Margin sub-components
        """
        df = df.copy()
        
        pbt_abs_col = f'PBT_Change_Abs_Adjusted{suffix}'
        
        # Four main components: Gross Margin, SG&A, Interest, Non-Recurring
        df[f'Gross_Margin_Score{suffix}'] = (df[f'Raw_Gross_Margin{suffix}'] / df[pbt_abs_col]) * 100
        df[f'SGA_Score{suffix}'] = (df[f'Raw_SGA{suffix}'] / df[pbt_abs_col]) * 100
        df[f'Interest_Score{suffix}'] = (df[f'Raw_Interest{suffix}'] / df[pbt_abs_col]) * 100
        df[f'Non_Recurring_Score{suffix}'] = (df[f'Raw_Non_Recurring{suffix}'] / df[pbt_abs_col]) * 100
        
        # Sub-component scores of Gross Margin (new decomposition)
        df[f'Revenue_Growth_Sub_Score{suffix}'] = (df[f'Raw_Revenue_Growth_Effect{suffix}'] / df[pbt_abs_col]) * 100
        df[f'Margin_Expansion_Sub_Score{suffix}'] = (df[f'Raw_Margin_Expansion_Effect{suffix}'] / df[pbt_abs_col]) * 100
        
        # Legacy columns for backward compatibility (will map to new ones in display)
        df[f'Revenue_Sub_Score{suffix}'] = df[f'Revenue_Growth_Sub_Score{suffix}']
        df[f'COGS_Sub_Score{suffix}'] = df[f'Margin_Expansion_Sub_Score{suffix}']
        
        # Calculate initial total (before scaling)
        df[f'Total_Score{suffix}'] = (df[f'Gross_Margin_Score{suffix}'] + 
                                      df[f'SGA_Score{suffix}'] + 
                                      df[f'Interest_Score{suffix}'] +
                                      df[f'Non_Recurring_Score{suffix}'])
        
        return df
    
    def apply_proportional_scaling(self, df: pd.DataFrame, suffix: str = '') -> pd.DataFrame:
        """
        Section 8: Proportional Scaling (Ensure ±100% Total)
        IDENTICAL to banking methodology
        """
        df = df.copy()
        
        small_pbt_col = f'Small_PBT_Flag{suffix}'
        pbt_change_col = f'PBT_Change{suffix}'
        total_score_col = f'Total_Score{suffix}'
        
        if small_pbt_col in df.columns:
            small_pbt_mask = df[small_pbt_col] & df[total_score_col].notna()
            
            if small_pbt_mask.any():
                # Target based on PBT change sign
                df.loc[small_pbt_mask, f'Target_Total{suffix}'] = np.where(
                    df.loc[small_pbt_mask, pbt_change_col] > 0, 100, -100
                )
                
                # Calculate scaling factor
                df.loc[small_pbt_mask, f'Scale_Factor{suffix}'] = (
                    df.loc[small_pbt_mask, f'Target_Total{suffix}'] / 
                    df.loc[small_pbt_mask, total_score_col]
                )
                
                # Apply to main scores
                main_scores = ['Gross_Margin_Score', 'SGA_Score', 'Interest_Score', 'Non_Recurring_Score']
                for score in main_scores:
                    score_col = f'{score}{suffix}'
                    df.loc[small_pbt_mask, score_col] = (
                        df.loc[small_pbt_mask, score_col] * df.loc[small_pbt_mask, f'Scale_Factor{suffix}']
                    )
                
                # Apply to sub-scores (both new and legacy names)
                sub_scores = ['Revenue_Growth_Sub_Score', 'Margin_Expansion_Sub_Score', 
                             'Revenue_Sub_Score', 'COGS_Sub_Score']
                for score in sub_scores:
                    score_col = f'{score}{suffix}'
                    if score_col in df.columns:
                        df.loc[small_pbt_mask, score_col] = (
                            df.loc[small_pbt_mask, score_col] * df.loc[small_pbt_mask, f'Scale_Factor{suffix}']
                        )
                
                # Recalculate total after scaling
                df[total_score_col] = (df[f'Gross_Margin_Score{suffix}'] + 
                                       df[f'SGA_Score{suffix}'] + 
                                       df[f'Interest_Score{suffix}'] +
                                       df[f'Non_Recurring_Score{suffix}'])
                
                # Clean up temporary columns
                df = df.drop([f'Target_Total{suffix}', f'Scale_Factor{suffix}'], axis=1, errors='ignore')
        
        return df
    
    def apply_score_capping(self, df: pd.DataFrame, suffix: str = '') -> pd.DataFrame:
        """
        Section 9: Score Capping - REMOVED
        No capping applied, scores can exceed ±500%
        """
        df = df.copy()
        
        # No capping - just recalculate total and round
        df[f'Total_Score{suffix}'] = (df[f'Gross_Margin_Score{suffix}'] + 
                                      df[f'SGA_Score{suffix}'] + 
                                      df[f'Interest_Score{suffix}'] +
                                      df[f'Non_Recurring_Score{suffix}'])
        
        # Round all scores for readability
        score_columns = [f'Gross_Margin_Score{suffix}', f'SGA_Score{suffix}', 
                        f'Interest_Score{suffix}', f'Non_Recurring_Score{suffix}',
                        f'Revenue_Sub_Score{suffix}', f'COGS_Sub_Score{suffix}',
                        f'Total_Score{suffix}']
        
        for col in score_columns:
            if col in df.columns:
                df[col] = df[col].round(1)
        
        return df
    
    def calculate_growth_rates(self, df: pd.DataFrame, suffix: str = '') -> pd.DataFrame:
        """
        Section 10: Growth Rate Calculation
        Non-financial uses NPATMI as primary, PBT as secondary
        """
        df = df.copy()
        
        # Get base period columns
        if suffix == '_T12M' or suffix == '':
            npatmi_base = 'NPATMI_T12M'
            pbt_base = 'PBT_T12M'
        elif suffix == '_QoQ':
            npatmi_base = 'NPATMI_QoQ'
            pbt_base = 'PBT_QoQ'
        elif suffix == '_YoY':
            npatmi_base = 'NPATMI_YoY'
            pbt_base = 'PBT_YoY'
        else:
            npatmi_base = 'NPATMI_T12M'
            pbt_base = 'PBT_T12M'
        
        # NPATMI Growth % (primary metric)
        if npatmi_base in df.columns:
            df[f'NPATMI_Growth_%{suffix}'] = np.where(
                df[npatmi_base].notna() & (df[npatmi_base] != 0),
                (df[f'NPATMI_Change{suffix}'] / df[npatmi_base].abs()) * 100,
                0
            )
        else:
            df[f'NPATMI_Growth_%{suffix}'] = 0
        
        # PBT Growth % (for reference)
        if pbt_base in df.columns:
            df[f'PBT_Growth_%{suffix}'] = np.where(
                df[pbt_base].notna() & (df[pbt_base] != 0),
                (df[f'PBT_Change{suffix}'] / df[pbt_base].abs()) * 100,
                0
            )
        else:
            df[f'PBT_Growth_%{suffix}'] = 0
        
        return df
    
    def calculate_weighted_impacts(self, df: pd.DataFrame, suffix: str = '') -> pd.DataFrame:
        """
        Section 11: Weighted Impact Calculation
        Impact = Score × |PBT_Growth_%| / 100
        Note: Display still shows NPATMI_Growth_% but impact calculation uses PBT_Growth_%
        """
        df = df.copy()
        
        growth_col = f'PBT_Growth_%{suffix}'
        
        # Main impacts
        main_scores = ['Gross_Margin_Score', 'SGA_Score', 'Interest_Score', 'Non_Recurring_Score']
        for score in main_scores:
            score_col = f'{score}{suffix}'
            impact_col = score_col.replace('_Score', '_Impact')
            if score_col in df.columns:
                df[impact_col] = (df[score_col] * df[growth_col].abs()) / 100
                df[impact_col] = df[impact_col].round(1)
        
        # Sub-impacts (new decomposition)
        new_sub_scores = ['Revenue_Growth_Sub_Score', 'Margin_Expansion_Sub_Score']
        for score in new_sub_scores:
            score_col = f'{score}{suffix}'
            impact_col = score_col.replace('_Sub_Score', '_Sub_Impact')
            if score_col in df.columns:
                df[impact_col] = (df[score_col] * df[growth_col].abs()) / 100
                df[impact_col] = df[impact_col].round(1)
        
        # Legacy sub-impact names for compatibility
        if f'Revenue_Growth_Sub_Impact{suffix}' in df.columns:
            df[f'Revenue_Sub_Impact{suffix}'] = df[f'Revenue_Growth_Sub_Impact{suffix}']
        if f'Margin_Expansion_Sub_Impact{suffix}' in df.columns:
            df[f'COGS_Sub_Impact{suffix}'] = df[f'Margin_Expansion_Sub_Impact{suffix}']
        
        # Total Impact should equal |PBT_Growth_%|
        df[f'Total_Impact{suffix}'] = (df[f'Gross_Margin_Impact{suffix}'] + 
                                       df[f'SGA_Impact{suffix}'] + 
                                       df[f'Interest_Impact{suffix}'] +
                                       df[f'Non_Recurring_Impact{suffix}'])
        df[f'Total_Impact{suffix}'] = df[f'Total_Impact{suffix}'].round(1)
        
        return df
    
    def calculate_complete_scores(self, df: pd.DataFrame, comparison: str = 'T12M') -> pd.DataFrame:
        """
        Complete calculation pipeline for a specific comparison period
        Following exact banking methodology
        """
        # Determine suffix
        if comparison == 'T12M':
            suffix = ''  # Default, no suffix
        elif comparison == 'QoQ':
            suffix = '_QoQ'
        elif comparison == 'YoY':
            suffix = '_YoY'
        else:
            suffix = ''
        
        # Step 1: Calculate period changes
        if comparison == 'QoQ':
            df = self.calculate_period_changes_qoq(df)
        elif comparison == 'YoY':
            df = self.calculate_period_changes_yoy(df)
        else:  # T12M
            df = self.calculate_period_changes_t12m(df)
        
        # Step 2: Calculate raw contributions
        df = self.calculate_raw_contributions(df, suffix)
        
        # Step 3: Apply PBT threshold
        df = self.apply_pbt_threshold(df, suffix)
        
        # Step 4: Calculate normalized scores
        df = self.calculate_normalized_scores(df, suffix)
        
        # Step 5: Apply proportional scaling (ensure ±100%)
        df = self.apply_proportional_scaling(df, suffix)
        
        # Step 6: Apply score rounding (no capping)
        df = self.apply_score_capping(df, suffix)
        
        # Step 7: Calculate growth rates
        df = self.calculate_growth_rates(df, suffix)
        
        # Step 8: Calculate weighted impacts
        df = self.calculate_weighted_impacts(df, suffix)
        
        return df
    
    def calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point: Calculate all metrics for all comparison periods
        """
        # Calculate core components first
        df = self.calculate_core_components(df)
        
        # Calculate for T12M (default)
        df = self.calculate_complete_scores(df, 'T12M')
        
        # Calculate for QoQ
        df = self.calculate_complete_scores(df, 'QoQ')
        
        # Calculate for YoY
        df = self.calculate_complete_scores(df, 'YoY')
        
        # Add some useful ratios for reference
        df['Gross_Margin_%'] = ((df['Net_Revenue'] + df['COGS']) / df['Net_Revenue']) * 100
        df['Operating_Cost_Ratio_%'] = (-df['Core_Operating_Costs'] / df['Net_Revenue']) * 100
        df['COGS_Ratio_%'] = (-df['COGS'] / df['Net_Revenue']) * 100
        df['SGA_Ratio_%'] = (-df['SG_A'] / df['Net_Revenue']) * 100
        df['IE_Ratio_%'] = (-df['IE'] / df['Net_Revenue']) * 100
        
        return df
    
    def get_waterfall_data(self, df: pd.DataFrame, ticker: str, year: int, quarter: int, 
                           comparison: str = 'QoQ') -> Dict:
        """Prepare data for waterfall chart showing complete PBT attribution"""
        # Get current and previous period data
        current = df[(df['Ticker'] == ticker) & (df['Year'] == year) & (df['Quarter'] == quarter)]
        
        if comparison == 'QoQ':
            if quarter == 1:
                prev_year, prev_quarter = year - 1, 4
            else:
                prev_year, prev_quarter = year, quarter - 1
        else:  # YoY
            prev_year, prev_quarter = year - 1, quarter
        
        previous = df[(df['Ticker'] == ticker) & (df['Year'] == prev_year) & (df['Quarter'] == prev_quarter)]
        
        if current.empty or previous.empty:
            return None
        
        current = current.iloc[0]
        previous = previous.iloc[0]
        
        # Build waterfall data with complete attribution
        waterfall = {
            'categories': ['Previous PBT', 'Revenue Change', 'COGS Change', 'SG&A Change', 
                          'Interest Change', 'Non-Recurring Change', 'Current PBT'],
            'values': [
                previous['PBT'],
                current['Net_Revenue'] - previous['Net_Revenue'],
                -(current['COGS'] - previous['COGS']),  # COGS is negative
                -(current['SG_A'] - previous['SG_A']),  # SG&A is negative
                -(current['IE'] - previous['IE']),      # IE is negative
                current['Non_Recurring'] - previous['Non_Recurring'],
                current['PBT']
            ]
        }
        
        return waterfall
    
    def get_sector_summary(self, df: pd.DataFrame, year: int = None, quarter: int = None) -> pd.DataFrame:
        """Get aggregated metrics by sector with complete attribution"""
        if year and quarter:
            period_df = df[(df['Year'] == year) & (df['Quarter'] == quarter)]
        else:
            # Get latest period for each company
            period_df = df.groupby('Ticker').last().reset_index()
        
        # Aggregate by sector
        sector_summary = period_df.groupby('Sector').agg({
            'Gross_Margin_Score_YoY': 'mean',
            'SGA_Score_YoY': 'mean',
            'Interest_Score_YoY': 'mean',
            'Non_Recurring_Score_YoY': 'mean',
            'Total_Score_YoY': 'mean',
            'NPATMI_Growth_%_YoY': 'mean',
            'PBT_Growth_%_YoY': 'mean',
            'Gross_Margin_%': 'mean',
            'Operating_Cost_Ratio_%': 'mean',
            'Ticker': 'count'
        }).round(1)
        
        sector_summary.rename(columns={'Ticker': 'Company_Count'}, inplace=True)
        
        return sector_summary
    
    def get_ttm_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trailing twelve months metrics"""
        # TTM is already calculated as default (no suffix)
        # This method is for compatibility
        return df
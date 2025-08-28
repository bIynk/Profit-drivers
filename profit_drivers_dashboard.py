import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_processor import ProfitDriverProcessor

#%%
# Page configuration
st.set_page_config(
    page_title="Profit Drivers Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dashboard Configuration
DASHBOARD_CONFIG = {
    'colors': {
        'positive': '#28a745',
        'negative': '#dc3545',
        'neutral': '#6c757d',
        'main_component': '#1976d2',
        'sub_component': '#64b5f6',
        'metric_header': '#0d47a1'
    },
    'display': {
        'max_companies_trend': 5,
        'table_height': 600,
        'waterfall_height': 500
    }
}

# Consolidated Custom CSS for entire dashboard
DASHBOARD_CSS = """
<style>
    /* Metric Cards */
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    
    .metric-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    
    /* Dataframe Styling */
    .stDataFrame [data-testid="stDataFrameResizable"] > div > div > div > div > table {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Main component column headers - bold with stronger color */
    .stDataFrame thead th:nth-child(4),
    .stDataFrame thead th:nth-child(7),
    .stDataFrame thead th:nth-child(8),
    .stDataFrame thead th:nth-child(9) {
        font-weight: 900 !important;
        font-size: 14px !important;
        background-color: rgba(25, 118, 210, 0.15) !important;
        color: #1976d2 !important;
    }
    
    /* Sub-component headers with tree symbols */
    .stDataFrame thead th:nth-child(5),
    .stDataFrame thead th:nth-child(6) {
        font-weight: normal !important;
        font-style: italic !important;
        font-size: 13px !important;
        background-color: rgba(100, 181, 246, 0.08) !important;
        padding-left: 20px !important;
    }
    
    /* Main component data cells */
    .stDataFrame tbody td:nth-child(4),
    .stDataFrame tbody td:nth-child(7),
    .stDataFrame tbody td:nth-child(8),
    .stDataFrame tbody td:nth-child(9) {
        font-weight: 600 !important;
        color: #333 !important;
    }
    
    /* Sub-component data cells with lighter colors */
    .stDataFrame tbody td:nth-child(5),
    .stDataFrame tbody td:nth-child(6) {
        font-style: italic !important;
        background-color: rgba(100, 181, 246, 0.08) !important;
        color: #6a6a6a !important;
        padding-left: 20px !important;
        font-size: 0.95em !important;
    }
    
    /* Row hover effects */
    .stDataFrame tbody tr:hover {
        background-color: rgba(102, 126, 234, 0.08) !important;
    }
    
    /* PBT Growth header highlight */
    .stDataFrame thead th:nth-child(3) {
        background-color: rgba(76, 175, 80, 0.08) !important;
        font-weight: bold !important;
    }
</style>
"""

# Apply consolidated CSS
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

#%%
# Load and process data
@st.cache_data
def load_data():
    """Load and process financial data with complete profit attribution"""
    processor = ProfitDriverProcessor()
    df, sector_map = processor.load_data()
    df = processor.calculate_metrics(df)  # Now uses exact banking methodology
    return df, sector_map, processor

# Initialize data
df, sector_map, processor = load_data()

# Remove any rows with all NaN values in key metrics
df = df.dropna(subset=['Net_Revenue', 'PBT', 'EBIT'], how='all')

#%%
# Sidebar configuration
st.sidebar.header("🔍 Filters")

# Create period options in YYYYQX format
periods = []
for _, row in df[['Year', 'Quarter']].drop_duplicates().sort_values(['Year', 'Quarter'], ascending=[False, False]).iterrows():
    periods.append(f"{int(row['Year'])}Q{int(row['Quarter'])}")

# Time period filter - single dropdown
selected_period = st.sidebar.selectbox("Select Period", periods, index=0)

# Extract year and quarter from selected period
selected_year = int(selected_period[:4])
selected_quarter = int(selected_period[-1])

# Sector filter
sectors = ['All'] + sorted(df['Sector'].dropna().unique().tolist())
selected_sector = st.sidebar.selectbox("Select Sector", sectors, index=0)

# Filter data by sector
if selected_sector != 'All':
    filtered_df = df[df['Sector'] == selected_sector].copy()
else:
    filtered_df = df.copy()

# Comparison type
comparison_type = st.sidebar.selectbox(
    "Comparison Period",
    ["YoY (Year-over-Year)", "QoQ (Quarter-over-Quarter)", "T12M (Trailing 12 Months)"],
    index=2  # Default to T12M (index 2)
)

# Additional display options can be added here if needed

# Company selection for trend analysis
companies = sorted(filtered_df['Ticker'].unique())

st.sidebar.markdown("---")
st.sidebar.subheader("Trend Analysis Settings")

selected_companies = st.sidebar.multiselect(
    f"Select Companies for Comparison (max {DASHBOARD_CONFIG['display']['max_companies_trend']})", 
    companies, 
    default=companies[:1] if companies else [],
    max_selections=DASHBOARD_CONFIG['display']['max_companies_trend']
)

#%%
# Create tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Overview", 
    "🏢 Company Analysis", 
    "📈 Trend Analysis"
])

#%%
# Helper function to prepare data for display
def prepare_display_data(df, year, quarter, comparison_type):
    """Prepare data for the selected period and comparison type"""
    
    # Filter for the selected period
    period_df = df[(df['Year'] == year) & (df['Quarter'] == quarter)].copy()
    
    # Get the right suffix based on comparison type
    if comparison_type.startswith("YoY"):
        suffix = "_YoY"
        comp_label = "YoY"
    elif comparison_type.startswith("QoQ"):
        suffix = "_QoQ"
        comp_label = "QoQ"
    else:  # T12M
        suffix = ""
        comp_label = "T12M"
    
    # Select relevant columns for display with v3.0.0 structure
    display_columns = [
        'Ticker', 'Sector', f'PBT_Growth_%{suffix}',
        f'Gross_Margin_Impact{suffix}', f'SGA_Impact{suffix}',
        f'Interest_Impact{suffix}', f'Non_Recurring_Impact{suffix}',
        f'Raw_Gross_Margin{suffix}', f'Raw_SGA{suffix}',
        f'Raw_Interest{suffix}', f'Raw_Non_Recurring{suffix}',
        f'Revenue_Change{suffix}', f'COGS_Change{suffix}',
        f'Gross_Margin_Score{suffix}', f'SGA_Score{suffix}',
        f'Interest_Score{suffix}', f'Non_Recurring_Score{suffix}',
        f'Revenue_Sub_Score{suffix}', f'COGS_Sub_Score{suffix}',
        f'Revenue_Sub_Impact{suffix}', f'COGS_Sub_Impact{suffix}',
        f'PBT_Change{suffix}', f'NPATMI_Growth_%{suffix}', f'NPATMI_Change{suffix}'
    ]
    
    # Filter to only include columns that exist
    available_columns = [col for col in display_columns if col in period_df.columns]
    
    # Create display dataframe
    display_df = period_df[available_columns].copy()
    
    # Sort by PBT Growth % (descending)
    if f'PBT_Growth_%{suffix}' in display_df.columns:
        display_df = display_df.sort_values(f'PBT_Growth_%{suffix}', ascending=False)
    
    return display_df, suffix, comp_label

#%%
# Tab 1: Overview with Enhanced Interactive Table
with tab1:
    st.header("Profit Attribution Overview")
    
    # Get data for display
    display_df, suffix, comp_label = prepare_display_data(
        filtered_df, selected_year, selected_quarter, comparison_type
    )
    
    # Remove rows with NaN in key columns
    key_cols = [f'PBT_Growth_%{suffix}', f'Gross_Margin_Impact{suffix}']
    available_key_cols = [col for col in key_cols if col in display_df.columns]
    if available_key_cols:
        display_df = display_df.dropna(subset=available_key_cols)
    
    if len(display_df) > 0:
        # Add company filter multiselect
        st.markdown("### 🎯 Filter Companies")
        
        # Create two columns for better layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            all_companies = sorted(display_df['Ticker'].unique())
            selected_display_companies = st.multiselect(
                "Select companies to display (leave empty to show all):",
                options=all_companies,
                default=[],
                key="company_filter",
                help="You can type to search for specific companies"
            )
        
        with col2:
            # Show selection count
            st.markdown("**Selection Info:**")
            if selected_display_companies:
                st.info(f"📊 Showing {len(selected_display_companies)} of {len(all_companies)} companies")
            else:
                st.info(f"📊 Showing all {len(all_companies)} companies")
        
        # Filter dataframe based on selection
        if selected_display_companies:
            display_df = display_df[display_df['Ticker'].isin(selected_display_companies)]
            
        # Add a separator line
        st.markdown("---")
        
        # Check if there's data after filtering
        if len(display_df) == 0:
            st.warning("No companies match the selected filter. Please adjust your selection.")
        else:
            # Prepare display dataframe with proper column names and formatting
            # Create a formatted version for display with all necessary columns
            formatted_df = display_df.copy()
            
            # Sub-component impacts should already be calculated in data_processor
            # No need to recalculate - just ensure columns exist
            # The data_processor creates these columns:
            # - Revenue_Sub_Impact{suffix} (legacy name for Revenue Growth)
            # - COGS_Sub_Impact{suffix} (legacy name for Margin Expansion)
            # These are already properly calculated with row-specific PBT_Growth_%
            
            # Select and rename columns with proper hierarchy
            # Order: Ticker, Sector, PBT Growth, Gross Margin (Revenue Growth, Margin Expansion), SG&A, Interest, Non-Recurring
            display_columns = {
                'Ticker': 'Ticker',
                'Sector': 'Sector',
                f'PBT_Growth_%{suffix}': 'PBT Growth %',
                f'Gross_Margin_Impact{suffix}': 'Gross Margin',
                f'Revenue_Sub_Impact{suffix}': '├ Revenue Growth',
                f'COGS_Sub_Impact{suffix}': '└ Margin Expansion',
                f'SGA_Impact{suffix}': 'SG&A',
                f'Interest_Impact{suffix}': 'Interest',
                f'Non_Recurring_Impact{suffix}': 'Non-Recurring'
            }
            
            # Filter to available columns and maintain order
            ordered_cols = []
            renamed_cols = {}
            for orig_col, new_col in display_columns.items():
                if orig_col in formatted_df.columns:
                    ordered_cols.append(orig_col)
                    renamed_cols[orig_col] = new_col
            
            formatted_df = formatted_df[ordered_cols].rename(columns=renamed_cols)
            
            # Display the main title
            st.markdown("### 📊 Interactive Profit Attribution Table")
            st.markdown("*Click column headers to sort*")
            
            # Configure column display for main table with hierarchy
            column_config = {
                'Ticker': st.column_config.TextColumn('Ticker', width=80),
                'Sector': st.column_config.TextColumn('Sector', width=130),
                'PBT Growth %': st.column_config.NumberColumn(
                    'PBT Growth %',
                    format='%.1f%%',
                    help='Profit Before Tax Growth Rate',
                    width=120
                ),
                'Gross Margin': st.column_config.NumberColumn(
                    'Gross Margin',
                    format='%.1f%%',
                    help='Gross Margin total impact (Revenue + COGS)',
                    width=110
                ),
                '├ Revenue Growth': st.column_config.NumberColumn(
                    '├ Revenue Growth',
                    format='%.1f%%',
                    help='Revenue Growth impact (sub-component of Gross Margin)',
                    width=120
                ),
                '└ Margin Expansion': st.column_config.NumberColumn(
                    '└ Margin Expansion',
                    format='%.1f%%',
                    help='Margin Expansion impact (sub-component of Gross Margin)',
                    width=130
                ),
                'SG&A': st.column_config.NumberColumn(
                    'SG&A',
                    format='%.1f%%',
                    help='SG&A impact on profit growth',
                    width=80
                ),
                'Interest': st.column_config.NumberColumn(
                    'Interest',
                    format='%.1f%%',
                    help='Interest impact on profit growth',
                    width=90
                ),
                'Non-Recurring': st.column_config.NumberColumn(
                    'Non-Recurring',
                    format='%.1f%%',
                    help='Non-Recurring impact on profit growth',
                    width=120
                )
            }
            
            # Apply comprehensive styling to the dataframe
            def style_dataframe(df):
                """Apply custom styling to differentiate main and sub-components"""
                
                # Create a style function for the entire dataframe
                def apply_styles(x):
                    # Initialize style dataframe with same shape
                    import pandas as pd
                    style_df = pd.DataFrame('', index=x.index, columns=x.columns)
                    
                    # Define column types
                    main_components = ['Gross Margin', 'SG&A', 'Interest', 'Non-Recurring']
                    sub_components = ['├ Revenue Growth', '└ Margin Expansion']
                    
                    for col in x.columns:
                        if col in main_components:
                            # Main component styling - bold with strong background
                            for idx in x.index:
                                val = x.loc[idx, col]
                                if pd.notna(val):
                                    if val > 0:
                                        style_df.loc[idx, col] = 'color: #0d3d1a; font-weight: 700; font-size: 13px; background-color: #c3e6cb; border-left: 3px solid #1e7e34'
                                    elif val < 0:
                                        style_df.loc[idx, col] = 'color: #5a1218; font-weight: 700; font-size: 13px; background-color: #f5c6cb; border-left: 3px solid #bd2130'
                                    else:
                                        style_df.loc[idx, col] = 'color: #2b2e32; font-weight: 700; font-size: 13px; background-color: #d6d8db; border-left: 3px solid #5a6268'
                        
                        elif col in sub_components:
                            # Sub-component styling - lighter green/red based on value
                            for idx in x.index:
                                val = x.loc[idx, col]
                                if pd.notna(val):
                                    if val > 0:
                                        # Lighter green for positive values
                                        style_df.loc[idx, col] = 'color: #155724; font-weight: 400; font-size: 12px; background-color: #d4edda; padding-left: 25px; font-style: italic'
                                    elif val < 0:
                                        # Lighter red for negative values  
                                        style_df.loc[idx, col] = 'color: #721c24; font-weight: 400; font-size: 12px; background-color: #f8d7da; padding-left: 25px; font-style: italic'
                                    else:
                                        style_df.loc[idx, col] = 'color: #495057; font-weight: 400; font-size: 12px; background-color: #eceff1; padding-left: 25px; font-style: italic'
                        
                        elif col == 'PBT Growth %':
                            # PBT Growth styling
                            for idx in x.index:
                                val = x.loc[idx, col]
                                if pd.notna(val):
                                    if val > 0:
                                        style_df.loc[idx, col] = 'color: #28a745; font-weight: bold; font-size: 13px'
                                    elif val < 0:
                                        style_df.loc[idx, col] = 'color: #dc3545; font-weight: bold; font-size: 13px'
                                    else:
                                        style_df.loc[idx, col] = 'font-weight: bold; font-size: 13px'
                    
                    return style_df
                
                return df.style.apply(apply_styles, axis=None)
            
            # Apply the comprehensive styling
            styled_df = style_dataframe(formatted_df)
            
            # Display the main interactive table with native sorting
            st.dataframe(
                styled_df,
                column_config=column_config,
                use_container_width=True,
                hide_index=True,
                height=DASHBOARD_CONFIG['display']['table_height']  # Fixed height for better scrolling
            )
            
            # Add summary metrics
            st.markdown("---")
            st.subheader("Summary Statistics")
            
            # Calculate key statistics
            total_companies = len(display_df)
            positive_growth_mask = display_df[f'PBT_Growth_%{suffix}'] > 0
            positive_companies = positive_growth_mask.sum()
            positive_percentage = (positive_companies / total_companies * 100) if total_companies > 0 else 0
            negative_companies = total_companies - positive_companies
            negative_percentage = (negative_companies / total_companies * 100) if total_companies > 0 else 0
            
            # Display company count metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Companies with Positive PBT Growth", 
                    f"{positive_companies} ({positive_percentage:.1f}%)",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Companies with Negative PBT Growth", 
                    f"{negative_companies} ({negative_percentage:.1f}%)",
                    delta=None
                )
            
            # Analyze primary profit drivers for companies with positive growth
            if positive_companies > 0:
                st.markdown("#### Primary Profit Drivers (for positive growth companies)")
                
                positive_df = display_df[positive_growth_mask].copy()
                
                # Identify primary driver for each company (component with highest absolute impact)
                impact_cols = [
                    f'Gross_Margin_Impact{suffix}',
                    f'SGA_Impact{suffix}',
                    f'Interest_Impact{suffix}',
                    f'Non_Recurring_Impact{suffix}'
                ]
                
                # Create a dataframe to store primary drivers
                driver_analysis = []
                for idx, row in positive_df.iterrows():
                    impacts = {}
                    for col in impact_cols:
                        if col in row and pd.notna(row[col]):
                            component_name = col.replace(f'_Impact{suffix}', '').replace('_', ' ')
                            impacts[component_name] = abs(row[col])
                    
                    if impacts:
                        primary_driver = max(impacts, key=impacts.get)
                        driver_analysis.append({
                            'Ticker': row['Ticker'],
                            'Primary Driver': primary_driver,
                            'Impact': impacts[primary_driver]
                        })
                
                if driver_analysis:
                    driver_df = pd.DataFrame(driver_analysis)
                    driver_summary = driver_df.groupby('Primary Driver').size().reset_index(name='Count')
                    driver_summary['Percentage'] = (driver_summary['Count'] / len(driver_df) * 100).round(1)
                    driver_summary = driver_summary.sort_values('Count', ascending=False)
                    
                    # Display breakdown in columns
                    cols = st.columns(len(driver_summary))
                    for i, (col, row) in enumerate(zip(cols, driver_summary.itertuples())):
                        with col:
                            st.metric(
                                row._1,  # Use full component name without abbreviation
                                f"{row.Percentage:.1f}%",
                                delta=f"{row.Count} companies"
                            )
            
            # Create histogram of PBT growth distribution
            st.markdown("#### PBT Growth Distribution")
            
            import plotly.graph_objects as go
            
            # Cap values at -100% and 100% for display
            growth_data = display_df[f'PBT_Growth_%{suffix}'].clip(-100, 100)
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=growth_data,
                nbinsx=40,
                name='PBT Growth Distribution',
                marker_color='rgba(26, 118, 255, 0.7)',
                hovertemplate='Growth Range: %{x}<br>Count: %{y}<extra></extra>'
            ))
            
            # Add vertical line at 0
            fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.5)
            
            fig.update_layout(
                title=f"PBT Growth Distribution ({comp_label})",
                xaxis_title="PBT Growth (%)",
                yaxis_title="Number of Companies",
                xaxis=dict(range=[-100, 100], tickformat='.0f', dtick=20),
                yaxis=dict(title="Number of Companies"),
                height=400,
                showlegend=False,
                hovermode='x unified'
            )
            
            # Add annotation for positive vs negative
            fig.add_annotation(
                x=50, y=0.95, 
                xref='x', yref='paper',
                text=f"Positive: {positive_companies} ({positive_percentage:.1f}%)",
                showarrow=False,
                bgcolor="rgba(40, 167, 69, 0.1)",
                bordercolor="rgba(40, 167, 69, 0.5)",
                borderwidth=1
            )
            
            fig.add_annotation(
                x=-50, y=0.95,
                xref='x', yref='paper', 
                text=f"Negative: {total_companies - positive_companies} ({100-positive_percentage:.1f}%)",
                showarrow=False,
                bgcolor="rgba(220, 53, 69, 0.1)",
                bordercolor="rgba(220, 53, 69, 0.5)",
                borderwidth=1
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("No data available for the selected period and filters.")

#%%
# Tab 2: Company Analysis
with tab2:
    st.header("Company Analysis")
    
    # Company selection at the top of the tab
    col1, col2 = st.columns([2, 3])
    with col1:
        selected_company_tab = st.selectbox(
            "Select Company",
            companies,
            index=0 if companies else None,
            key="company_tab_select"
        )
    
    # Get company data
    company_df = filtered_df[filtered_df['Ticker'] == selected_company_tab].copy()
    
    if len(company_df) > 0:
        # Get latest period data
        latest_period = company_df[(company_df['Year'] == selected_year) & 
                                  (company_df['Quarter'] == selected_quarter)]
        
        if len(latest_period) > 0:
            latest = latest_period.iloc[0]
            
            # Display company metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Sector", latest['Sector'])
            with col2:
                st.metric("Period", f"{selected_year}Q{selected_quarter}")
            with col3:
                growth_col = f'PBT_Growth_%{suffix}'
                if growth_col in latest:
                    st.metric(f"PBT Growth ({comp_label})", f"{latest[growth_col]:.1f}%")
            with col4:
                if 'Gross_Margin_%' in latest:
                    st.metric("Gross Margin %", f"{latest['Gross_Margin_%']:.1f}%")
            
            # Create waterfall chart for profit drivers with v3.0.0 structure
            st.subheader("PBT Evolution - Waterfall Analysis")
            
            # Get previous period PBT based on comparison type
            prev_pbt = 0
            if comparison_type.startswith("YoY"):
                prev_pbt_col = 'PBT_YoY'
            elif comparison_type.startswith("QoQ"):
                prev_pbt_col = 'PBT_QoQ'
            else:  # T12M
                prev_pbt_col = 'PBT_T12M'
            
            if prev_pbt_col in latest and pd.notna(latest[prev_pbt_col]):
                prev_pbt = latest[prev_pbt_col] / 1e9
            
            current_pbt = latest['PBT'] / 1e9 if 'PBT' in latest else 0
            
            # Prepare data for waterfall - include starting and ending PBT
            components = ['Previous PBT', 'Gross Margin', 'SG&A', 'Interest', 'Non-Recurring', 'Current PBT']
            values = [
                prev_pbt,  # Starting point
                latest[f'Raw_Gross_Margin{suffix}'] / 1e9 if f'Raw_Gross_Margin{suffix}' in latest else 0,
                latest[f'Raw_SGA{suffix}'] / 1e9 if f'Raw_SGA{suffix}' in latest else 0,
                latest[f'Raw_Interest{suffix}'] / 1e9 if f'Raw_Interest{suffix}' in latest else 0,
                latest[f'Raw_Non_Recurring{suffix}'] / 1e9 if f'Raw_Non_Recurring{suffix}' in latest else 0,
                None  # Total will be calculated automatically
            ]
            
            # Set measure types for waterfall
            measure = ['absolute', 'relative', 'relative', 'relative', 'relative', 'total']
            
            # Create waterfall chart
            fig = go.Figure(go.Waterfall(
                orientation="v",
                x=components,
                y=values,
                measure=measure,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                text=[f"{v:.1f}B" if v is not None else f"{current_pbt:.1f}B" for v in values],
                textposition="outside",
                increasing={"marker": {"color": "green"}},
                decreasing={"marker": {"color": "red"}},
                totals={"marker": {"color": "blue"}}
            ))
            
            fig.update_layout(
                title=f"PBT Evolution: {comp_label} (Billion VND)",
                showlegend=False,
                height=DASHBOARD_CONFIG['display']['waterfall_height'],
                yaxis_title="PBT (Billion VND)",
                xaxis_title="Components"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show historical trend
            st.subheader("Historical Performance")
            
            # Prepare historical data
            historical = company_df[['Year', 'Quarter', 'NPATMI', 'PBT', 'Gross_Margin']].copy()
            historical['Period'] = historical['Year'].astype(str) + 'Q' + historical['Quarter'].astype(str)
            historical = historical.sort_values(['Year', 'Quarter'])
            
            # Create line chart
            fig2 = make_subplots(
                rows=1, cols=2,
                subplot_titles=('NPATMI Trend', 'Gross Margin Trend')
            )
            
            fig2.add_trace(
                go.Scatter(x=historical['Period'], y=historical['NPATMI']/1e9,
                          mode='lines+markers', name='NPATMI'),
                row=1, col=1
            )
            
            fig2.add_trace(
                go.Scatter(x=historical['Period'], y=historical['Gross_Margin']/1e9,
                          mode='lines+markers', name='Gross Margin', line=dict(color='green')),
                row=1, col=2
            )
            
            fig2.update_xaxes(tickangle=45)
            fig2.update_yaxes(title_text="Billion VND", row=1, col=1)
            fig2.update_yaxes(title_text="Billion VND", row=1, col=2)
            fig2.update_layout(height=400, showlegend=False)
            
            st.plotly_chart(fig2, use_container_width=True)
            
        else:
            st.warning(f"No data available for {selected_company} in {selected_year}Q{selected_quarter}")
    else:
        st.warning(f"No data available for {selected_company}")

#%%
# Tab 3: Trend Analysis
with tab3:
    st.header("Trend Analysis")
    
    if selected_companies:
        # Filter data for selected companies
        trend_df = filtered_df[filtered_df['Ticker'].isin(selected_companies)].copy()
        
        # Prepare data for trending
        trend_df['Period'] = trend_df['Year'].astype(str) + 'Q' + trend_df['Quarter'].astype(str)
        
        # Create metric selector
        metric_options = {
            'PBT Growth %': f'PBT_Growth_%{suffix}',
            'Gross Margin Impact': f'Gross_Margin_Impact{suffix}',
            'SG&A Impact': f'SGA_Impact{suffix}',
            'Interest Impact': f'Interest_Impact{suffix}',
            'Non-Recurring Impact': f'Non_Recurring_Impact{suffix}'
        }
        
        selected_metric = st.selectbox("Select Metric to Compare", list(metric_options.keys()))
        metric_col = metric_options[selected_metric]
        
        # Create line chart for comparison
        fig = px.line(
            trend_df,
            x='Period',
            y=metric_col,
            color='Ticker',
            title=f'{selected_metric} Comparison ({comp_label})',
            markers=True
        )
        
        fig.update_xaxes(tickangle=45)
        fig.update_yaxis(title=selected_metric)
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show comparison table
        st.subheader("Latest Period Comparison")
        
        latest_comparison = trend_df[
            (trend_df['Year'] == selected_year) & 
            (trend_df['Quarter'] == selected_quarter)
        ]
        
        if len(latest_comparison) > 0:
            comparison_cols = ['Ticker', f'PBT_Growth_%{suffix}', 
                              f'Gross_Margin_Impact{suffix}', f'SGA_Impact{suffix}',
                              f'Interest_Impact{suffix}', f'Non_Recurring_Impact{suffix}']
            
            available_cols = [col for col in comparison_cols if col in latest_comparison.columns]
            
            st.dataframe(
                latest_comparison[available_cols].set_index('Ticker').style.format("{:.1f}%"),
                use_container_width=True
            )
    else:
        st.info("Please select companies from the sidebar to view trends.")

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import io
import base64

# Page config
st.set_page_config(
    page_title="European Power Prices (Monthly)",
    page_icon="⚡",
    layout="wide"
)

@st.cache_data
def load_local_csv(path: str) -> pd.DataFrame:
    """Load CSV file from local path"""
    return pd.read_csv(path)

@st.cache_data
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardise the dataframe"""
    df_clean = df.copy()
    
    # Rename columns
    df_clean.columns = df_clean.columns.str.strip()
    df_clean = df_clean.rename(columns={'Price (EUR/MWhe)': 'price_eur_mwh'})
    
    # Parse dates to monthly periods (first of month)
    df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%d/%m/%Y')
    df_clean['Date'] = df_clean['Date'].dt.to_period('M').dt.start_time
    
    return df_clean

@st.cache_data
def filter_df(df: pd.DataFrame, countries: list, start_date: date, end_date: date) -> pd.DataFrame:
    """Filter dataframe by countries and date range"""
    filtered = df[
        (df['Country'].isin(countries)) &
        (df['Date'] >= pd.Timestamp(start_date)) &
        (df['Date'] <= pd.Timestamp(end_date))
    ].copy()
    return filtered.sort_values(['Country', 'Date'])

@st.cache_data
def make_indexed(df: pd.DataFrame) -> pd.DataFrame:
    """Index prices to 100 at first visible month per country"""
    df_indexed = df.copy()
    
    for country in df_indexed['Country'].unique():
        country_mask = df_indexed['Country'] == country
        country_data = df_indexed[country_mask].sort_values('Date')
        
        if len(country_data) > 0:
            first_price = country_data['price_eur_mwh'].iloc[0]
            df_indexed.loc[country_mask, 'price_eur_mwh'] = (
                country_data['price_eur_mwh'] / first_price * 100
            )
    
    return df_indexed

@st.cache_data
def compute_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Compute latest, 3-month, and 12-month changes by country"""
    changes = []
    
    for country in df['Country'].unique():
        country_data = df[df['Country'] == country].sort_values('Date')
        
        if len(country_data) == 0:
            continue
            
        latest_price = country_data['price_eur_mwh'].iloc[-1]
        latest_date = country_data['Date'].iloc[-1]
        
        # 3-month change
        three_months_ago = latest_date - pd.DateOffset(months=3)
        three_month_data = country_data[country_data['Date'] <= three_months_ago]
        three_month_change = None
        if len(three_month_data) > 0:
            three_month_price = three_month_data['price_eur_mwh'].iloc[-1]
            three_month_change = ((latest_price - three_month_price) / three_month_price) * 100
        
        # 12-month change
        twelve_months_ago = latest_date - pd.DateOffset(months=12)
        twelve_month_data = country_data[country_data['Date'] <= twelve_months_ago]
        twelve_month_change = None
        if len(twelve_month_data) > 0:
            twelve_month_price = twelve_month_data['price_eur_mwh'].iloc[-1]
            twelve_month_change = ((latest_price - twelve_month_price) / twelve_month_price) * 100
        
        changes.append({
            'Country': country,
            'Latest Price': latest_price,
            'Latest Date': latest_date,
            '3M Change %': three_month_change,
            '12M Change %': twelve_month_change,
            'Data Points': len(country_data)
        })
    
    return pd.DataFrame(changes)

def plot_prices(df: pd.DataFrame, indexed: bool = False) -> go.Figure:
    """Create main price chart"""
    if indexed:
        df_plot = make_indexed(df)
        y_title = "Price Index (100 = First Month)"
    else:
        df_plot = df
        y_title = "Price (€/MWh)"
    
    fig = px.line(
        df_plot, 
        x='Date', 
        y='price_eur_mwh', 
        color='Country',
        title="Monthly Electricity Prices by Country",
        labels={'price_eur_mwh': y_title, 'Date': 'Date'}
    )
    
    fig.update_layout(
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def plot_yoy(df: pd.DataFrame) -> go.Figure:
    """Create year-over-year percentage change chart"""
    df_yoy = df.copy()
    df_yoy['YoY_Change'] = 0.0
    
    for country in df_yoy['Country'].unique():
        country_mask = df_yoy['Country'] == country
        country_data = df_yoy[country_mask].sort_values('Date')
        
        if len(country_data) > 12:  # Need at least 12 months for YoY
            country_data = country_data.copy()
            country_data['price_eur_mwh_12m_ago'] = country_data['price_eur_mwh'].shift(12)
            country_data['YoY_Change'] = (
                (country_data['price_eur_mwh'] - country_data['price_eur_mwh_12m_ago']) 
                / country_data['price_eur_mwh_12m_ago'] * 100
            )
            df_yoy.loc[country_mask, 'YoY_Change'] = country_data['YoY_Change']
    
    # Filter out rows with no YoY data
    df_yoy = df_yoy[df_yoy['YoY_Change'] != 0.0]
    
    fig = px.line(
        df_yoy,
        x='Date',
        y='YoY_Change',
        color='Country',
        title="Year-over-Year Price Change (%)",
        labels={'YoY_Change': 'YoY Change (%)', 'Date': 'Date'}
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(height=400, hovermode='x unified')
    
    return fig

def plot_yearly_prices(df: pd.DataFrame) -> go.Figure:
    """Create yearly average price chart"""
    df_yearly = df.copy()
    df_yearly['Year'] = df_yearly['Date'].dt.year
    yearly_avg = df_yearly.groupby(['Country', 'Year'])['price_eur_mwh'].mean().reset_index()
    
    fig = px.line(
        yearly_avg,
        x='Year',
        y='price_eur_mwh',
        color='Country',
        title="Annual Average Electricity Prices by Country",
        labels={'price_eur_mwh': 'Average Price (€/MWh)', 'Year': 'Year'}
    )
    
    fig.update_layout(height=400, hovermode='x unified')
    return fig

@st.cache_data
def create_yearly_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a pivot table with countries as rows and years as columns"""
    df_yearly = df.copy()
    df_yearly['Year'] = df_yearly['Date'].dt.year
    
    # Calculate yearly averages
    yearly_avg = df_yearly.groupby(['Country', 'Year'])['price_eur_mwh'].mean().reset_index()
    
    # Create pivot table with countries as rows and years as columns
    pivot_table = yearly_avg.pivot(index='Country', columns='Year', values='price_eur_mwh')
    
    # Round to 1 decimal place
    pivot_table = pivot_table.round(1)
    
    # Sort countries alphabetically
    pivot_table = pivot_table.sort_index()
    
    # Add a total row with average across all years
    pivot_table.loc['Average'] = pivot_table.mean()
    
    return pivot_table

@st.cache_data
def calculate_datacenter_costs(df: pd.DataFrame, it_load_mw: float, utilization_rate: float, pue: float) -> pd.DataFrame:
    """
    Calculate annual electricity costs for a data center in each country
    
    Parameters:
    - df: DataFrame with electricity price data
    - it_load_mw: IT load in MW
    - utilization_rate: Utilization rate as decimal (0.5 = 50%)
    - pue: Power Usage Effectiveness (e.g., 1.2)
    
    Returns:
    - DataFrame with country costs sorted by total cost
    """
    # Calculate effective IT load (considering utilization)
    effective_it_load_mw = it_load_mw * utilization_rate
    
    # Calculate total facility power (IT load * PUE)
    total_facility_power_mw = effective_it_load_mw * pue
    
    # Calculate annual MWh consumption (8760 hours in a year)
    annual_mwh = total_facility_power_mw * 8760
    
    # Get 2024 average prices for each country
    df_2024 = df[df['Date'].dt.year == 2024]
    if len(df_2024) == 0:
        # Fallback to latest prices if no 2024 data
        avg_prices = df.groupby('Country')['price_eur_mwh'].last().reset_index()
        price_column = 'Latest Price (€/MWh)'
    else:
        avg_prices = df_2024.groupby('Country')['price_eur_mwh'].mean().reset_index()
        price_column = '2024 Avg Price (€/MWh)'
    
    # Calculate annual costs
    costs = []
    for _, row in avg_prices.iterrows():
        country = row['Country']
        price_eur_mwh = row['price_eur_mwh']
        annual_cost_eur = annual_mwh * price_eur_mwh
        
        costs.append({
            'Country': country,
            price_column: round(price_eur_mwh, 2),
            'Annual MWh': round(annual_mwh, 0),
            'Annual Cost (€)': round(annual_cost_eur, 0),
            'Annual Cost (€M)': round(annual_cost_eur / 1_000_000, 2)
        })
    
    # Convert to DataFrame and sort by cost
    costs_df = pd.DataFrame(costs)
    costs_df = costs_df.sort_values('Annual Cost (€)', ascending=True)
    
    return costs_df

@st.cache_data
def calculate_datacenter_efficiency_metrics(it_load_mw: float, utilization_rate: float, pue: float) -> dict:
    """Calculate data center efficiency metrics"""
    effective_it_load_mw = it_load_mw * utilization_rate
    total_facility_power_mw = effective_it_load_mw * pue
    annual_mwh = total_facility_power_mw * 8760
    
    return {
        'IT Load (MW)': it_load_mw,
        'Utilization Rate': f"{utilization_rate * 100:.0f}%",
        'Average IT Load (MW)': round(effective_it_load_mw, 1),
        'PUE': pue,
        'Total Facility Power (MW)': round(total_facility_power_mw, 1),
        'Annual Consumption (MWh)': round(annual_mwh, 0)
    }

# Load and clean data
@st.cache_data
def load_data():
    df_raw = load_local_csv('european_wholesale_electricity_price_data_monthly.csv')
    return clean_df(df_raw)

df = load_data()

# Main app structure with tabs
st.title("⚡ European Power Prices Analysis")

# Initialize session state
if 'show_datacenter_results' not in st.session_state:
    st.session_state.show_datacenter_results = True  # Start with results showing
    st.session_state.calc_params = {
        'it_load_mw': 100,  # Default IT load
        'utilization_rate': 0.7,  # Default 70% utilization
        'pue': 1.2  # Default PUE
    }

# Create tabs - Data Center tab first to make it default
tab1, tab2 = st.tabs(["🏢 Data Center Cost Analysis", "📊 Price Analysis"])

with tab1:
    # Data Center Cost Analysis
    st.header("🏢 Data Center Cost Analysis")

    # Country selection
    all_countries = sorted(df['Country'].unique())
    # Default selection: Ireland, United Kingdom, Spain, Finland, Sweden, Portugal, Poland
    default_countries = ['Ireland', 'United Kingdom', 'Spain', 'Finland', 'Sweden', 'Portugal', 'Poland']

    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        all_countries,
        default=default_countries,
        help="Choose countries to display in charts and analysis"
    )

    # Date range
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Select the date range for analysis"
    )

    start_date, end_date = date_range

    # Toggles
    index_to_100 = st.sidebar.checkbox("Index to 100 at start", value=False)
    show_yoy = st.sidebar.checkbox("Show YoY %", value=False)

    # Filter data
    if selected_countries and start_date and end_date:
        filtered_df = filter_df(df, selected_countries, start_date, end_date)
        
        # Main content
        st.header("📊 European Power Prices (Monthly)")
        
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            latest_month = filtered_df['Date'].max().strftime('%B %Y')
            st.metric("Latest Month", latest_month)
        
        with col2:
            st.metric("Selected Countries", len(selected_countries))
        
        with col3:
            if len(filtered_df) > 0:
                avg_price = filtered_df.groupby('Country')['price_eur_mwh'].last().mean()
                st.metric("Average Latest Price", f"€{avg_price:.1f}/MWh")
        
        with col4:
            total_data_points = len(filtered_df)
            st.metric("Data Points", f"{total_data_points:,}")
        
        # Country statistics
        if len(filtered_df) > 0:
            st.subheader("Country Statistics")
            changes_df = compute_changes(filtered_df)
            
            # Add data quality warnings
            changes_df['Warning'] = changes_df['Data Points'].apply(
                lambda x: "⚠️" if x < 6 else ""
            )
            
            # Format the display
            display_df = changes_df.copy()
            display_df['Latest Price'] = display_df['Latest Price'].round(1)
            display_df['3M Change %'] = display_df['3M Change %'].round(1)
            display_df['12M Change %'] = display_df['12M Change %'].round(1)
            display_df['Country'] = display_df['Country'] + ' ' + display_df['Warning']
            
            st.dataframe(
                display_df[['Country', 'Latest Price', '3M Change %', '12M Change %', 'Data Points']],
                width='stretch'
            )
        
        # Charts
        st.subheader("Price Trends")
        
        # Main price chart
        fig_main = plot_prices(filtered_df, indexed=index_to_100)
        st.plotly_chart(fig_main, width='stretch')
        
        # YoY chart (if enabled)
        if show_yoy and len(filtered_df) > 0:
            st.subheader("Year-over-Year Changes")
            fig_yoy = plot_yoy(filtered_df)
            st.plotly_chart(fig_yoy, width='stretch')
        
        # Yearly chart
        st.subheader("Annual Averages")
        fig_yearly = plot_yearly_prices(filtered_df)
        st.plotly_chart(fig_yearly, width='stretch')
        
        # Yearly price table
        st.subheader("Yearly Price Table")
        yearly_table = create_yearly_table(filtered_df)
        st.dataframe(yearly_table, width='stretch')
        
        # Data table and downloads
        st.subheader("Data Table")
        st.dataframe(filtered_df, width='stretch')
        
        # Download buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV download
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"european_power_prices_{start_date}_{end_date}.csv",
                mime="text/csv"
            )
        
        with col2:
            # HTML download
            html = fig_main.to_html()
            st.download_button(
                label="Download Chart (HTML)",
                data=html,
                file_name=f"price_chart_{start_date}_{end_date}.html",
                mime="text/html"
            )
        
        with col3:
            # PNG download (if kaleido available)
            try:
                png = fig_main.to_image(format="png", width=1200, height=600)
                st.download_button(
                    label="Download Chart (PNG)",
                    data=png,
                    file_name=f"price_chart_{start_date}_{end_date}.png",
                    mime="image/png"
                )
            except Exception:
                st.info("PNG download requires kaleido package")

    else:
        st.warning("Please select at least one country and valid date range.")

with tab2:
    # Sidebar controls for price analysis
    st.sidebar.header("Price Analysis Controls")
    
    st.markdown("""
    This tool calculates the annual electricity costs for a data center across different European countries.
    Adjust the parameters below to see how location affects your data center's electricity costs.
    """)
    
    # Formula explanation
    with st.expander("📋 Calculation Formulas", expanded=False):
        st.markdown("""
        **Data Center Electricity Cost Calculation:**
        
        1. **Average IT Load** = IT Load (MW) × Utilization Rate (%)
        2. **Total Facility Power** = Average IT Load (MW) × PUE
        3. **Annual Consumption** = Total Facility Power (MW) × 8,760 hours/year
        4. **Annual Cost** = Annual Consumption (MWh) × 2024 Average Price (€/MWh)
        
        **Where:**
        - **IT Load**: Total IT equipment power capacity
        - **Utilization Rate**: Percentage of IT load actively used (accounts for idle time)
        - **PUE (Power Usage Effectiveness)**: Ratio of total facility power to IT power (includes cooling, lighting, etc.)
        - **8,760 hours**: Total hours in a year (24 × 365)
        - **2024 Average Price**: Mean electricity price for each country in 2024
        
        **Example:** 100MW IT load, 70% utilization, 1.2 PUE
        - Average IT Load = 100 × 0.7 = 70 MW
        - Total Facility Power = 70 × 1.2 = 84 MW  
        - Annual Consumption = 84 × 8,760 = 735,840 MWh
        - Annual Cost = 735,840 × Price (€/MWh)
        """)
    
    # Input controls in the main tab
    st.subheader("Data Center Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # IT Load selection
        it_load_options = [50, 100, 150, 200]
        default_it_load = st.session_state.calc_params.get('it_load_mw', 100)
        it_load_index = it_load_options.index(default_it_load) if default_it_load in it_load_options else 1
        it_load_mw = st.selectbox(
            "IT Load (MW)",
            it_load_options,
            index=it_load_index,
            help="Total IT load capacity of the data center"
        )
    
    with col2:
        # Utilization rate selection
        utilization_options = [0.5, 0.6, 0.7, 0.8]
        default_utilization = st.session_state.calc_params.get('utilization_rate', 0.7)
        util_index = utilization_options.index(default_utilization) if default_utilization in utilization_options else 2
        utilization_rate = st.selectbox(
            "Utilization Rate",
            utilization_options,
            index=util_index,
            format_func=lambda x: f"{x*100:.0f}%",
            help="Percentage of IT load that is actively used"
        )
    
    with col3:
        # PUE selection
        pue_options = [1.1, 1.2, 1.3, 1.4, 1.5]
        default_pue = st.session_state.calc_params.get('pue', 1.2)
        pue_index = pue_options.index(default_pue) if default_pue in pue_options else 1
        pue = st.selectbox(
            "Power Usage Effectiveness (PUE)",
            pue_options,
            index=pue_index,
            help="PUE measures data center efficiency (lower is better)"
        )
    
    # Update session state when parameters change
    current_params = {
        'it_load_mw': it_load_mw,
        'utilization_rate': utilization_rate,
        'pue': pue
    }
    
    # Check if parameters have changed
    if current_params != st.session_state.calc_params:
        st.session_state.calc_params = current_params
        st.session_state.show_datacenter_results = True
        st.rerun()
    
    # Calculate and display results
    if st.button("Calculate Costs", type="primary"):
        # Store calculation parameters in session state
        st.session_state.calc_params = {
            'it_load_mw': it_load_mw,
            'utilization_rate': utilization_rate,
            'pue': pue
        }
        st.session_state.show_datacenter_results = True
        st.rerun()
    
    # Check if we have calculation results to display
    if st.session_state.show_datacenter_results and 'calc_params' in st.session_state:
        # Get parameters from session state
        params = st.session_state.calc_params
        it_load_mw = params['it_load_mw']
        utilization_rate = params['utilization_rate']
        pue = params['pue']
        
        # Calculate efficiency metrics
        efficiency_metrics = calculate_datacenter_efficiency_metrics(it_load_mw, utilization_rate, pue)
        
        # Display efficiency metrics
        st.subheader("Data Center Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("IT Load", f"{efficiency_metrics['IT Load (MW)']} MW")
            st.metric("Utilization", efficiency_metrics['Utilization Rate'])
        
        with col2:
            st.metric("Average IT Load", f"{efficiency_metrics['Average IT Load (MW)']} MW")
            st.metric("PUE", efficiency_metrics['PUE'])
        
        with col3:
            st.metric("Total Facility Power", f"{efficiency_metrics['Total Facility Power (MW)']} MW")
            st.metric("Annual Consumption", f"{efficiency_metrics['Annual Consumption (MWh)']:,} MWh")
        
        # Calculate costs for all countries
        costs_df = calculate_datacenter_costs(df, it_load_mw, utilization_rate, pue)
        
        # Display results
        st.subheader("Annual Electricity Costs by Country")
        st.markdown(f"*Sorted by total annual cost (cheapest to most expensive)*")
        
        # Get the price column name (either 2024 Avg Price or Latest Price)
        price_column = [col for col in costs_df.columns if 'Price' in col][0]
        
        # Format the dataframe for better display
        display_costs_df = costs_df.copy()
        display_costs_df['Annual Cost (€)'] = display_costs_df['Annual Cost (€)'].apply(lambda x: f"€{x:,.0f}")
        display_costs_df['Annual Cost (€M)'] = display_costs_df['Annual Cost (€M)'].apply(lambda x: f"€{x:.1f}M")
        display_costs_df['Annual MWh'] = display_costs_df['Annual MWh'].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(
            display_costs_df[['Country', price_column, 'Annual Cost (€M)']],
            width='stretch',
            hide_index=True
        )
        
        # Bar chart visualization
        st.subheader("Cost Comparison Chart")
        
        # Create horizontal bar chart with better colors and all countries
        fig_bar = px.bar(
            costs_df,  # Show ALL countries
            x='Annual Cost (€M)',
            y='Country',
            orientation='h',
            title="Annual Electricity Costs by Country",
            color='Annual Cost (€M)',
            color_continuous_scale=[[0, 'green'], [0.5, 'gold'], [1, 'red']],  # Custom: green to gold to red
            labels={'Annual Cost (€M)': 'Annual Cost (€M)', 'Country': 'Country'}
        )
        
        # Update layout for better appearance
        fig_bar.update_layout(
            height=max(400, len(costs_df) * 25),  # Dynamic height based on number of countries
            showlegend=False,
            plot_bgcolor='rgba(248,249,250,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            font=dict(size=11, family="Arial, sans-serif"),
            title_font_size=20,
            title_x=0.0,  # Left align the title
            margin=dict(l=20, r=20, t=80, b=20)  # Proper top margin for title above chart
        )
        
        # Update x-axis (costs)
        fig_bar.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200,200,200,0.3)',
            title="Annual Cost (€M)",
            title_font_size=14,
            tickfont_size=11
        )
        
        # Update y-axis (countries)
        fig_bar.update_yaxes(
            showgrid=False,
            title="",
            tickfont_size=10,
            categoryorder='total descending'  # Sort by value (cheapest first at top)
        )
        
        # Update color bar for better visibility
        fig_bar.update_coloraxes(
            colorbar=dict(
                title="Cost (€M)",
                title_font_size=12,
                tickfont_size=10,
                len=0.8,
                y=0.5,
                yanchor='middle'
            )
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Country Comparison Section
        st.subheader("Compare Specific Countries")
        
        # Get unique countries for selection
        available_countries = costs_df['Country'].tolist()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            country1 = st.selectbox(
                "Select Country 1",
                available_countries,
                index=0,
                help="Choose first country for comparison"
            )
        
        with col2:
            country2 = st.selectbox(
                "Select Country 2", 
                available_countries,
                index=1,
                help="Choose second country for comparison"
            )
        
        with col3:
            country3 = st.selectbox(
                "Select Country 3",
                available_countries,
                index=2,
                help="Choose third country for comparison"
            )
        
        # Filter data for selected countries
        selected_countries_data = costs_df[costs_df['Country'].isin([country1, country2, country3])].copy()
        
        if len(selected_countries_data) > 0:
            st.subheader("Selected Countries Cost Comparison")
            
            # Create comparison metrics
            comparison_data = []
            for _, row in selected_countries_data.iterrows():
                country = row['Country']
                price = row[price_column]
                annual_cost = row['Annual Cost (€M)']
                
                # Calculate rank (1 = cheapest)
                rank = costs_df[costs_df['Country'] == country].index[0] + 1
                
                comparison_data.append({
                    'Country': country,
                    'Rank': f"#{rank}",
                    'Price (€/MWh)': price,
                    'Annual Cost (€M)': annual_cost,
                    'vs Cheapest': f"{((annual_cost / selected_countries_data['Annual Cost (€M)'].min()) - 1) * 100:.1f}%"
                })
            
            # Sort by cost for display
            comparison_df = pd.DataFrame(comparison_data).sort_values('Annual Cost (€M)')
            
            # Display comparison table
            st.dataframe(
                comparison_df[['Country', 'Rank', 'Price (€/MWh)', 'Annual Cost (€M)', 'vs Cheapest']],
                width='stretch',
                hide_index=True
            )
            
            # Key insights
            st.subheader("Key Insights")
            cheapest = comparison_df.iloc[0]
            most_expensive = comparison_df.iloc[-1]
            cost_difference = most_expensive['Annual Cost (€M)'] - cheapest['Annual Cost (€M)']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Cheapest Selected", 
                    cheapest['Country'], 
                    f"€{cheapest['Annual Cost (€M)']:.1f}M"
                )
            
            with col2:
                st.metric(
                    "Most Expensive Selected", 
                    most_expensive['Country'], 
                    f"€{most_expensive['Annual Cost (€M)']:.1f}M"
                )
            
            with col3:
                st.metric(
                    "Cost Difference", 
                    f"€{cost_difference:.1f}M", 
                    f"{((most_expensive['Annual Cost (€M)'] / cheapest['Annual Cost (€M)']) - 1) * 100:.1f}% more"
                )
        
        # Summary statistics
        st.subheader("Overall Cost Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cheapest_country = costs_df.iloc[0]['Country']
            cheapest_cost = costs_df.iloc[0]['Annual Cost (€M)']
            st.metric("Cheapest Location", f"{cheapest_country}", f"€{cheapest_cost:.1f}M")
        
        with col2:
            most_expensive_country = costs_df.iloc[-1]['Country']
            most_expensive_cost = costs_df.iloc[-1]['Annual Cost (€M)']
            st.metric("Most Expensive Location", f"{most_expensive_country}", f"€{most_expensive_cost:.1f}M")
        
        with col3:
            cost_range = most_expensive_cost - cheapest_cost
            st.metric("Cost Range", f"€{cost_range:.1f}M")
        
        with col4:
            avg_cost = costs_df['Annual Cost (€M)'].mean()
            st.metric("Average Cost", f"€{avg_cost:.1f}M")
        
        # Download results
        st.subheader("Download Results")
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download
            csv = costs_df.to_csv(index=False)
            st.download_button(
                label="Download Cost Analysis (CSV)",
                data=csv,
                file_name=f"datacenter_costs_{it_load_mw}MW_{int(utilization_rate*100)}pct_PUE{pue}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Create a summary report
            summary_text = f"""
Data Center Cost Analysis Summary
================================

Configuration:
- IT Load: {it_load_mw} MW
- Utilization Rate: {utilization_rate*100:.0f}%
- PUE: {pue}
- Average IT Load: {efficiency_metrics['Average IT Load (MW)']} MW
- Total Facility Power: {efficiency_metrics['Total Facility Power (MW)']} MW
- Annual Consumption: {efficiency_metrics['Annual Consumption (MWh)']:,} MWh

Cost Analysis:
- Cheapest Location: {cheapest_country} (€{cheapest_cost:.1f}M/year)
- Most Expensive Location: {most_expensive_country} (€{most_expensive_cost:.1f}M/year)
- Cost Range: €{cost_range:.1f}M
- Average Cost: €{avg_cost:.1f}M/year

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            st.download_button(
                label="Download Summary Report (TXT)",
                data=summary_text,
                file_name=f"datacenter_summary_{it_load_mw}MW_{int(utilization_rate*100)}pct_PUE{pue}.txt",
                mime="text/plain"
            )
    
    # Add a clear results button if results are showing
    if st.session_state.show_datacenter_results:
        if st.button("Clear Results", type="secondary"):
            st.session_state.show_datacenter_results = False
            st.session_state.calc_params = {}
            st.rerun()

# Footer
st.markdown("---")
st.markdown("Built with Streamlit")

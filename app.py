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
tab1, tab2, tab3 = st.tabs(["📊 Price Analysis", "🏢 Data Center Cost Analysis", "🗺️ Interactive Map"])

with tab1:
    # Price Analysis
    st.header("📊 Price Analysis")
    
    # Sidebar controls for price analysis
    st.sidebar.header("Price Analysis Controls")

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
    # Data Center Cost Analysis
    st.header("🏢 Data Center Cost Analysis")
    
    # Streamlined introduction
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Calculate annual electricity costs for your data center across European countries.**
        Configure your data center parameters below to see location-based cost comparisons.
        """)
    
    with col2:
        with st.expander("📋 How it works", expanded=False):
            st.markdown("""
            **Formula:** Annual Cost = IT Load × Utilization × PUE × 8,760 hours × Electricity Price
            
            - **IT Load**: Equipment power capacity (MW)
            - **Utilization**: Active usage percentage
            - **PUE**: Power efficiency ratio (includes cooling, etc.)
            - **8,760**: Hours per year
            - **Price**: 2024 average electricity price (€/MWh)
            """)
    
    # Streamlined configuration section
    st.subheader("⚙️ Data Center Configuration")
    
    # Create a more compact configuration layout
    config_col1, config_col2, config_col3, config_col4 = st.columns([2, 2, 2, 1])
    
    with config_col1:
        it_load_options = [50, 100, 150, 200]
        default_it_load = st.session_state.calc_params.get('it_load_mw', 100)
        it_load_index = it_load_options.index(default_it_load) if default_it_load in it_load_options else 1
        it_load_mw = st.selectbox(
            "IT Load (MW)",
            it_load_options,
            index=it_load_index,
            help="Total IT equipment power capacity"
        )
    
    with config_col2:
        utilization_options = [0.5, 0.6, 0.7, 0.8]
        default_utilization = st.session_state.calc_params.get('utilization_rate', 0.7)
        util_index = utilization_options.index(default_utilization) if default_utilization in utilization_options else 2
        utilization_rate = st.selectbox(
            "Utilization Rate",
            utilization_options,
            index=util_index,
            format_func=lambda x: f"{x*100:.0f}%",
            help="Percentage of IT load actively used"
        )
    
    with config_col3:
        pue_options = [1.1, 1.2, 1.3, 1.4, 1.5]
        default_pue = st.session_state.calc_params.get('pue', 1.2)
        pue_index = pue_options.index(default_pue) if default_pue in pue_options else 1
        pue = st.selectbox(
            "PUE",
            pue_options,
            index=pue_index,
            help="Power Usage Effectiveness (lower = better)"
        )
    
    with config_col4:
        st.markdown("")  # Spacer
        if st.button("🔄 Calculate", type="primary"):
            st.session_state.calc_params = {
                'it_load_mw': it_load_mw,
                'utilization_rate': utilization_rate,
                'pue': pue
            }
            st.session_state.show_datacenter_results = True
            st.rerun()
    
    # Auto-update when parameters change
    current_params = {
        'it_load_mw': it_load_mw,
        'utilization_rate': utilization_rate,
        'pue': pue
    }
    
    if current_params != st.session_state.calc_params:
        st.session_state.calc_params = current_params
        st.session_state.show_datacenter_results = True
        st.rerun()
    
    # Display results if available
    if st.session_state.show_datacenter_results and 'calc_params' in st.session_state:
        params = st.session_state.calc_params
        it_load_mw = params['it_load_mw']
        utilization_rate = params['utilization_rate']
        pue = params['pue']
        
        # Calculate metrics
        efficiency_metrics = calculate_datacenter_efficiency_metrics(it_load_mw, utilization_rate, pue)
        costs_df = calculate_datacenter_costs(df, it_load_mw, utilization_rate, pue)
        price_column = [col for col in costs_df.columns if 'Price' in col][0]
        
        # Key metrics at the top
        st.subheader("📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Power", f"{efficiency_metrics['Total Facility Power (MW)']} MW")
        with col2:
            st.metric("Annual Consumption", f"{efficiency_metrics['Annual Consumption (MWh)']:,} MWh")
        with col3:
            cheapest_country = costs_df.iloc[0]['Country']
            cheapest_cost = costs_df.iloc[0]['Annual Cost (€M)']
            st.metric("Cheapest Location", cheapest_country, f"€{cheapest_cost:.1f}M")
        with col4:
            most_expensive_country = costs_df.iloc[-1]['Country']
            most_expensive_cost = costs_df.iloc[-1]['Annual Cost (€M)']
            st.metric("Most Expensive", most_expensive_country, f"€{most_expensive_cost:.1f}M")
        
        # Main results in tabs for better organization
        tab_results, tab_comparison, tab_insights = st.tabs(["📈 All Countries", "🔍 Compare Countries", "💡 Insights"])
        
        with tab_results:
            st.subheader("Annual Electricity Costs by Country")
            
            # Compact table with better formatting
            display_costs_df = costs_df.copy()
            display_costs_df['Annual Cost (€M)'] = display_costs_df['Annual Cost (€M)'].apply(lambda x: f"€{x:.1f}M")
            display_costs_df['Price (€/MWh)'] = display_costs_df[price_column].apply(lambda x: f"€{x:.2f}")
            
            st.dataframe(
                display_costs_df[['Country', 'Price (€/MWh)', 'Annual Cost (€M)']],
                width='stretch',
                hide_index=True
            )
            
            # Chart
            fig_bar = px.bar(
                costs_df,
                x='Annual Cost (€M)',
                y='Country',
                orientation='h',
                title="Cost Comparison by Country",
                color='Annual Cost (€M)',
                color_continuous_scale=[[0, 'green'], [0.5, 'gold'], [1, 'red']],
                labels={'Annual Cost (€M)': 'Annual Cost (€M)', 'Country': 'Country'}
            )
            
            fig_bar.update_layout(
                height=max(400, len(costs_df) * 25),
                showlegend=False,
                plot_bgcolor='rgba(248,249,250,1)',
                paper_bgcolor='rgba(255,255,255,1)',
                font=dict(size=11, family="Arial, sans-serif"),
                title_font_size=16,
                title_x=0.0,
                margin=dict(l=20, r=20, t=60, b=20)
            )
            
            fig_bar.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(200,200,200,0.3)',
                title="Annual Cost (€M)",
                title_font_size=14,
                tickfont_size=11
            )
            
            fig_bar.update_yaxes(
                showgrid=False,
                title="",
                tickfont_size=10,
                categoryorder='total descending'
            )
            
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
            
            st.plotly_chart(fig_bar, width='stretch')
        
        with tab_comparison:
            st.subheader("🔍 Compare Specific Countries")
            
            # Country selection with modern styling
            st.markdown("**Select up to 3 countries to compare:**")
            available_countries = costs_df['Country'].tolist()
            default_countries = ['Finland', 'United Kingdom', 'Spain']
            default_indices = []
            
            for country in default_countries:
                if country in available_countries:
                    default_indices.append(available_countries.index(country))
                else:
                    default_indices.append(0)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                country1 = st.selectbox("🌍 Country 1", available_countries, index=default_indices[0], key="country1")
            with col2:
                country2 = st.selectbox("🌍 Country 2", available_countries, index=default_indices[1], key="country2")
            with col3:
                country3 = st.selectbox("🌍 Country 3", available_countries, index=default_indices[2], key="country3")
            
            # Filter and display comparison
            selected_countries_data = costs_df[costs_df['Country'].isin([country1, country2, country3])].copy()
            
            if len(selected_countries_data) > 0:
                # Create comparison metrics
                comparison_data = []
                
                # Calculate price-based ranking
                df_2024 = df[df['Date'].dt.year == 2024]
                if len(df_2024) > 0:
                    price_rankings = df_2024.groupby('Country')['price_eur_mwh'].mean().reset_index()
                    price_rankings = price_rankings.sort_values('price_eur_mwh', ascending=True).reset_index(drop=True)
                else:
                    price_rankings = df.groupby('Country')['price_eur_mwh'].last().reset_index()
                    price_rankings = price_rankings.sort_values('price_eur_mwh', ascending=True).reset_index(drop=True)
                
                for _, row in selected_countries_data.iterrows():
                    country = row['Country']
                    price = row[price_column]
                    annual_cost = row['Annual Cost (€M)']
                    
                    rank = price_rankings[price_rankings['Country'] == country].index[0] + 1
                    
                    comparison_data.append({
                        'Country': country,
                        'Rank': f"#{rank}",
                        'Price (€/MWh)': price,
                        'Annual Cost (€M)': annual_cost,
                        'vs Cheapest': f"{((annual_cost / selected_countries_data['Annual Cost (€M)'].min()) - 1) * 100:.1f}%"
                    })
                
                comparison_df = pd.DataFrame(comparison_data).sort_values('Annual Cost (€M)')
                
                # Modern country cards
                st.subheader("📊 Country Comparison Cards")
                
                # Create country cards in a grid
                card_cols = st.columns(len(comparison_df))
                
                for i, (_, row) in enumerate(comparison_df.iterrows()):
                    with card_cols[i]:
                        country = row['Country']
                        rank = row['Rank']
                        price = row['Price (€/MWh)']
                        annual_cost = row['Annual Cost (€M)']
                        vs_cheapest = row['vs Cheapest']
                        
                        # Determine card color based on cost
                        if i == 0:  # Cheapest
                            card_color = "success"
                            card_icon = "🏆"
                            card_title = f"{card_icon} Best Value"
                        elif i == len(comparison_df) - 1:  # Most expensive
                            card_color = "error"
                            card_icon = "💸"
                            card_title = f"{card_icon} Premium"
                        else:  # Middle
                            card_color = "warning"
                            card_icon = "⚖️"
                            card_title = f"{card_icon} Balanced"
                        
                        # Create the card
                        if card_color == "success":
                            st.success(f"""
                            **{card_title}**
                            
                            **{country}** {rank}
                            
                            **€{price:.2f}/MWh**
                            **€{annual_cost:.1f}M/year**
                            
                            {vs_cheapest} vs cheapest
                            """)
                        elif card_color == "error":
                            st.error(f"""
                            **{card_title}**
                            
                            **{country}** {rank}
                            
                            **€{price:.2f}/MWh**
                            **€{annual_cost:.1f}M/year**
                            
                            {vs_cheapest} vs cheapest
                            """)
                        else:
                            st.warning(f"""
                            **{card_title}**
                            
                            **{country}** {rank}
                            
                            **€{price:.2f}/MWh**
                            **€{annual_cost:.1f}M/year**
                            
                            {vs_cheapest} vs cheapest
                            """)
                
                # Visual comparison chart
                st.subheader("📈 Visual Comparison")
                
                # Create a modern comparison chart
                fig_comparison = px.bar(
                    comparison_df,
                    x='Country',
                    y='Annual Cost (€M)',
                    color='Annual Cost (€M)',
                    color_continuous_scale=[[0, '#2E8B57'], [0.5, '#FFD700'], [1, '#DC143C']],
                    title="Annual Cost Comparison",
                    labels={'Annual Cost (€M)': 'Annual Cost (€M)', 'Country': 'Country'},
                    text='Annual Cost (€M)'
                )
                
                # Update layout for modern look
                fig_comparison.update_layout(
                    height=400,
                    showlegend=False,
                    plot_bgcolor='rgba(248,249,250,1)',
                    paper_bgcolor='rgba(255,255,255,1)',
                    font=dict(size=12, family="Arial, sans-serif"),
                    title_font_size=18,
                    title_x=0.0,
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                
                # Update axes
                fig_comparison.update_xaxes(
                    title="Country",
                    title_font_size=14,
                    tickfont_size=12,
                    showgrid=False
                )
                
                fig_comparison.update_yaxes(
                    title="Annual Cost (€M)",
                    title_font_size=14,
                    tickfont_size=12,
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(200,200,200,0.3)'
                )
                
                # Add value labels on bars
                fig_comparison.update_traces(
                    texttemplate='€%{y:.1f}M',
                    textposition='outside'
                )
                
                st.plotly_chart(fig_comparison, width='stretch')
                
                # Transmission charges section with modern cards
                st.subheader("⚡ Transmission Charges Analysis")
                st.markdown("**Transmission charges** are additional costs for high-voltage power delivery (20% of wholesale price).")
                
                transmission_multiplier = 0.2
                transmission_data = []
                
                for _, row in comparison_df.iterrows():
                    country = row['Country']
                    wholesale_price = row['Price (€/MWh)']
                    transmission_charge = wholesale_price * transmission_multiplier
                    total_price = wholesale_price + transmission_charge
                    
                    annual_mwh = efficiency_metrics['Annual Consumption (MWh)']
                    annual_cost_with_transmission = annual_mwh * total_price / 1_000_000
                    
                    transmission_data.append({
                        'Country': country,
                        'Wholesale Price (€/MWh)': round(wholesale_price, 2),
                        'Transmission Charge (€/MWh)': round(transmission_charge, 2),
                        'Total Price (€/MWh)': round(total_price, 2),
                        'Annual Cost with Transmission (€M)': round(annual_cost_with_transmission, 2)
                    })
                
                transmission_df = pd.DataFrame(transmission_data)
                
                # Modern transmission cards
                st.subheader("🔌 Transmission Cost Breakdown")
                
                trans_cols = st.columns(len(transmission_df))
                
                for i, (_, row) in enumerate(transmission_df.iterrows()):
                    with trans_cols[i]:
                        country = row['Country']
                        wholesale_price = row['Wholesale Price (€/MWh)']
                        transmission_charge = row['Transmission Charge (€/MWh)']
                        total_price = row['Total Price (€/MWh)']
                        annual_cost = row['Annual Cost with Transmission (€M)']
                        
                        # Calculate percentage increase
                        price_increase_pct = (transmission_charge / wholesale_price) * 100
                        
                        st.info(f"""
                        **⚡ {country}**
                        
                        **Wholesale**: €{wholesale_price:.2f}/MWh
                        **Transmission**: €{transmission_charge:.2f}/MWh (+{price_increase_pct:.0f}%)
                        **Total**: €{total_price:.2f}/MWh
                        
                        **Annual Cost**: €{annual_cost:.1f}M
                        """)
                
                # Key insights with modern metrics
                st.subheader("💡 Key Insights")
                cheapest = comparison_df.iloc[0]
                most_expensive = comparison_df.iloc[-1]
                cost_difference = most_expensive['Annual Cost (€M)'] - cheapest['Annual Cost (€M)']
                
                cheapest_transmission = transmission_df[transmission_df['Country'] == cheapest['Country']].iloc[0]
                most_expensive_transmission = transmission_df[transmission_df['Country'] == most_expensive['Country']].iloc[0]
                
                # Modern metrics with better styling
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "🏆 Cheapest (Wholesale)", 
                        f"€{cheapest['Annual Cost (€M)']:.1f}M",
                        delta=f"With transmission: €{cheapest_transmission['Annual Cost with Transmission (€M)']:.1f}M"
                    )
                    st.caption(f"**{cheapest['Country']}**")
                
                with col2:
                    st.metric(
                        "💸 Most Expensive (Wholesale)", 
                        f"€{most_expensive['Annual Cost (€M)']:.1f}M",
                        delta=f"With transmission: €{most_expensive_transmission['Annual Cost with Transmission (€M)']:.1f}M"
                    )
                    st.caption(f"**{most_expensive['Country']}**")
                
                with col3:
                    transmission_cost_diff = most_expensive_transmission['Annual Cost with Transmission (€M)'] - cheapest_transmission['Annual Cost with Transmission (€M)']
                    st.metric(
                        "📊 Cost Difference", 
                        f"€{cost_difference:.1f}M", 
                        delta=f"{((most_expensive['Annual Cost (€M)'] / cheapest['Annual Cost (€M)']) - 1) * 100:.1f}% more"
                    )
                    st.caption(f"With transmission: €{transmission_cost_diff:.1f}M difference")
                
                # Summary insights
                st.subheader("📋 Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"""
                    **🎯 Best Choice: {cheapest['Country']}**
                    
                    - Lowest electricity costs
                    - €{cheapest['Annual Cost (€M)']:.1f}M/year (wholesale)
                    - €{cheapest_transmission['Annual Cost with Transmission (€M)']:.1f}M/year (with transmission)
                    - Global rank: {cheapest['Rank']}
                    """)
                
                with col2:
                    st.warning(f"""
                    **⚠️ Cost Impact**
                    
                    - Location choice matters: €{cost_difference:.1f}M difference
                    - Transmission adds ~20% to costs
                    - Total range: €{cheapest_transmission['Annual Cost with Transmission (€M)']:.1f}M - €{most_expensive_transmission['Annual Cost with Transmission (€M)']:.1f}M
                    """)
        
        with tab_insights:
            st.subheader("💡 Key Insights & Recommendations")
            
            # Calculate insights
            cheapest_country = costs_df.iloc[0]['Country']
            cheapest_cost = costs_df.iloc[0]['Annual Cost (€M)']
            most_expensive_country = costs_df.iloc[-1]['Country']
            most_expensive_cost = costs_df.iloc[-1]['Annual Cost (€M)']
            cost_range = most_expensive_cost - cheapest_cost
            avg_cost = costs_df['Annual Cost (€M)'].mean()
            
            # Top insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **🏆 Best Value Location**
                
                **{cheapest_country}** offers the lowest electricity costs at **€{cheapest_cost:.1f}M/year**.
                
                This represents a **{((most_expensive_cost / cheapest_cost) - 1) * 100:.0f}%** savings compared to the most expensive location.
                """)
            
            with col2:
                st.warning(f"""
                **⚠️ Cost Impact**
                
                Location choice can impact your annual electricity costs by **€{cost_range:.1f}M**.
                
                The average cost across all countries is **€{avg_cost:.1f}M/year**.
                """)
            
            # Cost categories
            st.subheader("Cost Categories")
            
            # Calculate price-based ranking for cost categories
            df_2024 = df[df['Date'].dt.year == 2024]
            if len(df_2024) > 0:
                price_rankings = df_2024.groupby('Country')['price_eur_mwh'].mean().reset_index()
                price_rankings = price_rankings.sort_values('price_eur_mwh', ascending=True).reset_index(drop=True)
            else:
                price_rankings = df.groupby('Country')['price_eur_mwh'].last().reset_index()
                price_rankings = price_rankings.sort_values('price_eur_mwh', ascending=True).reset_index(drop=True)
            
            # Categorize countries
            low_cost_countries = []
            medium_cost_countries = []
            high_cost_countries = []
            
            for _, row in costs_df.iterrows():
                country = row['Country']
                rank = price_rankings[price_rankings['Country'] == country].index[0] + 1
                total_countries = len(price_rankings)
                
                if rank <= total_countries * 0.33:
                    low_cost_countries.append(country)
                elif rank <= total_countries * 0.67:
                    medium_cost_countries.append(country)
                else:
                    high_cost_countries.append(country)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.success(f"**🟢 Low Cost Countries**\n\n{', '.join(low_cost_countries[:5])}")
                if len(low_cost_countries) > 5:
                    st.caption(f"+ {len(low_cost_countries) - 5} more")
            
            with col2:
                st.warning(f"**🟡 Medium Cost Countries**\n\n{', '.join(medium_cost_countries[:5])}")
                if len(medium_cost_countries) > 5:
                    st.caption(f"+ {len(medium_cost_countries) - 5} more")
            
            with col3:
                st.error(f"**🔴 High Cost Countries**\n\n{', '.join(high_cost_countries[:5])}")
                if len(high_cost_countries) > 5:
                    st.caption(f"+ {len(high_cost_countries) - 5} more")
            
            # Recommendations
            st.subheader("📋 Recommendations")
            
            st.markdown(f"""
            **For your {it_load_mw}MW data center with {utilization_rate*100:.0f}% utilization:**
            
            1. **Primary Recommendation**: Consider **{cheapest_country}** for the lowest operational costs
            2. **Cost Range**: Budget between €{cheapest_cost:.1f}M - €{most_expensive_cost:.1f}M annually
            3. **Transmission Impact**: Add ~20% to wholesale prices for transmission charges
            4. **PUE Impact**: Your PUE of {pue} is {'excellent' if pue <= 1.2 else 'good' if pue <= 1.3 else 'average'}
            5. **Utilization Impact**: {utilization_rate*100:.0f}% utilization is {'high' if utilization_rate >= 0.8 else 'good' if utilization_rate >= 0.7 else 'moderate'}
            """)
        
        # Download section
        st.subheader("📥 Download Results")
        col1, col2 = st.columns(2)
        
        with col1:
            csv = costs_df.to_csv(index=False)
            st.download_button(
                label="Download Cost Analysis (CSV)",
                data=csv,
                file_name=f"datacenter_costs_{it_load_mw}MW_{int(utilization_rate*100)}pct_PUE{pue}.csv",
                mime="text/csv"
            )
        
        with col2:
            summary_text = f"""
Data Center Cost Analysis Summary
================================

Configuration:
- IT Load: {it_load_mw} MW
- Utilization Rate: {utilization_rate*100:.0f}%
- PUE: {pue}
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
    
    # Clear results button
    if st.session_state.show_datacenter_results:
        if st.button("🗑️ Clear Results", type="secondary"):
            st.session_state.show_datacenter_results = False
            st.session_state.calc_params = {}
            st.rerun()

with tab3:
    # Interactive Map
    st.header("🗺️ Interactive Map of European Electricity Prices")
    
    st.markdown("""
    **Explore electricity prices across Europe with this interactive map.** 
    Countries are color-coded by their average 2024 electricity prices, with darker colors indicating higher costs.
    """)
    
    # Calculate 2024 average prices for mapping
    df_2024 = df[df['Date'].dt.year == 2024]
    if len(df_2024) > 0:
        map_data = df_2024.groupby('Country')['price_eur_mwh'].mean().reset_index()
        map_data = map_data.sort_values('price_eur_mwh', ascending=True)
    else:
        # Fallback to latest prices if no 2024 data
        map_data = df.groupby('Country')['price_eur_mwh'].last().reset_index()
        map_data = map_data.sort_values('price_eur_mwh', ascending=True)
    
    # Add country codes for mapping (ISO 3166-1 alpha-3)
    country_codes = {
        'Austria': 'AUT',
        'Belgium': 'BEL', 
        'Bulgaria': 'BGR',
        'Croatia': 'HRV',
        'Cyprus': 'CYP',
        'Czech Republic': 'CZE',
        'Denmark': 'DNK',
        'Estonia': 'EST',
        'Finland': 'FIN',
        'France': 'FRA',
        'Germany': 'DEU',
        'Greece': 'GRC',
        'Hungary': 'HUN',
        'Ireland': 'IRL',
        'Italy': 'ITA',
        'Latvia': 'LVA',
        'Lithuania': 'LTU',
        'Luxembourg': 'LUX',
        'Malta': 'MLT',
        'Netherlands': 'NLD',
        'Poland': 'POL',
        'Portugal': 'PRT',
        'Romania': 'ROU',
        'Slovakia': 'SVK',
        'Slovenia': 'SVN',
        'Spain': 'ESP',
        'Sweden': 'SWE',
        'United Kingdom': 'GBR'
    }
    
    # Add country codes to map data
    map_data['Country_Code'] = map_data['Country'].map(country_codes)
    
    # Remove countries without country codes
    map_data = map_data.dropna(subset=['Country_Code'])
    
    # Create the choropleth map
    fig_map = px.choropleth(
        map_data,
        locations='Country_Code',
        color='price_eur_mwh',
        hover_name='Country',
        hover_data={'price_eur_mwh': ':.2f', 'Country_Code': False},
        color_continuous_scale=[[0, '#2E8B57'], [0.3, '#FFD700'], [0.7, '#FF8C00'], [1, '#DC143C']],
        title="Average 2024 Electricity Prices in Europe (€/MWh)",
        labels={'price_eur_mwh': 'Price (€/MWh)'},
        scope='europe'
    )
    
    # Update layout for better appearance
    fig_map.update_layout(
        height=600,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        ),
        title_font_size=18,
        title_x=0.5,
        font=dict(size=12, family="Arial, sans-serif"),
        margin=dict(l=0, r=0, t=60, b=0)
    )
    
    # Update colorbar
    fig_map.update_coloraxes(
        colorbar=dict(
            title="Price (€/MWh)",
            title_font_size=14,
            tickfont_size=12,
            len=0.8,
            y=0.5,
            yanchor='middle'
        )
    )
    
    # Display the map
    st.plotly_chart(fig_map, width='stretch')
    
    # Add insights below the map
    st.subheader("🗺️ Map Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cheapest_country = map_data.iloc[0]
        st.success(f"""
        **🏆 Cheapest: {cheapest_country['Country']}**
        
        €{cheapest_country['price_eur_mwh']:.2f}/MWh
        """)
    
    with col2:
        most_expensive_country = map_data.iloc[-1]
        st.error(f"""
        **💸 Most Expensive: {most_expensive_country['Country']}**
        
        €{most_expensive_country['price_eur_mwh']:.2f}/MWh
        """)
    
    with col3:
        price_range = most_expensive_country['price_eur_mwh'] - cheapest_country['price_eur_mwh']
        st.info(f"""
        **📊 Price Range**
        
        €{price_range:.2f}/MWh difference
        """)
    
    # Add price categories
    st.subheader("🎨 Price Categories")
    
    # Calculate price percentiles for categorization
    q33 = map_data['price_eur_mwh'].quantile(0.33)
    q67 = map_data['price_eur_mwh'].quantile(0.67)
    
    low_cost = map_data[map_data['price_eur_mwh'] <= q33]
    medium_cost = map_data[(map_data['price_eur_mwh'] > q33) & (map_data['price_eur_mwh'] <= q67)]
    high_cost = map_data[map_data['price_eur_mwh'] > q67]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🟢 Low Cost Countries**")
        for _, row in low_cost.iterrows():
            st.caption(f"• {row['Country']}: €{row['price_eur_mwh']:.2f}/MWh")
    
    with col2:
        st.markdown("**🟡 Medium Cost Countries**")
        for _, row in medium_cost.iterrows():
            st.caption(f"• {row['Country']}: €{row['price_eur_mwh']:.2f}/MWh")
    
    with col3:
        st.markdown("**🔴 High Cost Countries**")
        for _, row in high_cost.iterrows():
            st.caption(f"• {row['Country']}: €{row['price_eur_mwh']:.2f}/MWh")
    
    # Add interactive features
    st.subheader("🔍 Interactive Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Map Controls:**
        - **Hover** over countries to see exact prices
        - **Zoom** in/out to explore specific regions
        - **Pan** around to view different areas
        - **Color scale** shows price ranges from green (low) to red (high)
        """)
    
    with col2:
        st.markdown("""
        **Data Insights:**
        - Prices are based on 2024 monthly averages
        - Color coding uses 33rd and 67th percentiles
        - Hover data shows precise values
        - Map focuses on EU/EEA countries
        """)
    
    # Add download option for map data
    st.subheader("📥 Download Map Data")
    
    csv_map = map_data[['Country', 'price_eur_mwh']].to_csv(index=False)
    st.download_button(
        label="Download Country Prices (CSV)",
        data=csv_map,
        file_name="european_electricity_prices_2024.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("Built with Streamlit")

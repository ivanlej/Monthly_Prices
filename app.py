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

# Load and clean data
@st.cache_data
def load_data():
    df_raw = load_local_csv('european_wholesale_electricity_price_data_monthly.csv')
    return clean_df(df_raw)

df = load_data()

# Sidebar controls
st.sidebar.header("Controls")

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
    st.title("⚡ European Power Prices (Monthly)")
    
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

# Footer
st.markdown("---")
st.markdown("Built with Streamlit")

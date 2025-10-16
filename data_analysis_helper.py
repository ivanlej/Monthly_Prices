#!/usr/bin/env python3
"""
Data Analysis Helper for European Power Prices
This script helps analyze the electricity price data and verify rankings.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_and_clean_data(csv_path='european_wholesale_electricity_price_data_monthly.csv'):
    """Load and clean the electricity price data"""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'Price (EUR/MWhe)': 'price_eur_mwh'})
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    return df

def analyze_yearly_rankings(df, year=2024):
    """Analyze yearly rankings for a specific year"""
    print(f"=== {year} ELECTRICITY PRICE RANKINGS ===")
    
    # Filter data for the specific year
    df_year = df[df['Date'].dt.year == year]
    
    if len(df_year) == 0:
        print(f"No data found for year {year}")
        return None
    
    # Calculate average prices by country
    avg_prices = df_year.groupby('Country')['price_eur_mwh'].mean().reset_index()
    avg_prices = avg_prices.sort_values('price_eur_mwh', ascending=True)
    
    print(f"Total countries with data in {year}: {len(avg_prices)}")
    print(f"Date range: {df_year['Date'].min().strftime('%Y-%m-%d')} to {df_year['Date'].max().strftime('%Y-%m-%d')}")
    print()
    
    # Display rankings
    print("RANKING BY AVERAGE PRICE:")
    print("Rank | Country        | Avg Price (€/MWh) | Data Points")
    print("-----|----------------|-------------------|------------")
    
    for i, (_, row) in enumerate(avg_prices.iterrows(), 1):
        country = row['Country']
        price = row['price_eur_mwh']
        data_points = len(df_year[df_year['Country'] == country])
        print(f"{i:4d} | {country:14s} | {price:15.2f} | {data_points:11d}")
    
    return avg_prices

def simulate_datacenter_costs(df, year=2024, it_load_mw=100, utilization_rate=0.7, pue=1.2):
    """Simulate the datacenter cost calculation exactly as in the app"""
    print(f"\n=== DATACENTER COST SIMULATION ({year}) ===")
    print(f"Configuration: {it_load_mw}MW IT load, {utilization_rate*100:.0f}% utilization, PUE {pue}")
    
    # Get yearly data
    df_year = df[df['Date'].dt.year == year]
    
    if len(df_year) == 0:
        print(f"No data found for year {year}")
        return None
    
    # Calculate average prices (same as app logic)
    avg_prices = df_year.groupby('Country')['price_eur_mwh'].mean().reset_index()
    
    # Calculate datacenter costs (same as app logic)
    effective_it_load_mw = it_load_mw * utilization_rate
    total_facility_power_mw = effective_it_load_mw * pue
    annual_mwh = total_facility_power_mw * 8760
    
    costs = []
    for _, row in avg_prices.iterrows():
        country = row['Country']
        price_eur_mwh = row['price_eur_mwh']
        annual_cost_eur = annual_mwh * price_eur_mwh
        
        costs.append({
            'Country': country,
            'Price (€/MWh)': round(price_eur_mwh, 2),
            'Annual MWh': round(annual_mwh, 0),
            'Annual Cost (€)': round(annual_cost_eur, 0),
            'Annual Cost (€M)': round(annual_cost_eur / 1_000_000, 2)
        })
    
    # Convert to DataFrame and sort by cost (same as app logic)
    costs_df = pd.DataFrame(costs)
    costs_df = costs_df.sort_values('Annual Cost (€)', ascending=True)
    
    print(f"Annual consumption: {annual_mwh:,.0f} MWh")
    print()
    print("RANKING BY ANNUAL COST:")
    print("Rank | Country        | Price (€/MWh) | Annual Cost (€M)")
    print("-----|----------------|---------------|------------------")
    
    for i, (_, row) in enumerate(costs_df.iterrows(), 1):
        country = row['Country']
        price = row['Price (€/MWh)']
        cost = row['Annual Cost (€M)']
        print(f"{i:4d} | {country:14s} | {price:13.2f} | {cost:16.2f}")
    
    return costs_df

def check_specific_countries(df, year=2024, countries=['Finland', 'Spain', 'United Kingdom']):
    """Check specific countries' rankings and costs"""
    print(f"\n=== SPECIFIC COUNTRIES CHECK ({year}) ===")
    
    # Get yearly rankings
    df_year = df[df['Date'].dt.year == year]
    avg_prices = df_year.groupby('Country')['price_eur_mwh'].mean().reset_index()
    avg_prices = avg_prices.sort_values('price_eur_mwh', ascending=True)
    
    # Get datacenter costs
    costs_df = simulate_datacenter_costs(df, year)
    
    print(f"\nSelected Countries Analysis:")
    print("Country        | Price Rank | Cost Rank | Price (€/MWh) | Annual Cost (€M)")
    print("----------------|------------|-----------|---------------|------------------")
    
    for country in countries:
        if country in avg_prices['Country'].values:
            # Price ranking
            price_rank = avg_prices[avg_prices['Country'] == country].index[0] + 1
            price = avg_prices[avg_prices['Country'] == country]['price_eur_mwh'].iloc[0]
            
            # Cost ranking
            cost_rank = costs_df[costs_df['Country'] == country].index[0] + 1
            cost = costs_df[costs_df['Country'] == country]['Annual Cost (€M)'].iloc[0]
            
            print(f"{country:15s} | {price_rank:10d} | {cost_rank:9d} | {price:13.2f} | {cost:16.2f}")
        else:
            print(f"{country:15s} | Not found in {year} data")

def main():
    """Main analysis function"""
    print("European Power Prices Data Analysis Helper")
    print("=" * 50)
    
    # Load data
    df = load_and_clean_data()
    print(f"Data loaded: {len(df)} records from {df['Date'].min().year} to {df['Date'].max().year}")
    print(f"Countries: {df['Country'].nunique()}")
    print()
    
    # Analyze 2024 rankings
    yearly_rankings = analyze_yearly_rankings(df, 2024)
    
    # Simulate datacenter costs
    costs_df = simulate_datacenter_costs(df, 2024)
    
    # Check specific countries
    check_specific_countries(df, 2024, ['Finland', 'Spain', 'United Kingdom'])
    
    # Check if there are any data quality issues
    print(f"\n=== DATA QUALITY CHECK ===")
    df_2024 = df[df['Date'].dt.year == 2024]
    print(f"2024 data points: {len(df_2024)}")
    print(f"Countries in 2024: {df_2024['Country'].nunique()}")
    print(f"Date range: {df_2024['Date'].min()} to {df_2024['Date'].max()}")
    
    # Check for missing months
    monthly_counts = df_2024.groupby('Country').size()
    print(f"\nData points per country in 2024:")
    for country in ['Finland', 'Spain', 'United Kingdom']:
        if country in monthly_counts.index:
            print(f"{country}: {monthly_counts[country]} months")
        else:
            print(f"{country}: No data")

if __name__ == "__main__":
    main()


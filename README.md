# European Power Prices Dashboard

A Streamlit dashboard for visualizing European wholesale electricity price data from 2015-2024.

## Features

- **Interactive Controls**: Multi-select countries, date range slider, indexing options
- **Key Performance Indicators**: Latest month, country count, average prices, data points
- **Country Statistics**: Latest prices, 3-month and 12-month changes with data quality warnings
- **Visualizations**: 
  - Monthly price trends (with optional indexing to 100)
  - Year-over-year percentage changes
  - Annual average prices by country
- **Data Export**: Download filtered data as CSV or charts as HTML/PNG

## Data

The dashboard reads from `european_wholesale_electricity_price_data_monthly.csv` containing:
- **Country**: European country names
- **ISO3 Code**: Three-letter country codes
- **Date**: Monthly data points (first of each month)
- **Price (EUR/MWhe)**: Wholesale electricity prices in €/MWh

## Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the dashboard**:
   ```bash
   streamlit run app.py
   ```

3. **Access the dashboard**: Open http://localhost:8501 in your browser

## Deployment on Streamlit Community Cloud

### Prerequisites
- GitHub account
- Streamlit Community Cloud account (free)

### Deployment Steps

1. **Create a GitHub repository**:
   - Create a new repository on GitHub
   - Upload all three files: `app.py`, `requirements.txt`, and `european_wholesale_electricity_price_data_monthly.csv`
   - Make sure the CSV file is in the root directory

2. **Deploy on Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository
   - Set the main file path to `app.py`
   - Click "Deploy"

3. **Your dashboard will be live** at: `https://your-username-your-app-name.streamlit.app`

### Important Notes for Streamlit Community Cloud

- **File Structure**: Ensure the CSV file is in the root directory alongside `app.py`
- **Repository Access**: Make sure your repository is public (required for free Streamlit Community Cloud)
- **Dependencies**: The `requirements.txt` file will automatically install the necessary packages
- **Data Loading**: The app uses `pd.read_csv('european_wholesale_electricity_price_data_monthly.csv')` to load the local CSV file

### Troubleshooting

- **CSV not found**: Ensure the CSV file is in the root directory and has the exact filename
- **Import errors**: Check that all dependencies in `requirements.txt` are correctly specified
- **Performance**: The app uses `@st.cache_data` for optimal performance with large datasets

## File Structure

```
your-repository/
├── app.py                                          # Main Streamlit application
├── requirements.txt                                # Python dependencies
├── european_wholesale_electricity_price_data_monthly.csv  # Data file
└── README.md                                       # This file
```

## Technical Details

- **Framework**: Streamlit
- **Visualization**: Plotly Express
- **Data Processing**: Pandas
- **Caching**: Streamlit's `@st.cache_data` for performance
- **Export**: CSV, HTML, and PNG (with kaleido) downloads

Built with Streamlit ⚡

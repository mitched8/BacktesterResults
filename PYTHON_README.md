# Trading Dashboard - Python/Streamlit Version

A sophisticated financial trading dashboard built with Streamlit and AG Grid, matching the React version's functionality and styling.

## Features

- 📊 **Portfolio Holdings Grid** - Track your investments with P&L calculations
- 🌐 **Market Watch Grid** - Real-time market data for 15 tech stocks
- 📋 **Recent Orders Grid** - Trading activity and order history
- 💰 **Portfolio Summary Cards** - Total cost basis, current value, P&L, and return rate
- 🎨 **Modern UI** - Gradient backgrounds, glassmorphism effects, color-coded data

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Dashboard

Run the Streamlit app:
```bash
streamlit run trading_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Features Showcased

### AG Grid Features
- Custom cell renderers (currency, percentages, P&L with color coding)
- Value formatters
- Cell styling based on values
- Pinned columns
- Row selection (multiple and single)
- Filtering on all columns
- Sorting
- Resizable columns

### Data Generation
- Synthetic market data for 15 tech stocks
- Portfolio holdings with cost basis and P&L calculations
- Order history with various statuses

### Styling
- Gradient backgrounds and cards
- Glassmorphism effects
- Color-coded profit/loss indicators
- Responsive layout
- Professional fintech aesthetic

## Comparison with React Version

This Python/Streamlit version closely mirrors the React version with:
- Same layout and structure
- Matching color schemes and gradients
- Identical data generation logic
- Similar AG Grid configurations
- Comparable styling and effects

## Tech Stack

- **Streamlit** - Web framework
- **streamlit-aggrid** - AG Grid integration for Streamlit
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations

## Notes

- Data is randomly generated on each page load
- Uses `@st.cache_data` for performance optimization
- Custom CSS for advanced styling
- JavaScript cell renderers for AG Grid customization


import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from st_aggrid.shared import GridUpdateMode
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling - JP Morgan Professional Theme
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background: linear-gradient(180deg, #0a1929 0%, #1a2332 100%);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Header styling */
    .dashboard-header {
        background: linear-gradient(135deg, #071426 0%, #0f2744 100%);
        padding: 28px 40px;
        border-radius: 0;
        margin: -6rem -6rem 2rem -6rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        border-bottom: 2px solid #1e4976;
    }
    
    .dashboard-title {
        color: #ffffff;
        font-size: 32px;
        font-weight: 700;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
        font-family: 'Georgia', 'Times New Roman', serif;
    }
    
    .dashboard-subtitle {
        color: rgba(255,255,255,0.7);
        font-size: 14px;
        margin: 0;
        font-family: 'Arial', sans-serif;
    }
    
    .market-status {
        background: linear-gradient(135deg, #1e4976 0%, #2563ab 100%);
        padding: 12px 24px;
        border-radius: 8px;
        border: 1px solid #3b82f6;
        display: inline-block;
        float: right;
        margin-top: -50px;
        box-shadow: 0 4px 12px rgba(37, 99, 171, 0.3);
    }
    
    .market-status-label {
        color: rgba(255,255,255,0.8);
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        font-family: 'Arial', sans-serif;
    }
    
    .market-status-value {
        color: #10b981;
        font-size: 15px;
        font-weight: 700;
        font-family: 'Arial', sans-serif;
    }
    
    /* Pulse animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .pulse-dot {
        width: 7px;
        height: 7px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 2s infinite;
        margin-right: 8px;
    }
    
    /* Card styling */
    .metric-card {
        padding: 24px 28px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        border: 1px solid #2d3748;
        position: relative;
        overflow: hidden;
        margin-bottom: 20px;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #3b82f6 0%, #1e40af 100%);
    }
    
    .metric-label {
        color: rgba(255,255,255,0.7);
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
        font-family: 'Arial', sans-serif;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 28px;
        font-weight: 700;
        position: relative;
        z-index: 1;
        font-family: 'Georgia', serif;
    }
    
    /* Grid container styling */
    .grid-container {
        background: linear-gradient(135deg, #1a2332 0%, #0f1923 100%);
        border-radius: 8px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        overflow: hidden;
        border: 1px solid #2d3748;
        margin-bottom: 32px;
        padding: 0;
    }
    
    .grid-header {
        padding: 20px 28px;
        border-bottom: 2px solid #2d3748;
        background: linear-gradient(135deg, #0f1923 0%, #162031 100%);
    }
    
    .grid-title {
        color: #ffffff;
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 0px;
        margin: 0;
        font-family: 'Georgia', serif;
    }
    
    .grid-description {
        color: rgba(255,255,255,0.6);
        font-size: 12px;
        margin: 4px 0 0 0;
        font-family: 'Arial', sans-serif;
    }
    
    /* Remove streamlit padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# Generate synthetic market data
@st.cache_data
def generate_market_data():
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC', 'ORCL', 'CRM', 'ADBE', 'PYPL', 'UBER']
    companies = [
        'Apple Inc.', 'Microsoft Corp.', 'Alphabet Inc.', 'Amazon.com Inc.', 
        'Tesla Inc.', 'Meta Platforms Inc.', 'NVIDIA Corp.', 'Netflix Inc.',
        'Advanced Micro Devices', 'Intel Corp.', 'Oracle Corp.', 'Salesforce Inc.',
        'Adobe Inc.', 'PayPal Holdings', 'Uber Technologies'
    ]
    sectors = ['Technology', 'Technology', 'Technology', 'E-commerce', 'Automotive', 
               'Social Media', 'Semiconductors', 'Entertainment', 'Semiconductors', 
               'Semiconductors', 'Enterprise', 'SaaS', 'Software', 'Fintech', 'Transportation']
    
    data = []
    for symbol, company, sector in zip(symbols, companies, sectors):
        base_price = 50 + np.random.random() * 450
        change_percent = (np.random.random() - 0.5) * 10
        volume = int(np.random.random() * 50000000) + 1000000
        
        data.append({
            'Symbol': symbol,
            'Company': company,
            'Sector': sector,
            'Price': round(base_price, 2),
            'Change %': round(change_percent, 2),
            'Volume': volume,
            'Market Cap': int(base_price * volume * 0.1),
            '52W High': round(base_price * 1.3, 2),
            '52W Low': round(base_price * 0.7, 2),
            'P/E Ratio': round(10 + np.random.random() * 40, 2),
            'Dividend %': round(np.random.random() * 3, 2)
        })
    
    return pd.DataFrame(data)

# Generate portfolio data
@st.cache_data
def generate_portfolio_data(market_data):
    portfolio = []
    for _, stock in market_data.head(8).iterrows():
        shares = int(np.random.random() * 100) + 10
        avg_price = stock['Price'] * (0.85 + np.random.random() * 0.3)
        current_value = shares * stock['Price']
        cost_basis = shares * avg_price
        pnl = current_value - cost_basis
        pnl_percent = (pnl / cost_basis) * 100
        
        portfolio.append({
            'Symbol': stock['Symbol'],
            'Company': stock['Company'],
            'Shares': shares,
            'Avg Price': round(avg_price, 2),
            'Current Price': stock['Price'],
            'Cost Basis': round(cost_basis, 2),
            'Current Value': round(current_value, 2),
            'P&L': round(pnl, 2),
            'P&L %': round(pnl_percent, 2),
            'Allocation %': round(np.random.random() * 20 + 5, 2)
        })
    
    return pd.DataFrame(portfolio)

# Generate orders data
@st.cache_data
def generate_orders_data(market_data):
    order_types = ['BUY', 'SELL']
    statuses = ['FILLED', 'PARTIAL', 'PENDING']
    
    orders = []
    for idx, (_, stock) in enumerate(market_data.head(10).iterrows()):
        shares = int(np.random.random() * 50) + 5
        order_price = stock['Price'] * (0.98 + np.random.random() * 0.04)
        order_type = random.choice(order_types)
        status = random.choice(statuses)
        timestamp = datetime.now() - timedelta(days=np.random.random() * 7)
        
        orders.append({
            'Order ID': f'ORD-{1000 + idx}',
            'Symbol': stock['Symbol'],
            'Type': order_type,
            'Shares': shares,
            'Price': round(order_price, 2),
            'Total': round(shares * order_price, 2),
            'Status': status,
            'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return pd.DataFrame(orders)

# Generate historical price data for charts
@st.cache_data
def generate_price_history(symbol, current_price, days=90):
    """Generate synthetic historical price data for a stock"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate random walk for price
    returns = np.random.normal(0.001, 0.02, days)
    prices = [current_price * 0.85]  # Start 15% lower
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Adjust to end at current price
    adjustment = current_price / prices[-1]
    prices = [p * adjustment for p in prices]
    
    # Generate OHLC data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else close * 0.99
        volume = int(np.random.uniform(1000000, 10000000))
        
        data.append({
            'Date': date,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
    
    return pd.DataFrame(data)

# Create sophisticated price chart
def create_price_chart(symbol, company, price_data, current_price, change_percent):
    """Create a sophisticated candlestick chart with volume"""
    
    # Create subplot with secondary y-axis for volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('', '')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=price_data['Date'],
            open=price_data['Open'],
            high=price_data['High'],
            low=price_data['Low'],
            close=price_data['Close'],
            name='Price',
            increasing_line_color='#10b981',
            decreasing_line_color='#ef4444',
            increasing_fillcolor='rgba(16, 185, 129, 0.3)',
            decreasing_fillcolor='rgba(239, 68, 68, 0.3)',
        ),
        row=1, col=1
    )
    
    # Add moving average
    ma_20 = price_data['Close'].rolling(window=20).mean()
    fig.add_trace(
        go.Scatter(
            x=price_data['Date'],
            y=ma_20,
            name='20-Day MA',
            line=dict(color='rgba(102, 126, 234, 0.8)', width=2),
        ),
        row=1, col=1
    )
    
    # Volume bars
    colors = ['#10b981' if price_data['Close'].iloc[i] >= price_data['Open'].iloc[i] 
              else '#ef4444' for i in range(len(price_data))]
    
    fig.add_trace(
        go.Bar(
            x=price_data['Date'],
            y=price_data['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.5,
        ),
        row=2, col=1
    )
    
    # Update layout with sophisticated styling
    change_color = '#10b981' if change_percent >= 0 else '#ef4444'
    change_sign = '+' if change_percent >= 0 else ''
    
    fig.update_layout(
        title={
            'text': f'{symbol} - {company}<br><sub style="color: {change_color}; font-size: 16px;">${current_price:.2f} ({change_sign}{change_percent:.2f}%)</sub>',
            'x': 0,
            'xanchor': 'left',
            'font': {'size': 22, 'color': '#ffffff', 'family': 'Georgia, serif'}
        },
        xaxis_rangeslider_visible=False,
        height=600,
        hovermode='x unified',
        plot_bgcolor='rgba(15, 25, 35, 0.3)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(family='Arial, sans-serif', color='#e5e7eb'),
        margin=dict(l=60, r=40, t=80, b=40),
        xaxis2=dict(
            showgrid=True,
            gridcolor='rgba(55, 65, 81, 0.3)',
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(55, 65, 81, 0.3)',
            tickformat='$,.2f',
            color='#e5e7eb'
        ),
        yaxis2=dict(
            showgrid=False,
            tickformat='.2s',
            color='#e5e7eb'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(15, 25, 35, 0.9)',
            bordercolor='rgba(55, 65, 81, 0.5)',
            borderwidth=1,
            font=dict(color='#e5e7eb')
        ),
        hoverlabel=dict(
            bgcolor='rgba(15, 25, 35, 0.95)',
            font_size=13,
            font_family='Arial, sans-serif',
            font_color='white',
            bordercolor='#3b82f6'
        )
    )
    
    # Update xaxis properties
    fig.update_xaxes(
        showgrid=True,
        gridcolor='rgba(55, 65, 81, 0.3)',
        color='#e5e7eb'
    )
    
    return fig

# Create portfolio allocation pie chart
def create_allocation_chart(portfolio_data):
    """Create a sophisticated donut chart for portfolio allocation"""
    
    fig = go.Figure(data=[go.Pie(
        labels=portfolio_data['Symbol'],
        values=portfolio_data['Current Value'],
        hole=0.6,
        marker=dict(
            colors=['#1e4976', '#2563ab', '#3b82f6', '#60a5fa', '#0d5a3d', '#10b981', '#1e3a52', '#334155'],
            line=dict(color='#0f1923', width=2)
        ),
        textposition='outside',
        textinfo='label+percent',
        textfont=dict(size=12, family='Arial, sans-serif', color='#e5e7eb'),
        hovertemplate='<b>%{label}</b><br>' +
                      'Value: $%{value:,.2f}<br>' +
                      'Percent: %{percent}<br>' +
                      '<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': 'Portfolio Allocation',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#ffffff', 'family': 'Georgia, serif'}
        },
        height=450,
        plot_bgcolor='rgba(15, 25, 35, 0.3)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(family='Arial, sans-serif', color='#e5e7eb'),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.1,
            bgcolor='rgba(15, 25, 35, 0.9)',
            bordercolor='rgba(55, 65, 81, 0.5)',
            borderwidth=1,
            font=dict(color='#e5e7eb')
        ),
        hoverlabel=dict(
            bgcolor='rgba(15, 25, 35, 0.95)',
            font_size=13,
            font_family='Arial, sans-serif',
            font_color='white',
            bordercolor='#3b82f6'
        ),
        margin=dict(l=40, r=160, t=80, b=40)
    )
    
    # Add annotation in the center
    total_value = portfolio_data['Current Value'].sum()
    fig.add_annotation(
        text=f'<b>Total</b><br>${total_value:,.0f}',
        x=0.5, y=0.5,
        font=dict(size=18, family='Georgia, serif', color='#ffffff'),
        showarrow=False
    )
    
    return fig

# Create P&L trend chart
def create_pnl_trend_chart(portfolio_data):
    """Create a bar chart showing P&L by stock"""
    
    # Sort by P&L
    df_sorted = portfolio_data.sort_values('P&L', ascending=True)
    
    colors = ['#10b981' if val >= 0 else '#ef4444' for val in df_sorted['P&L']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=df_sorted['P&L'],
            y=df_sorted['Symbol'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(15, 25, 35, 0.8)', width=1)
            ),
            text=[f'${val:,.0f}' for val in df_sorted['P&L']],
            textposition='outside',
            textfont=dict(size=11, family='Arial, sans-serif', color='#e5e7eb'),
            hovertemplate='<b>%{y}</b><br>' +
                          'P&L: $%{x:,.2f}<br>' +
                          '<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Profit & Loss by Position',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#ffffff', 'family': 'Georgia, serif'}
        },
        height=450,
        plot_bgcolor='rgba(15, 25, 35, 0.3)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(family='Arial, sans-serif', color='#e5e7eb'),
        xaxis=dict(
            title='Profit / Loss ($)',
            showgrid=True,
            gridcolor='rgba(55, 65, 81, 0.3)',
            zeroline=True,
            zerolinecolor='rgba(107, 114, 128, 0.5)',
            zerolinewidth=2,
            tickformat='$,.0f',
            color='#e5e7eb'
        ),
        yaxis=dict(
            title='',
            showgrid=False,
            color='#e5e7eb'
        ),
        hoverlabel=dict(
            bgcolor='rgba(15, 25, 35, 0.95)',
            font_size=13,
            font_family='Arial, sans-serif',
            font_color='white',
            bordercolor='#3b82f6'
        ),
        margin=dict(l=60, r=100, t=80, b=60)
    )
    
    return fig

# Custom cell style functions for dynamic coloring
def get_pnl_style(params):
    if params['value'] >= 0:
        return {'color': '#10b981', 'fontWeight': 'bold'}
    return {'color': '#ef4444', 'fontWeight': 'bold'}

def get_order_type_style(params):
    if params['value'] == 'BUY':
        return {'color': '#10b981', 'fontWeight': 'bold'}
    return {'color': '#ef4444', 'fontWeight': 'bold'}

def get_order_status_style(params):
    colors = {
        'FILLED': '#10b981',
        'PARTIAL': '#f59e0b',
        'PENDING': '#6b7280'
    }
    return {'color': colors.get(params['value'], '#000'), 'fontWeight': 'bold'}

# Main app
def main():
    # Header
    st.markdown("""
        <div class="dashboard-header">
            <h1 class="dashboard-title">Trading Dashboard</h1>
            <p class="dashboard-subtitle">Real-time market data & portfolio analytics</p>
            <div class="market-status">
                <div class="market-status-label">Live Market Status</div>
                <div class="market-status-value">
                    <span class="pulse-dot"></span>MARKETS OPEN
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Generate data
    market_data = generate_market_data()
    portfolio_data = generate_portfolio_data(market_data)
    orders_data = generate_orders_data(market_data)
    
    # Calculate portfolio totals
    total_cost_basis = portfolio_data['Cost Basis'].sum()
    total_current_value = portfolio_data['Current Value'].sum()
    total_pnl = total_current_value - total_cost_basis
    total_pnl_percent = (total_pnl / total_cost_basis) * 100
    
    # Portfolio Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #1e3a52 0%, #0f2744 100%);">
                <div class="metric-label">Cost Basis</div>
                <div class="metric-value">${total_cost_basis:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #1e4976 0%, #0f2744 100%);">
                <div class="metric-label">Current Value</div>
                <div class="metric-value">${total_current_value:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        gradient = "linear-gradient(135deg, #0d5a3d 0%, #0a4530 100%)" if total_pnl >= 0 else "linear-gradient(135deg, #7f1d1d 0%, #5c1010 100%)"
        sign = "+" if total_pnl >= 0 else ""
        st.markdown(f"""
            <div class="metric-card" style="background: {gradient};">
                <div class="metric-label">Total P&L</div>
                <div class="metric-value">{sign}${total_pnl:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        gradient = "linear-gradient(135deg, #0d5a3d 0%, #0a4530 100%)" if total_pnl_percent >= 0 else "linear-gradient(135deg, #7f1d1d 0%, #5c1010 100%)"
        sign = "+" if total_pnl_percent >= 0 else ""
        st.markdown(f"""
            <div class="metric-card" style="background: {gradient};">
                <div class="metric-label">Return Rate</div>
                <div class="metric-value">{sign}{total_pnl_percent:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Portfolio Holdings Grid
    st.markdown("""
        <div class="grid-container">
            <div class="grid-header">
                <h2 class="grid-title">Portfolio Holdings</h2>
                <p class="grid-description">Track your investments and performance</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    gb_portfolio = GridOptionsBuilder.from_dataframe(portfolio_data)
    gb_portfolio.configure_default_column(
        resizable=True,
        filterable=True,
        sortable=True,
        editable=False
    )
    gb_portfolio.configure_column("Symbol", pinned='left', width=100)
    gb_portfolio.configure_column("Company", width=200)
    gb_portfolio.configure_column("Shares", width=100, type=['numericColumn'], cellStyle={'textAlign': 'right'})
    gb_portfolio.configure_column("Avg Price", width=120, type=['numericColumn', 'numberColumnFilter'], valueFormatter="'$' + value.toFixed(2)", cellStyle={'textAlign': 'right'})
    gb_portfolio.configure_column("Current Price", width=130, type=['numericColumn', 'numberColumnFilter'], valueFormatter="'$' + value.toFixed(2)", cellStyle={'textAlign': 'right'})
    gb_portfolio.configure_column("Cost Basis", width=130, type=['numericColumn', 'numberColumnFilter'], valueFormatter="'$' + value.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})", cellStyle={'textAlign': 'right'})
    gb_portfolio.configure_column("Current Value", width=140, type=['numericColumn', 'numberColumnFilter'], valueFormatter="'$' + value.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})", cellStyle={'textAlign': 'right'})
    gb_portfolio.configure_column("P&L", width=130, type=['numericColumn'], valueFormatter="(value >= 0 ? '+$' : '-$') + Math.abs(value).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})", cellStyle={'textAlign': 'right', 'color': 'x.value >= 0 ? "#10b981" : "#ef4444"', 'fontWeight': 'bold'})
    gb_portfolio.configure_column("P&L %", width=100, type=['numericColumn'], valueFormatter="(value >= 0 ? '+' : '') + value.toFixed(2) + '%'", cellStyle={'textAlign': 'right', 'color': 'x.value >= 0 ? "#10b981" : "#ef4444"', 'fontWeight': 'bold'})
    gb_portfolio.configure_column("Allocation %", width=130, valueFormatter="value + '%'", cellStyle={'textAlign': 'right'})
    gb_portfolio.configure_selection(selection_mode='single', use_checkbox=False)
    
    gridOptions_portfolio = gb_portfolio.build()
    
    portfolio_response = AgGrid(
        portfolio_data,
        gridOptions=gridOptions_portfolio,
        height=400,
        theme='alpine',
        allow_unsafe_jscode=True,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        enable_enterprise_modules=False,
        custom_css={
            ".ag-theme-alpine": {
                "border-radius": "0 0 16px 16px"
            }
        }
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Interactive Charts Section
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="color: #ffffff; font-size: 22px; font-weight: 700; margin-bottom: 8px; font-family: 'Georgia', serif;">
                Analytics & Charts
            </h2>
            <p style="color: rgba(255,255,255,0.6); font-size: 13px; margin: 0; font-family: 'Arial', sans-serif;">
                Click on a stock in the Portfolio Holdings grid above to view detailed charts
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Check if a row is selected
    selected_rows = portfolio_response.get('selected_rows', None)
    
    # Handle both DataFrame and list responses
    if selected_rows is not None:
        if hasattr(selected_rows, 'to_dict'):
            # It's a DataFrame, convert to list of dicts
            selected_rows = selected_rows.to_dict('records')
        
        if isinstance(selected_rows, list) and len(selected_rows) > 0:
            selected_stock = selected_rows[0]
            symbol = selected_stock['Symbol']
            company = selected_stock['Company']
            current_price = selected_stock['Current Price']
            change_percent = selected_stock['P&L %']
            
            # Generate price history for selected stock
            price_history = generate_price_history(symbol, current_price)
            
            # Create chart container
            st.markdown("""
                <div class="grid-container">
                    <div class="grid-header">
                        <h2 class="grid-title">Price Chart</h2>
                        <p class="grid-description">90-day price history with volume</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Display price chart
            chart = create_price_chart(symbol, company, price_history, current_price, change_percent)
            st.plotly_chart(chart, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Portfolio analytics charts side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    <div class="grid-container">
                        <div class="grid-header">
                            <h2 class="grid-title">Portfolio Allocation</h2>
                            <p class="grid-description">Distribution by current value</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                allocation_chart = create_allocation_chart(portfolio_data)
                st.plotly_chart(allocation_chart, use_container_width=True)
            
            with col2:
                st.markdown("""
                    <div class="grid-container">
                        <div class="grid-header">
                            <h2 class="grid-title">P&L Analysis</h2>
                            <p class="grid-description">Profit & loss by position</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                pnl_chart = create_pnl_trend_chart(portfolio_data)
                st.plotly_chart(pnl_chart, use_container_width=True)
        else:
            # No selection, show default charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    <div class="grid-container">
                        <div class="grid-header">
                            <h2 class="grid-title">Portfolio Allocation</h2>
                            <p class="grid-description">Distribution by current value</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                allocation_chart = create_allocation_chart(portfolio_data)
                st.plotly_chart(allocation_chart, use_container_width=True)
            
            with col2:
                st.markdown("""
                    <div class="grid-container">
                        <div class="grid-header">
                            <h2 class="grid-title">P&L Analysis</h2>
                            <p class="grid-description">Profit & loss by position</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                pnl_chart = create_pnl_trend_chart(portfolio_data)
                st.plotly_chart(pnl_chart, use_container_width=True)
    else:
        # No selection data, show default charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="grid-container">
                    <div class="grid-header">
                        <h2 class="grid-title">Portfolio Allocation</h2>
                        <p class="grid-description">Distribution by current value</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            allocation_chart = create_allocation_chart(portfolio_data)
            st.plotly_chart(allocation_chart, use_container_width=True)
        
        with col2:
            st.markdown("""
                <div class="grid-container">
                    <div class="grid-header">
                        <h2 class="grid-title">P&L Analysis</h2>
                        <p class="grid-description">Profit & loss by position</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            pnl_chart = create_pnl_trend_chart(portfolio_data)
            st.plotly_chart(pnl_chart, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Market Watch Grid
    st.markdown("""
        <div class="grid-container">
            <div class="grid-header">
                <h2 class="grid-title">Market Watch</h2>
                <p class="grid-description">Real-time market data across sectors</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    gb_market = GridOptionsBuilder.from_dataframe(market_data)
    gb_market.configure_default_column(
        resizable=True,
        filterable=True,
        sortable=True,
        editable=False
    )
    gb_market.configure_column("Symbol", pinned='left', width=100, checkboxSelection=True, headerCheckboxSelection=True)
    gb_market.configure_column("Company", width=200)
    gb_market.configure_column("Sector", width=150, rowGroup=False)
    gb_market.configure_column("Price", width=120, type=['numericColumn'], valueFormatter="'$' + value.toFixed(2)", cellStyle={'textAlign': 'right'})
    gb_market.configure_column("Change %", width=120, type=['numericColumn'], valueFormatter="(value >= 0 ? '+' : '') + value.toFixed(2) + '%'", cellStyle={'textAlign': 'right', 'color': 'x.value >= 0 ? "#10b981" : "#ef4444"', 'fontWeight': 'bold'})
    gb_market.configure_column("Volume", width=130, type=['numericColumn'], valueFormatter="value >= 1000000 ? (value/1000000).toFixed(2) + 'M' : (value >= 1000 ? (value/1000).toFixed(2) + 'K' : value.toLocaleString())", cellStyle={'textAlign': 'right'})
    gb_market.configure_column("Market Cap", width=150, type=['numericColumn'], valueFormatter="value >= 1000000000 ? '$' + (value/1000000000).toFixed(2) + 'B' : '$' + (value/1000000).toFixed(2) + 'M'", cellStyle={'textAlign': 'right'})
    gb_market.configure_column("52W High", width=120, type=['numericColumn'], valueFormatter="'$' + value.toFixed(2)", cellStyle={'textAlign': 'right'})
    gb_market.configure_column("52W Low", width=120, type=['numericColumn'], valueFormatter="'$' + value.toFixed(2)", cellStyle={'textAlign': 'right'})
    gb_market.configure_column("P/E Ratio", width=100, cellStyle={'textAlign': 'right'})
    gb_market.configure_column("Dividend %", width=120, valueFormatter="value + '%'", cellStyle={'textAlign': 'right'})
    gb_market.configure_selection(selection_mode='multiple', use_checkbox=True)
    
    gridOptions_market = gb_market.build()
    
    market_response = AgGrid(
        market_data,
        gridOptions=gridOptions_market,
        height=500,
        theme='alpine',
        allow_unsafe_jscode=True,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        custom_css={
            ".ag-theme-alpine": {
                "border-radius": "0 0 16px 16px"
            }
        }
    )
    
    # Check if a stock is selected from Market Watch
    market_selected_rows = market_response.get('selected_rows', None)
    
    # Handle both DataFrame and list responses
    if market_selected_rows is not None:
        if hasattr(market_selected_rows, 'to_dict'):
            # It's a DataFrame, convert to list of dicts
            market_selected_rows = market_selected_rows.to_dict('records')
        
        if isinstance(market_selected_rows, list) and len(market_selected_rows) > 0:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("""
                <div style="margin-bottom: 20px;">
                    <h2 style="color: #ffffff; font-size: 22px; font-weight: 700; margin-bottom: 8px; font-family: 'Georgia', serif;">
                        Market Watch Charts
                    </h2>
                    <p style="color: rgba(255,255,255,0.6); font-size: 13px; margin: 0; font-family: 'Arial', sans-serif;">
                        Detailed analysis for selected stocks
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display charts for up to 2 selected stocks
            for i, selected_stock in enumerate(market_selected_rows[:2]):
                symbol = selected_stock['Symbol']
                company = selected_stock['Company']
                current_price = selected_stock['Price']
                change_percent = selected_stock['Change %']
                
                # Generate price history
                price_history = generate_price_history(symbol, current_price)
                
                # Display chart
                st.markdown(f"""
                    <div class="grid-container">
                        <div class="grid-header">
                            <h2 class="grid-title">Price Chart</h2>
                            <p class="grid-description">90-day price history with volume</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                chart = create_price_chart(symbol, company, price_history, current_price, change_percent)
                st.plotly_chart(chart, use_container_width=True)
                
                if i < len(market_selected_rows[:2]) - 1:
                    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Orders Grid
    st.markdown("""
        <div class="grid-container">
            <div class="grid-header">
                <h2 class="grid-title">Recent Orders</h2>
                <p class="grid-description">Your trading activity and order history</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    gb_orders = GridOptionsBuilder.from_dataframe(orders_data)
    gb_orders.configure_default_column(
        resizable=True,
        filterable=True,
        sortable=True,
        editable=False
    )
    gb_orders.configure_column("Order ID", width=120)
    gb_orders.configure_column("Symbol", width=100)
    gb_orders.configure_column("Type", width=80, cellStyle={'textAlign': 'center', 'color': 'x.value === "BUY" ? "#10b981" : "#ef4444"', 'fontWeight': 'bold'})
    gb_orders.configure_column("Shares", width=100, cellStyle={'textAlign': 'right'})
    gb_orders.configure_column("Price", width=120, type=['numericColumn'], valueFormatter="'$' + value.toFixed(2)", cellStyle={'textAlign': 'right'})
    gb_orders.configure_column("Total", width=130, type=['numericColumn'], valueFormatter="'$' + value.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})", cellStyle={'textAlign': 'right'})
    gb_orders.configure_column("Status", width=120, cellStyle={'textAlign': 'center', 'fontWeight': 'bold'})
    gb_orders.configure_column("Timestamp", width=180, sort='desc')
    gb_orders.configure_selection(selection_mode='single')
    
    gridOptions_orders = gb_orders.build()
    
    AgGrid(
        orders_data,
        gridOptions=gridOptions_orders,
        height=400,
        theme='alpine',
        allow_unsafe_jscode=True,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        custom_css={
            ".ag-theme-alpine": {
                "border-radius": "0 0 16px 16px"
            }
        }
    )

if __name__ == "__main__":
    main()


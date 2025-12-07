import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Options Greeks Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS - Light Professional Theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1f2937 !important;
        font-family: 'Georgia', serif;
    }
    
    .metric-label {
        color: #4b5563;
        font-size: 14px;
    }
    
    /* Streamlit metric styling */
    [data-testid="stMetricLabel"] {
        color: #4b5563 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #1f2937 !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: #ffffff !important;
    }
    
    /* Text elements */
    p, span, div {
        color: #374151;
    }
    
    /* Selectbox styling */
    .stSelectbox label {
        color: #1f2937 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        color: #1f2937 !important;
    }
</style>
""", unsafe_allow_html=True)

# Generate synthetic data
@st.cache_data
def generate_greeks_data():
    """Generate synthetic options Greeks data"""
    tenors = ['1M', '3M', '6M', '1Y']
    deltas = ['10P', '25P', '50P', '25C', '10C']
    
    # Create strategy indices (e.g., "1M_10P", "1M_25P", etc.)
    strategies = []
    for tenor in tenors:
        for delta in deltas:
            strategies.append(f"{tenor}_{delta}")
    
    # Generate random data for PnL, Vega, Gamma
    np.random.seed(42)  # For reproducible data
    
    data = {
        'PnL': np.random.uniform(-50000, 50000, len(strategies)),
        'Vega': np.random.uniform(1000, 10000, len(strategies)),
        'Gamma': np.random.uniform(-500, 500, len(strategies))
    }
    
    df = pd.DataFrame(data, index=strategies)
    return df

# Generate historical data for a strategy
@st.cache_data
def generate_historical_data(strategy, metric, days=30):
    """Generate synthetic historical data for a strategy"""
    np.random.seed(hash(strategy) % 2**32)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate random walk
    base_value = np.random.uniform(-10000, 10000) if metric == 'PnL' else np.random.uniform(1000, 5000)
    values = [base_value]
    
    for _ in range(days - 1):
        change = np.random.normal(0, base_value * 0.05)
        values.append(values[-1] + change)
    
    return pd.DataFrame({
        'Date': dates,
        metric: values
    })

# Create interactive heatmap
def create_heatmap(display_df, metric, selected_tenor=None, selected_delta=None):
    """Create an interactive heatmap with clickable cells"""
    
    # Prepare data for heatmap
    z_values = display_df.values
    x_labels = display_df.columns.tolist()
    y_labels = display_df.index.tolist()
    
    # Create text annotations for each cell
    text_values = []
    for i, row in enumerate(z_values):
        row_text = []
        for j, val in enumerate(row):
            if metric == 'PnL':
                row_text.append(f'${val:,.0f}')
            else:
                row_text.append(f'{val:,.0f}')
        text_values.append(row_text)
    
    # Determine color scale based on metric
    if metric == 'PnL' or metric == 'Gamma':
        colorscale = [[0, '#ef4444'], [0.5, '#f3f4f6'], [1, '#10b981']]
    else:
        colorscale = 'Blues'
    
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=x_labels,
        y=y_labels,
        text=text_values,
        texttemplate='%{text}',
        textfont={"size": 14, "color": "white"},
        colorscale=colorscale,
        hovertemplate='<b>Tenor:</b> %{y}<br><b>Delta:</b> %{x}<br><b>' + metric + ':</b> %{text}<extra></extra>',
        showscale=True,
        colorbar=dict(
            title=dict(text=metric, side='right'),
            tickmode="linear",
            tick0=0,
            dtick=10000 if metric == 'PnL' else 1000
        )
    ))
    
    # Highlight selected cell
    if selected_tenor and selected_delta:
        tenor_idx = y_labels.index(selected_tenor)
        delta_idx = x_labels.index(selected_delta)
        
        fig.add_shape(
            type="rect",
            x0=delta_idx - 0.5,
            y0=tenor_idx - 0.5,
            x1=delta_idx + 0.5,
            y1=tenor_idx + 0.5,
            line=dict(color="#3b82f6", width=3),
        )
    
    fig.update_layout(
        title=f'{metric} Heatmap',
        xaxis_title='Delta',
        yaxis_title='Tenor',
        height=450,
        font=dict(size=12),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')  # This ensures proper ordering
    )
    
    return fig

# Create time series chart
def create_timeseries_chart(hist_data, strategy, metric):
    """Create a time series chart for the selected strategy"""
    
    fig = go.Figure()
    
    # Add line trace
    color = '#10b981' if hist_data[metric].iloc[-1] >= 0 else '#ef4444'
    
    fig.add_trace(go.Scatter(
        x=hist_data['Date'],
        y=hist_data[metric],
        mode='lines+markers',
        name=metric,
        line=dict(color=color, width=2),
        marker=dict(size=6),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>' + metric + ':</b> %{y:,.0f}<extra></extra>'
    ))
    
    # Add zero line for PnL and Gamma
    if metric in ['PnL', 'Gamma']:
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f'{strategy} - 30 Day {metric} History',
        xaxis_title='Date',
        yaxis_title=metric,
        height=400,
        hovermode='x unified',
        plot_bgcolor='#f9fafb',
        paper_bgcolor='#ffffff',
        font=dict(size=12)
    )
    
    return fig

# Main app
def main():
    st.title("Options Greeks Dashboard")
    st.markdown('<p class="metric-label">Monitor risk metrics across tenors and deltas</p>', unsafe_allow_html=True)
    
    # Generate data
    greeks_df = generate_greeks_data()
    
    # Scaling controls
    st.markdown("### Scaling Options")
    
    scale_col1, scale_col2, scale_col3, scale_col4 = st.columns([1, 1, 1, 2])
    
    with scale_col1:
        enable_scaling = st.checkbox("Enable Scaling", value=False)
    
    with scale_col2:
        scale_greek = st.selectbox(
            "Scale By",
            options=['Vega', 'Gamma'],
            index=0,
            disabled=not enable_scaling
        )
    
    with scale_col3:
        reference_tenor = st.selectbox(
            "Reference Tenor",
            options=['1M', '3M', '6M', '1Y'],
            index=0,
            disabled=not enable_scaling
        )
    
    with scale_col4:
        reference_delta = st.selectbox(
            "Reference Delta",
            options=['10P', '25P', '50P', '25C', '10C'],
            index=2,  # Default to 50P
            disabled=not enable_scaling
        )
    
    # Apply scaling if enabled
    if enable_scaling:
        reference_strategy = f"{reference_tenor}_{reference_delta}"
        reference_value = greeks_df.loc[reference_strategy, scale_greek]
        
        # Scale all Greeks by the ratio of their scale_greek to the reference
        scaled_df = greeks_df.copy()
        
        for strategy in greeks_df.index:
            # Calculate scaling factor for this strategy based on the selected Greek
            strategy_greek_value = greeks_df.loc[strategy, scale_greek]
            ratio = strategy_greek_value / reference_value
            
            # Apply the ratio to all Greeks
            scaled_df.loc[strategy, 'PnL'] = greeks_df.loc[strategy, 'PnL'] * ratio
            scaled_df.loc[strategy, 'Vega'] = greeks_df.loc[strategy, 'Vega'] * ratio
            scaled_df.loc[strategy, 'Gamma'] = greeks_df.loc[strategy, 'Gamma'] * ratio
        
        # Show scaling info
        st.info(f"📊 Normalizing all Greeks relative to **{reference_strategy}** {scale_greek} = {reference_value:,.2f}. Reference contract values remain unchanged.")
        
        # Use scaled data for display
        display_greeks_df = scaled_df
        scaling_factor = None  # Not used in new approach
    else:
        display_greeks_df = greeks_df
        scaling_factor = 1
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Metric selector
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        metric = st.selectbox(
            "Select Metric",
            options=['PnL', 'Vega', 'Gamma'],
            index=0
        )
    
    with col2:
        total_value = display_greeks_df[metric].sum()
        st.metric(
            label=f"Total {metric}",
            value=f"${total_value:,.0f}" if metric == 'PnL' else f"{total_value:,.0f}"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create pivot table for display
    tenors = ['1M', '3M', '6M', '1Y']
    deltas = ['10P', '25P', '50P', '25C', '10C']
    
    # Build the grid data
    grid_data = []
    for tenor in tenors:
        row = {'Tenor': tenor}
        for delta in deltas:
            strategy = f"{tenor}_{delta}"
            value = display_greeks_df.loc[strategy, metric]
            row[delta] = value
        grid_data.append(row)
    
    # Create DataFrame for display
    display_df = pd.DataFrame(grid_data)
    display_df = display_df.set_index('Tenor')
    
    # Initialize session state for selected cell
    if 'selected_tenor' not in st.session_state:
        st.session_state.selected_tenor = '1M'
    if 'selected_delta' not in st.session_state:
        st.session_state.selected_delta = '50P'
    
    # Display interactive heatmap
    st.markdown(f"### {metric} by Tenor and Delta")
    
    heatmap = create_heatmap(display_df, metric, st.session_state.selected_tenor, st.session_state.selected_delta)
    st.plotly_chart(heatmap, use_container_width=True)
    
    # Manual cell selector
    st.markdown("### Select a Position")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_tenor = st.selectbox(
            "Tenor",
            options=tenors,
            index=tenors.index(st.session_state.selected_tenor),
            key='tenor_selector'
        )
    
    with col2:
        selected_delta = st.selectbox(
            "Delta",
            options=deltas,
            index=deltas.index(st.session_state.selected_delta),
            key='delta_selector'
        )
    
    # Update session state
    st.session_state.selected_tenor = selected_tenor
    st.session_state.selected_delta = selected_delta
    
    # Show selected position details
    if selected_tenor and selected_delta:
        strategy = f"{selected_tenor}_{selected_delta}"
        
        st.markdown(f"### Position: {strategy}")
        
        if enable_scaling:
            st.caption(f"Values scaled relative to {reference_strategy}")
        
        # Show all metrics for selected position
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pnl_val = display_greeks_df.loc[strategy, 'PnL']
            st.metric(
                label="P&L" + (" (normalized)" if enable_scaling else ""),
                value=f"${pnl_val:,.0f}",
                delta=f"{(pnl_val/abs(pnl_val)*100) if pnl_val != 0 else 0:.1f}%" if pnl_val != 0 else "0%"
            )
        
        with col2:
            vega_val = display_greeks_df.loc[strategy, 'Vega']
            st.metric(
                label="Vega" + (" (normalized)" if enable_scaling else ""),
                value=f"{vega_val:,.0f}"
            )
        
        with col3:
            gamma_val = display_greeks_df.loc[strategy, 'Gamma']
            st.metric(
                label="Gamma" + (" (normalized)" if enable_scaling else ""),
                value=f"{gamma_val:,.2f}"
            )
        
        # Generate and display historical chart
        st.markdown(f"#### Historical {metric}")
        hist_data = generate_historical_data(strategy, metric)
        chart = create_timeseries_chart(hist_data, strategy, metric)
        st.plotly_chart(chart, use_container_width=True)
    
    # Show summary statistics
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=f"Max {metric}",
            value=f"${display_df.values.max():,.0f}" if metric == 'PnL' else f"{display_df.values.max():,.0f}"
        )
    
    with col2:
        st.metric(
            label=f"Min {metric}",
            value=f"${display_df.values.min():,.0f}" if metric == 'PnL' else f"{display_df.values.min():,.0f}"
        )
    
    with col3:
        st.metric(
            label=f"Mean {metric}",
            value=f"${display_df.values.mean():,.0f}" if metric == 'PnL' else f"{display_df.values.mean():,.0f}"
        )
    
    with col4:
        st.metric(
            label=f"Std Dev",
            value=f"{display_df.values.std():,.0f}"
        )
    
    # Show raw data (optional)
    with st.expander("View Raw Data"):
        display_data = display_greeks_df if enable_scaling else greeks_df
        st.dataframe(
            display_data.style.format({
                'PnL': '${:,.2f}',
                'Vega': '{:,.2f}',
                'Gamma': '{:,.2f}'
            }),
            use_container_width=True
        )
        
        if enable_scaling:
            st.caption(f"All Greeks normalized by {scale_greek} ratio relative to {reference_strategy}")
            st.caption(f"Reference {scale_greek}: {reference_value:,.2f}")

if __name__ == "__main__":
    main()


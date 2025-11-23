import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from numpy import log, sqrt, exp  # Make sure to import these
import matplotlib.pyplot as plt
import seaborn as sns

#######################
# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")


# Custom CSS to inject into Streamlit
st.markdown("""
<style>
/* Adjust the size and alignment of the CALL and PUT value containers */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px; /* Adjust the padding to control height */
    width: auto; /* Auto width for responsiveness, or set a fixed width if necessary */
    margin: 0 auto; /* Center the container */
}

/* Custom classes for CALL and PUT values */
.metric-call {
    background-color: #90ee90; /* Light green background */
    color: black; /* Black font color */
    margin-right: 10px; /* Spacing between CALL and PUT */
    border-radius: 10px; /* Rounded corners */
}

.metric-put {
    background-color: #ffcccb; /* Light red background */
    color: black; /* Black font color */
    border-radius: 10px; /* Rounded corners */
}

/* Style for the value text */
.metric-value {
    font-size: 1.5rem; /* Adjust font size */
    font-weight: bold;
    margin: 0; /* Remove default margins */
}

/* Style for the label text */
.metric-label {
    font-size: 1rem; /* Adjust font size */
    margin-bottom: 4px; /* Spacing between label and value */
}

</style>
""", unsafe_allow_html=True)

# (Include the BlackScholes class definition here)

class BlackScholes:
    def __init__(
        self,
        time_to_maturity: float,
        strike: float,
        current_price: float,
        volatility: float,
        interest_rate: float,
    ):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def calculate_prices(
        self,
    ):
        time_to_maturity = self.time_to_maturity
        strike = self.strike
        current_price = self.current_price
        volatility = self.volatility
        interest_rate = self.interest_rate

        d1 = (
            log(current_price / strike) +
            (interest_rate + 0.5 * volatility ** 2) * time_to_maturity
            ) / (
                volatility * sqrt(time_to_maturity)
            )
        d2 = d1 - volatility * sqrt(time_to_maturity)

        call_price = current_price * norm.cdf(d1) - (
            strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(d2)
        )
        put_price = (
            strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(-d2)
        ) - current_price * norm.cdf(-d1)

        self.call_price = call_price
        self.put_price = put_price

        # GREEKS
        # Delta
        self.call_delta = norm.cdf(d1)
        self.put_delta = 1 - norm.cdf(d1)

        # Gamma
        self.call_gamma = norm.pdf(d1) / (
            strike * volatility * sqrt(time_to_maturity)
        )
        self.put_gamma = self.call_gamma

        return call_price, put_price

# Function to generate heatmaps
# ... your existing imports and BlackScholes class definition ...


# Sidebar for User Inputs
with st.sidebar:
    st.title("📊 Black-Scholes Model")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/elliot-webster123"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Elliot Webster`</a>', unsafe_allow_html=True)


    greeks_mode = st.checkbox("Enable Greeks Mode")

    if greeks_mode:
        st.subheader("Input Greek Variables")
        call_delta = st.number_input("Call Delta (Δ)", value=0.5)
        put_delta = st.number_input("Put Delta (Δ)", value=0.5)
        gamma = st.number_input("Gamma (Γ)", value=0.1)
        vega = st.number_input("Vega (ν)", value=0.2)
        theta = st.number_input("Theta (Θ)", value=-0.05)
        rho = st.number_input("Rho (ρ)", value=0.03)

        with st.expander("Delta Heatmap Parameters", expanded=True):
            spot_min_greek = st.number_input('Min Spot Price (Heatmap)', min_value=0.01, value=80.0, step=0.01)
            spot_max_greek = st.number_input('Max Spot Price (Heatmap)', min_value=0.01, value=120.0, step=0.01)
            vol_min_greek = st.slider('Min Volatility (Heatmap)', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
            vol_max_greek = st.slider('Max Volatility (Heatmap)', min_value=0.01, max_value=1.0, value=0.5, step=0.01)
            strike_greek = st.number_input('Strike Price (Heatmap)', min_value=0.01, value=100.0, step=0.01)
            ttm_greek = st.number_input('Time to Maturity (Years, Heatmap)', min_value=0.01, value=1.0, step=0.01)
            spot_range_demo = np.linspace(spot_min_greek, spot_max_greek, 10)
            vol_range_demo = np.linspace(vol_min_greek, vol_max_greek, 10)

        with st.expander("Volatility Surface Preset", expanded=True):
            preset = {
                "x_label": "Strike Price (K)",
                "y_label": "Time to Expiration (T)",
                "z_label": "Vega (Sensitivity)"
            }
            x_min = st.number_input(f'Min {preset["x_label"]}', min_value=0.01, value=80.0, step=0.01)
            x_max = st.number_input(f'Max {preset["x_label"]}', min_value=0.01, value=120.0, step=0.01)
            y_min = st.number_input(f'Min {preset["y_label"]}', min_value=0.01, value=0.1, step=0.01)
            y_max = st.number_input(f'Max {preset["y_label"]}', min_value=0.01, value=2.0, step=0.01)
            x_range = np.linspace(x_min, x_max, 10)
            y_range = np.linspace(y_min, y_max, 10)
    else:
        current_price = st.number_input("Current Asset Price", value=100.0)
        strike = st.number_input("Strike Price", value=100.0)
        time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0)
        volatility = st.number_input("Volatility (σ)", value=0.2)
        interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05)

        st.markdown("---")
        calculate_btn = st.button('Heatmap Parameters')
        spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
        spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
        vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*0.5, step=0.01)
        vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*1.5, step=0.01)
        spot_range = np.linspace(spot_min, spot_max, 10)
        vol_range = np.linspace(vol_min, vol_max, 10)



def plot_heatmap(bs_model, spot_range, vol_range, strike):
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            bs_temp.calculate_prices()
            call_prices[i, j] = bs_temp.call_price
            put_prices[i, j] = bs_temp.put_price
    
    # Plotting Call Price Heatmap
    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(call_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="viridis", ax=ax_call)
    ax_call.set_title('CALL')
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility')
    
    # Plotting Put Price Heatmap
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(put_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="viridis", ax=ax_put)
    ax_put.set_title('PUT')
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility')
    
    return fig_call, fig_put


# Main Page for Output Display
st.title("Black-Scholes Pricing Model")


# Table of Inputs and Output Display
if 'greeks_mode' in locals() and greeks_mode:
    st.subheader("Greeks Mode Enabled")
    greeks_data = {
        "Call Delta": [call_delta],
        "Put Delta": [put_delta],
        "Gamma": [gamma],
        "Vega": [vega],
        "Theta": [theta],
        "Rho": [rho],
    }
    greeks_df = pd.DataFrame(greeks_data)
    st.table(greeks_df)

    col1, col2 = st.columns([1,1], gap="small")
    with col1:
        st.markdown(f"""
            <div class="metric-container metric-call">
                <div>
                    <div class="metric-label">Call Delta</div>
                    <div class="metric-value">{call_delta:.2f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-container metric-put">
                <div>
                    <div class="metric-label">Put Delta</div>
                    <div class="metric-value">{put_delta:.2f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("")
    st.info("Greeks mode is enabled. Input your own Greek values above.")

    # Use sidebar parameters for Greeks mode
    # spot_range_demo, vol_range_demo, strike_greek, ttm_greek, vega are all set in sidebar


    # Delta Heatmaps (Call and Put)
    st.markdown("### Delta Heatmaps")
    st.info("Visualize how Call and Put Delta change with spot price and volatility.")
    call_delta_matrix = np.zeros((len(vol_range_demo), len(spot_range_demo)))
    put_delta_matrix = np.zeros((len(vol_range_demo), len(spot_range_demo)))
    for i, vol in enumerate(vol_range_demo):
        for j, spot in enumerate(spot_range_demo):
            bs_temp = BlackScholes(
                time_to_maturity=ttm_greek,
                strike=strike_greek,
                current_price=spot,
                volatility=vol,
                interest_rate=0.05
            )
            bs_temp.calculate_prices()
            call_delta_matrix[i, j] = bs_temp.call_delta
            put_delta_matrix[i, j] = bs_temp.put_delta
    col1, col2 = st.columns(2)
    with col1:
        fig_call_delta, ax_call_delta = plt.subplots(figsize=(7, 5))
        sns.heatmap(call_delta_matrix, xticklabels=np.round(spot_range_demo, 2), yticklabels=np.round(vol_range_demo, 2), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_call_delta, cbar=True)
        ax_call_delta.set_title('Call Delta Heatmap')
        ax_call_delta.set_xlabel('Spot Price')
        ax_call_delta.set_ylabel('Volatility')
        st.pyplot(fig_call_delta)
    with col2:
        fig_put_delta, ax_put_delta = plt.subplots(figsize=(7, 5))
        sns.heatmap(put_delta_matrix, xticklabels=np.round(spot_range_demo, 2), yticklabels=np.round(vol_range_demo, 2), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_put_delta, cbar=True)
        ax_put_delta.set_title('Put Delta Heatmap')
        ax_put_delta.set_xlabel('Spot Price')
        ax_put_delta.set_ylabel('Volatility')
        st.pyplot(fig_put_delta)

    st.info("Interact with the 3D surface for Vega sensitivity.")
    st.markdown(f"**X Axis:** Strike Price (K) | **Y Axis:** {preset['y_label']} | **Z Axis:** {preset['z_label']}")
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_surface = np.zeros_like(x_grid)
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            # X: Strike Price (K), Y: Expiry (T), Z: Vega
            S = 100.0
            K = x_grid[i, j]
            T = y_grid[i, j]
            sigma = 0.2
            bs_temp = BlackScholes(
                time_to_maturity=T,
                strike=K,
                current_price=S,
                volatility=sigma,
                interest_rate=0.05
            )
            bs_temp.calculate_prices()
            d1 = (log(S / K) + (0.05 + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
            vega_val = S * norm.pdf(d1) * sqrt(T)
            z_surface[i, j] = vega_val
    surface_fig = go.Figure(data=[go.Surface(z=z_surface, x=x_range, y=y_range, colorscale='Viridis', colorbar=dict(title=preset['z_label']))])
    surface_fig.update_layout(
        title='Volatility Surface: Vega Sensitivity',
        scene = dict(
            xaxis_title="Strike Price (K)",
            yaxis_title=preset['y_label'],
            zaxis_title=preset['z_label'],
        ),
        autosize=True,
        margin=dict(l=65, r=50, b=65, t=90)
    )
    st.plotly_chart(surface_fig, use_container_width=True)
else:
    input_data = {
        "Current Asset Price": [current_price],
        "Strike Price": [strike],
        "Time to Maturity (Years)": [time_to_maturity],
        "Volatility (σ)": [volatility],
        "Risk-Free Interest Rate": [interest_rate],
    }
    input_df = pd.DataFrame(input_data)
    st.table(input_df)

    # Calculate Call and Put values
    bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
    call_price, put_price = bs_model.calculate_prices()

    # Display Call and Put Values in colored tables
    col1, col2 = st.columns([1,1], gap="small")
    with col1:
        st.markdown(f"""
            <div class="metric-container metric-call">
                <div>
                    <div class="metric-label">CALL Value</div>
                    <div class="metric-value">${call_price:.2f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-container metric-put">
                <div>
                    <div class="metric-label">PUT Value</div>
                    <div class="metric-value">${put_price:.2f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("")
    st.title("Options Price - Interactive Heatmap")
    st.info("Explore how option prices fluctuate with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters, all while maintaining a constant 'Strike Price'.")

    # Interactive Sliders and Heatmaps for Call and Put Options
    col1, col2 = st.columns([1,1], gap="small")
    with col1:
        st.subheader("Call Price Heatmap")
        heatmap_fig_call, _ = plot_heatmap(bs_model, spot_range, vol_range, strike)
        st.pyplot(heatmap_fig_call)
    with col2:
        st.subheader("Put Price Heatmap")
        _, heatmap_fig_put = plot_heatmap(bs_model, spot_range, vol_range, strike)
        st.pyplot(heatmap_fig_put)

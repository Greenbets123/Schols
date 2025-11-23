# Black-Scholes Pricing Model

This repository provides an interactive Black-Scholes Pricing Model dashboard that helps in visualizing option prices, volatility surfaces, and Greeks under varying market conditions. The dashboard is user-friendly and interactive, allowing users to explore how changes in spot price, volatility, and other parameters influence option values.

**Website** https://blackscholesplusgreek.streamlit.app

## 🚀 Features:

1. **Options Pricing Visualization**  
   - Displays both Call and Put option prices using an interactive heatmap.  
   - The heatmap dynamically updates as you adjust parameters like Spot Price, Volatility, and Time to Maturity.  

2. **Volatility Surface**  
   - Generates a 3D volatility surface for options across different strikes and maturities.  
   - Visualizes how implied volatility varies with strike price and time to maturity.  
   - Helps identify patterns like volatility smile and skew.  

3. **Greeks Mode**  
   - Calculates and displays option Greeks: Delta, Gamma, Theta, Vega, and Rho.  
   - Users can explore the sensitivity of option prices to changes in underlying parameters.  
   - Supports both Call and Put options for comprehensive risk analysis.  

4. **Interactive Dashboard**  
   - Allows real-time updates to the Black-Scholes model parameters.  
   - Users can input Spot Price, Strike Price, Volatility, Time to Maturity, and Risk-Free Interest Rate.  
   - Call and Put prices and Greeks are calculated and displayed for immediate comparison.  

5. **Customizable Parameters**  
   - Set custom ranges for Spot Price and Volatility to generate a comprehensive view of option prices under different market conditions.  

## 🔧 Dependencies:

- `yfinance`: To fetch current asset prices  
- `numpy`: For numerical operations  
- `matplotlib`: For heatmap visualization  
- `plotly`: For interactive 3D volatility surface  
- `streamlit`: For the interactive dashboard  
- `scipy`: For Greeks calculation  

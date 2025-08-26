# XGBoost Solana

This project predicts Solana (SOL-USD) returns using an **XGBoost regression model** and simulates future portfolio growth with Monte Carlo simulations.

## Features
- Lagged returns (`Return_Open_1`, `Return_Close_1`, etc.)
- Rolling means (`MA_Close_5_lag`, `MA_Close_10`)
- Rolling volatility (`Vol_Close_5_lag`, `Vol_High_5`)
- Rolling high/low (`Rolling_High_5`, `Rolling_Low_5`)
- Normalized features (`HL_Ratio_5_lag`, `Close_vs_MA5_lag`)
- Technical indicators (`RSI_14_lag`, `MACD_lag`)
- Future return simulation using trained XGBoost model
- Monte Carlo simulation to account for randomness and market volatility

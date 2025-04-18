# StockSavvy Architecture

## System Architecture Diagram

```
                                    EXTERNAL DATA SOURCES
                                 ┌─────────────────────────┐
                                 │                         │
                                 │  Alpha Vantage API      │
                                 │  Yahoo Finance API      │
                                 │  News APIs              │
                                 │  CoinMarketCap API      │
                                 │                         │
                                 └───────────┬─────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│                                DATA INTEGRATION LAYER                           │
│                                                                                 │
│   ┌────────────────────┐     ┌────────────────────┐    ┌────────────────────┐  │
│   │                    │     │                    │    │                    │  │
│   │  Stock Data        │     │  Cryptocurrency    │    │  News Service      │  │
│   │  Module            │     │  Module            │    │  Module            │  │
│   │  (get_stock_data)  │     │  (format_crypto)   │    │  (get_news)        │  │
│   │                    │     │                    │    │                    │  │
│   └─────────┬──────────┘     └──────────┬─────────┘    └──────────┬─────────┘  │
│             │                           │                          │            │
└─────────────┼───────────────────────────┼──────────────────────────┼────────────┘
              │                           │                          │
              ▼                           ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│                              ANALYSIS & PROCESSING LAYER                        │
│                                                                                 │
│   ┌────────────────────┐     ┌────────────────────┐    ┌────────────────────┐  │
│   │                    │     │                    │    │                    │  │
│   │  Technical         │     │  Prediction        │    │  Fundamental       │  │
│   │  Analysis Engine   │     │  Model System      │    │  Analysis Engine   │  │
│   │                    │     │                    │    │                    │  │
│   └─────────┬──────────┘     └──────────┬─────────┘    └──────────┬─────────┘  │
│             │                           │                          │            │
│   ┌─────────▼──────────┐                │               ┌──────────▼─────────┐ │
│   │                    │     ┌──────────▼─────────┐     │                    │ │
│   │  Signal            │     │                    │     │  Backtesting       │ │
│   │  Generation        │     │  ML Models         │     │  Engine            │ │
│   │                    │     │  - LSTM Networks   │     │                    │ │
│   └────────────────────┘     │  - Time Series     │     └────────────────────┘ │
│                              │  - Ensemble        │                            │
│                              │                    │                            │
│                              └────────────────────┘                            │
│                                                                                 │
└─────────────┬───────────────────────────┬──────────────────────────┬────────────┘
              │                           │                          │
              ▼                           ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│                              PRESENTATION LAYER                                 │
│                                                                                 │
│   ┌────────────────────┐     ┌────────────────────┐    ┌────────────────────┐  │
│   │                    │     │                    │    │                    │  │
│   │  Streamlit Web     │     │  Technical         │    │  Visualization     │  │
│   │  Interface         │     │  Charts            │    │  Components        │  │
│   │  (main.py)         │     │  (plot.ly)         │    │  (components/)     │  │
│   │                    │     │                    │    │                    │  │
│   └────────────────────┘     └────────────────────┘    └────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
                                 ┌─────────────────────────┐
                                 │                         │
                                 │     USER INTERFACES     │
                                 │                         │
                                 │  - Web Browser          │
                                 │  - React Native App     │
                                 │    (Mobile)             │
                                 │                         │
                                 └─────────────────────────┘
```

## Component Descriptions

### Data Integration Layer

- **Stock Data Module**: Located in `utils/stock_data.py`
  - Fetches stock data from Alpha Vantage API with fallback to Yahoo Finance
  - Handles API errors and retries
  - Formats and standardizes data structure

- **Cryptocurrency Module**: Located in `utils/stock_data.py`
  - Special handling for cryptocurrency symbols
  - Integration with CoinMarketCap API
  - Format conversion

- **News Service Module**: Located in `utils/news_service.py`
  - Fetches news articles related to stocks/crypto
  - Performs sentiment analysis
  - Filters and sorts news by relevance

### Analysis & Processing Layer

- **Technical Analysis Engine**: Located in `utils/technical_analysis.py`
  - Calculates technical indicators (RSI, MACD, Bollinger Bands, etc.)
  - Computes moving averages
  - Identifies patterns

- **Signal Generation**: Located in `utils/technical_analysis.py`
  - Identifies buy/sell signals based on indicators
  - Calculates signal strength
  - Provides trading recommendations

- **Prediction Model System**: Located in `utils/prediction.py`
  - Orchestrates multiple prediction models
  - Combines models into ensemble predictions
  - Handles timeframes from 1-day to 6-month forecasts

- **ML Models**: Located in `utils/prediction.py`
  - LSTM neural networks for time series forecasting
  - Statistical models for trend analysis
  - Seasonal decomposition

- **Fundamental Analysis Engine**: Located in `utils/fundamental_analysis.py`
  - Analyzes company financial metrics
  - Calculates financial ratios
  - Provides value assessments

- **Backtesting Engine**: Located in `utils/backtest.py`
  - Simulates trading strategies on historical data
  - Calculates performance metrics
  - Optimizes strategy parameters

### Presentation Layer

- **Streamlit Web Interface**: Located in `main.py`
  - Renders the user interface
  - Handles user interactions
  - Coordinates between components

- **Technical Charts**: Using Plotly
  - Interactive candlestick charts
  - Indicator overlays
  - Volume analysis

- **Visualization Components**: Located in `components/`
  - Prediction cards
  - News displays
  - Signal indicators
  - Technical analysis summaries

### User Interfaces

- **Web Browser**: Primary interface through Streamlit
- **React Native Mobile App**: Located in `App.tsx`, `src/`
  - Cross-platform mobile interface
  - Connects to the same data sources
  - Native mobile UI components

## Data Flow

1. User searches for a stock/crypto symbol
2. Request goes to Data Integration Layer
3. Data fetched from external APIs
4. Raw data processed through Analysis Layer
   - Technical indicators calculated
   - Predictions generated via ML models
   - Signals identified
5. Processed results sent to Presentation Layer
6. UI components render visualizations and insights
7. User views and interacts with the results

## Prediction Model Architecture

The prediction system uses a multi-model ensemble approach:

```
                     ┌───────────────────┐
                     │                   │
                     │  Input Features   │
                     │                   │
                     └─────────┬─────────┘
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
     ┌───────────▼───┐ ┌───────▼───┐ ┌───────▼───┐
     │               │ │           │ │           │
     │  LSTM Model   │ │ Stat Model│ │  Pattern  │
     │               │ │           │ │Recognition│
     │               │ │           │ │           │
     └───────┬───────┘ └─────┬─────┘ └─────┬─────┘
             │               │             │
             └───────┬───────┴─────┬───────┘
                     │             │
           ┌─────────▼─────────────▼───────┐
           │                               │
           │     Ensemble Aggregator       │
           │                               │
           └───────────────┬───────────────┘
                           │
                 ┌─────────▼─────────┐
                 │                   │
                 │  Confidence       │
                 │  Calculation      │
                 │                   │
                 └─────────┬─────────┘
                           │
                 ┌─────────▼─────────┐
                 │                   │
                 │ Price Prediction  │
                 │ with Range        │
                 │                   │
                 └───────────────────┘
```

## Key Files and Their Functions

- `main.py`: Entry point and UI orchestration
- `utils/stock_data.py`: Data fetching and preprocessing
- `utils/technical_analysis.py`: Technical indicators and signals
- `utils/prediction.py`: Price prediction models
- `utils/fundamental_analysis.py`: Company metrics analysis
- `utils/news_service.py`: News fetching and sentiment analysis
- `utils/backtest.py`: Strategy backtesting engine
- `components/*.py`: UI components for visualization
- `App.tsx`: Mobile app entry point
- `src/screens/*.tsx`: Mobile app screens
- `src/services/*.ts`: Mobile API services
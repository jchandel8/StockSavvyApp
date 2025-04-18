# StockSavvy - Advanced Stock Analysis Platform

StockSavvy is a comprehensive financial analysis platform that empowers traders with actionable insights through advanced data processing and machine learning techniques.

## 🚀 Features

- **Real-time Market Data**: Fetch and analyze the latest stock and cryptocurrency data with enhanced error handling and API fallback mechanisms
- **Technical Analysis**: View and interpret key technical indicators such as RSI, MACD, Moving Averages, and more
- **Price Predictions**: Access machine learning-powered predictions across multiple timeframes (1 day to 6 months)
- **News Aggregation**: Stay informed with the latest news and sentiment analysis for your selected assets
- **Backtesting**: Test trading strategies against historical data to evaluate performance

## 💻 Technology Stack

### Frontend
- Streamlit for the interactive web interface
- Plotly for advanced data visualization
- Custom CSS for professional dark-themed design

### Backend
- Python for core data processing and analysis
- Advanced technical analysis algorithms
- Machine learning models for price prediction
- Multi-source API integration

### Data Sources
- Alpha Vantage API for primary stock data
- Yahoo Finance as a secondary data source
- News APIs for sentiment analysis

## 📊 Key Components

1. **Technical Analysis Engine**
   - Comprehensive indicator calculations
   - Signal generation for trading opportunities
   - Pattern recognition

2. **Prediction System**
   - Time-series forecasting models
   - Ensemble machine learning approach
   - Confidence metrics and price range estimates

3. **News & Sentiment Analysis**
   - Real-time news aggregation
   - NLP-based sentiment scoring
   - Impact analysis on price movements

4. **Backtesting Framework**
   - Historical performance simulation
   - Risk assessment
   - Strategy optimization

## 🔧 Setup & Installation

### Prerequisites
- Python 3.10+
- Node.js (for React Native mobile app)

### Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/stocksavvy.git
   cd stocksavvy
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up API keys:
   - Create an account on Alpha Vantage to get an API key
   - Add your API keys to the environment variables or a `.env` file

### Running the App
1. Start the Streamlit web interface:
   ```
   streamlit run main.py
   ```

2. Access the app in your browser:
   ```
   http://localhost:5000
   ```

## 📱 Mobile App (React Native)

The repository also includes a React Native mobile application for accessing StockSavvy on iOS and Android devices.

To run the mobile app:
1. Navigate to the project directory
2. Install dependencies: `npm install`
3. Start the development server: `npx react-native start`
4. Run on iOS: `npx react-native run-ios`
5. Run on Android: `npx react-native run-android`

## 🔍 Usage

1. **Search**: Use the search bar to find stocks or cryptocurrencies
2. **Analysis**: View comprehensive technical analysis with key indicators
3. **Predictions**: Check price predictions across multiple time frames
4. **News**: Stay updated with the latest relevant news
5. **Backtesting**: Test and optimize your trading strategies

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Alpha Vantage for providing financial market data
- Streamlit for the awesome web framework
- The open-source community for various libraries and tools used in this project
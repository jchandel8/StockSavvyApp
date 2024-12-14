import axios from 'axios';

const API_BASE_URL = 'YOUR_API_BASE_URL';

export const searchStocks = async (query: string) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/search?q=${query}`);
    return response.data;
  } catch (error) {
    console.error('Error searching stocks:', error);
    return [];
  }
};

export const getStockData = async (symbol: string) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/stock/${symbol}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching stock data:', error);
    throw error;
  }
};

export const getStockNews = async (symbol: string) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/news/${symbol}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching news:', error);
    return [];
  }
};

export const runBacktest = async (symbol: string, parameters: any) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/backtest/${symbol}`, parameters);
    return response.data;
  } catch (error) {
    console.error('Error running backtest:', error);
    throw error;
  }
};

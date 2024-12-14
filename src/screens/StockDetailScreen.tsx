import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import { LineChart } from 'react-native-chart-kit';
import { getStockData } from '../services/stockService';

const StockDetailScreen = ({ route }) => {
  const { symbol } = route.params;
  const [stockData, setStockData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStockData();
  }, [symbol]);

  const fetchStockData = async () => {
    try {
      const data = await getStockData(symbol);
      setStockData(data);
    } catch (error) {
      console.error('Error fetching stock data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#FF4B4B" />
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      {stockData && (
        <>
          <View style={styles.header}>
            <Text style={styles.symbol}>{symbol}</Text>
            <Text style={styles.price}>${stockData.currentPrice}</Text>
            <Text style={[
              styles.change,
              { color: stockData.priceChange >= 0 ? '#00C853' : '#FF3D00' }
            ]}>
              {stockData.priceChange >= 0 ? '+' : ''}{stockData.priceChange.toFixed(2)}%
            </Text>
          </View>

          <View style={styles.chartContainer}>
            <LineChart
              data={{
                labels: stockData.labels,
                datasets: [{
                  data: stockData.prices
                }]
              }}
              width={350}
              height={220}
              chartConfig={{
                backgroundColor: '#262730',
                backgroundGradientFrom: '#262730',
                backgroundGradientTo: '#262730',
                decimalPlaces: 2,
                color: (opacity = 1) => `rgba(255, 75, 75, ${opacity})`,
                labelColor: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
                style: {
                  borderRadius: 16
                }
              }}
              bezier
              style={styles.chart}
            />
          </View>

          <View style={styles.indicatorsContainer}>
            <Text style={styles.sectionTitle}>Technical Indicators</Text>
            {stockData.technicalIndicators.map((indicator, index) => (
              <View key={index} style={styles.indicator}>
                <Text style={styles.indicatorName}>{indicator.name}</Text>
                <Text style={styles.indicatorValue}>{indicator.value}</Text>
              </View>
            ))}
          </View>

          <View style={styles.predictionsContainer}>
            <Text style={styles.sectionTitle}>Price Predictions</Text>
            {stockData.predictions.map((prediction, index) => (
              <View key={index} style={styles.prediction}>
                <Text style={styles.predictionTimeframe}>{prediction.timeframe}</Text>
                <Text style={[
                  styles.predictionDirection,
                  { color: prediction.direction === 'UP' ? '#00C853' : '#FF3D00' }
                ]}>
                  {prediction.direction}
                </Text>
                <Text style={styles.predictionConfidence}>
                  Confidence: {(prediction.confidence * 100).toFixed(1)}%
                </Text>
              </View>
            ))}
          </View>
        </>
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0E1117',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0E1117',
  },
  header: {
    padding: 16,
    backgroundColor: '#262730',
    marginBottom: 16,
  },
  symbol: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFFFFF',
  },
  price: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginTop: 8,
  },
  change: {
    fontSize: 18,
    fontWeight: '500',
    marginTop: 4,
  },
  chartContainer: {
    padding: 16,
    backgroundColor: '#262730',
    borderRadius: 8,
    marginBottom: 16,
  },
  chart: {
    marginVertical: 8,
    borderRadius: 16,
  },
  indicatorsContainer: {
    padding: 16,
    backgroundColor: '#262730',
    borderRadius: 8,
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 12,
  },
  indicator: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  indicatorName: {
    color: '#FFFFFF',
    fontSize: 16,
  },
  indicatorValue: {
    color: '#FF4B4B',
    fontSize: 16,
    fontWeight: '500',
  },
  predictionsContainer: {
    padding: 16,
    backgroundColor: '#262730',
    borderRadius: 8,
    marginBottom: 16,
  },
  prediction: {
    marginBottom: 12,
  },
  predictionTimeframe: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '500',
  },
  predictionDirection: {
    fontSize: 24,
    fontWeight: 'bold',
    marginTop: 4,
  },
  predictionConfidence: {
    color: '#FFFFFF',
    fontSize: 14,
    marginTop: 4,
  },
});

export default StockDetailScreen;

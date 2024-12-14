import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  ActivityIndicator,
} from 'react-native';
import { LineChart } from 'react-native-chart-kit';
import { runBacktest } from '../services/stockService';

const BacktestScreen = ({ route }) => {
  const { symbol } = route.params;
  const [investment, setInvestment] = useState('10000');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  const handleBacktest = async () => {
    setLoading(true);
    try {
      const data = await runBacktest(symbol, {
        initialInvestment: parseFloat(investment)
      });
      setResults(data);
    } catch (error) {
      console.error('Error running backtest:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.inputContainer}>
        <Text style={styles.label}>Initial Investment ($)</Text>
        <TextInput
          style={styles.input}
          value={investment}
          onChangeText={setInvestment}
          keyboardType="numeric"
          placeholder="Enter initial investment"
          placeholderTextColor="#666666"
        />
        <TouchableOpacity
          style={styles.button}
          onPress={handleBacktest}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="#FFFFFF" />
          ) : (
            <Text style={styles.buttonText}>Run Backtest</Text>
          )}
        </TouchableOpacity>
      </View>

      {results && (
        <>
          <View style={styles.resultsContainer}>
            <View style={styles.metric}>
              <Text style={styles.metricLabel}>Final Value</Text>
              <Text style={styles.metricValue}>
                ${results.finalValue.toLocaleString()}
              </Text>
            </View>
            <View style={styles.metric}>
              <Text style={styles.metricLabel}>Total Return</Text>
              <Text style={[
                styles.metricValue,
                { color: results.totalReturn >= 0 ? '#00C853' : '#FF3D00' }
              ]}>
                {results.totalReturn.toFixed(1)}%
              </Text>
            </View>
            <View style={styles.metric}>
              <Text style={styles.metricLabel}>Win Rate</Text>
              <Text style={styles.metricValue}>{results.winRate.toFixed(1)}%</Text>
            </View>
            <View style={styles.metric}>
              <Text style={styles.metricLabel}>Total Trades</Text>
              <Text style={styles.metricValue}>{results.totalTrades}</Text>
            </View>
          </View>

          <View style={styles.chartContainer}>
            <Text style={styles.chartTitle}>Portfolio Value History</Text>
            <LineChart
              data={{
                labels: results.labels,
                datasets: [{
                  data: results.portfolioValues
                }]
              }}
              width={350}
              height={220}
              chartConfig={{
                backgroundColor: '#262730',
                backgroundGradientFrom: '#262730',
                backgroundGradientTo: '#262730',
                decimalPlaces: 0,
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

          <View style={styles.tradesContainer}>
            <Text style={styles.sectionTitle}>Trading History</Text>
            {results.trades.map((trade, index) => (
              <View key={index} style={styles.trade}>
                <Text style={styles.tradeDate}>{trade.date}</Text>
                <View style={styles.tradeDetails}>
                  <Text style={styles.tradeType}>
                    {trade.type} @ ${trade.price}
                  </Text>
                  <Text style={[
                    styles.tradeProfit,
                    { color: trade.profit >= 0 ? '#00C853' : '#FF3D00' }
                  ]}>
                    ${trade.profit.toFixed(2)}
                  </Text>
                </View>
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
  inputContainer: {
    padding: 16,
    backgroundColor: '#262730',
    marginBottom: 16,
  },
  label: {
    color: '#FFFFFF',
    fontSize: 16,
    marginBottom: 8,
  },
  input: {
    backgroundColor: '#1A1D24',
    color: '#FFFFFF',
    padding: 12,
    borderRadius: 8,
    fontSize: 16,
    marginBottom: 16,
  },
  button: {
    backgroundColor: '#FF4B4B',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  resultsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    padding: 16,
    backgroundColor: '#262730',
    marginBottom: 16,
  },
  metric: {
    width: '50%',
    marginBottom: 16,
  },
  metricLabel: {
    color: '#FFFFFF',
    fontSize: 14,
  },
  metricValue: {
    color: '#FFFFFF',
    fontSize: 24,
    fontWeight: 'bold',
    marginTop: 4,
  },
  chartContainer: {
    padding: 16,
    backgroundColor: '#262730',
    marginBottom: 16,
  },
  chartTitle: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  chart: {
    marginVertical: 8,
    borderRadius: 16,
  },
  tradesContainer: {
    padding: 16,
    backgroundColor: '#262730',
    marginBottom: 16,
  },
  sectionTitle: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  trade: {
    marginBottom: 12,
  },
  tradeDate: {
    color: '#FFFFFF',
    fontSize: 14,
  },
  tradeDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 4,
  },
  tradeType: {
    color: '#FFFFFF',
    fontSize: 16,
  },
  tradeProfit: {
    fontSize: 16,
    fontWeight: '500',
  },
});

export default BacktestScreen;

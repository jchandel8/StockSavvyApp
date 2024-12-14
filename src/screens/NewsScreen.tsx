import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import { getStockNews } from '../services/stockService';

const NewsScreen = ({ route }) => {
  const { symbol } = route.params;
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchNews();
  }, [symbol]);

  const fetchNews = async () => {
    try {
      const newsData = await getStockNews(symbol);
      setNews(newsData);
    } catch (error) {
      console.error('Error fetching news:', error);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment.toLowerCase()) {
      case 'positive':
        return '#00C853';
      case 'negative':
        return '#FF3D00';
      default:
        return '#FFC107';
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
      {news.map((article, index) => (
        <View
          key={index}
          style={[
            styles.articleCard,
            { borderLeftColor: getSentimentColor(article.sentiment) }
          ]}
        >
          <Text style={styles.title}>{article.title}</Text>
          <Text style={styles.summary}>{article.summary}</Text>
          <View style={styles.footer}>
            <Text style={styles.source}>{article.source}</Text>
            <Text style={[
              styles.sentiment,
              { color: getSentimentColor(article.sentiment) }
            ]}>
              {article.sentiment}
            </Text>
          </View>
        </View>
      ))}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0E1117',
    padding: 16,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0E1117',
  },
  articleCard: {
    backgroundColor: '#262730',
    borderRadius: 8,
    padding: 16,
    marginBottom: 16,
    borderLeftWidth: 4,
  },
  title: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  summary: {
    color: '#FFFFFF',
    fontSize: 14,
    marginBottom: 12,
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  source: {
    color: '#666666',
    fontSize: 12,
  },
  sentiment: {
    fontSize: 12,
    fontWeight: '500',
  },
});

export default NewsScreen;

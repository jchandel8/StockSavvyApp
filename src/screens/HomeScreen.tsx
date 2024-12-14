import React, { useState } from 'react';
import {
  View,
  TextInput,
  FlatList,
  TouchableOpacity,
  Text,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';
import { searchStocks } from '../services/stockService';

const HomeScreen = ({ navigation }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async (query: string) => {
    if (query.length < 2) return;
    
    setLoading(true);
    try {
      const results = await searchStocks(query);
      setSearchResults(results);
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.searchInput}
        placeholder="Search stocks (e.g., AAPL)"
        value={searchQuery}
        onChangeText={(text) => {
          setSearchQuery(text);
          handleSearch(text);
        }}
        autoCapitalize="characters"
      />

      {loading ? (
        <ActivityIndicator size="large" color="#FF4B4B" />
      ) : (
        <FlatList
          data={searchResults}
          keyExtractor={(item) => item.symbol}
          renderItem={({ item }) => (
            <TouchableOpacity
              style={styles.resultItem}
              onPress={() => navigation.navigate('StockDetail', { symbol: item.symbol })}
            >
              <Text style={styles.symbolText}>{item.symbol}</Text>
              <Text style={styles.nameText}>{item.name}</Text>
            </TouchableOpacity>
          )}
        />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    backgroundColor: '#0E1117',
  },
  searchInput: {
    height: 48,
    borderWidth: 1,
    borderColor: '#262730',
    borderRadius: 8,
    paddingHorizontal: 16,
    marginBottom: 16,
    color: '#FAFAFA',
    backgroundColor: '#262730',
  },
  resultItem: {
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#262730',
  },
  symbolText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FF4B4B',
  },
  nameText: {
    fontSize: 14,
    color: '#FAFAFA',
    marginTop: 4,
  },
});

export default HomeScreen;

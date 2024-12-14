import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import HomeScreen from './src/screens/HomeScreen';
import StockDetailScreen from './src/screens/StockDetailScreen';
import BacktestScreen from './src/screens/BacktestScreen';
import NewsScreen from './src/screens/NewsScreen';

const Stack = createStackNavigator();

const App = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Home">
        <Stack.Screen 
          name="Home" 
          component={HomeScreen} 
          options={{ title: 'Stock Analysis' }}
        />
        <Stack.Screen 
          name="StockDetail" 
          component={StockDetailScreen} 
          options={({ route }) => ({ title: route.params?.symbol || 'Stock Detail' })}
        />
        <Stack.Screen 
          name="Backtest" 
          component={BacktestScreen} 
          options={{ title: 'Backtesting' }}
        />
        <Stack.Screen 
          name="News" 
          component={NewsScreen} 
          options={{ title: 'Market News' }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default App;

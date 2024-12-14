import React from 'react';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import HomeScreen from './src/screens/HomeScreen';
import StockDetailScreen from './src/screens/StockDetailScreen';
import BacktestScreen from './src/screens/BacktestScreen';
import NewsScreen from './src/screens/NewsScreen';

// Initialize navigation
const Stack = createStackNavigator();

const App = () => {
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <SafeAreaProvider>
        <NavigationContainer>
          <Stack.Navigator 
            initialRouteName="Home"
            screenOptions={{
              headerStyle: {
                backgroundColor: '#262730',
              },
              headerTintColor: '#fff',
              headerTitleStyle: {
                fontWeight: 'bold',
              },
            }}
          >
            <Stack.Screen 
              name="Home" 
              component={HomeScreen} 
              options={{ title: 'Stock Analysis' }}
            />
            <Stack.Screen 
              name="StockDetail" 
              component={StockDetailScreen} 
              options={({ route }) => ({ 
                title: route.params?.symbol || 'Stock Detail',
                headerBackTitleVisible: false,
              })}
            />
            <Stack.Screen 
              name="Backtest" 
              component={BacktestScreen} 
              options={{ 
                title: 'Backtesting',
                headerBackTitleVisible: false,
              }}
            />
            <Stack.Screen 
              name="News" 
              component={NewsScreen} 
              options={{ 
                title: 'Market News',
                headerBackTitleVisible: false,
              }}
            />
          </Stack.Navigator>
        </NavigationContainer>
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
};

export default App;

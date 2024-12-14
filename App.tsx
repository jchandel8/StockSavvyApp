import React from 'react';
import { StyleSheet, View } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { NavigationContainer, DefaultTheme } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import HomeScreen from './src/screens/HomeScreen';
import StockDetailScreen from './src/screens/StockDetailScreen';
import BacktestScreen from './src/screens/BacktestScreen';
import NewsScreen from './src/screens/NewsScreen';

export type RootStackParamList = {
  Home: undefined;
  StockDetail: { symbol: string };
  Backtest: { symbol: string };
  News: { symbol: string };
};

const Stack = createStackNavigator<RootStackParamList>();

const MyTheme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    background: '#0E1117',
    primary: '#FF4B4B',
    card: '#262730',
    text: '#FFFFFF',
    border: '#262730',
  },
};

const App = () => {
  return (
    <View style={styles.container}>
      <GestureHandlerRootView style={styles.container}>
        <SafeAreaProvider>
          <NavigationContainer theme={MyTheme}>
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
                cardStyle: { backgroundColor: '#0E1117' },
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
                  title: route.params.symbol,
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
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0E1117',
  },
});

export default App;

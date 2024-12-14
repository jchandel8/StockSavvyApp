import 'react-native-gesture-handler';
import { AppRegistry } from 'react-native';
import { enableScreens } from 'react-native-screens';
import App from './App';

// Enable native screens for better performance
enableScreens();

// Register the app
AppRegistry.registerComponent('StockSavvyApp', () => App);

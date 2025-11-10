import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import PostListScreen from '@screens/PostListScreen';
import PostMapScreen from '@screens/PostMapScreen';

export type MainTabsParamList = {
  Posts: undefined;
  Map: undefined;
};

const Tab = createBottomTabNavigator<MainTabsParamList>();

const iconNameForRoute = (routeName: keyof MainTabsParamList): keyof typeof Ionicons.glyphMap => {
  switch (routeName) {
    case 'Map':
      return 'map';
    case 'Posts':
    default:
      return 'list';
  }
};

const MainTabs = (): JSX.Element => {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarActiveTintColor: '#2563eb',
        tabBarInactiveTintColor: '#6b7280',
        tabBarIcon: ({ color, size }) => (
          <Ionicons name={iconNameForRoute(route.name)} size={size} color={color} />
        )
      })}
    >
      <Tab.Screen name="Posts" component={PostListScreen} />
      <Tab.Screen name="Map" component={PostMapScreen} />
    </Tab.Navigator>
  );
};

export default MainTabs;

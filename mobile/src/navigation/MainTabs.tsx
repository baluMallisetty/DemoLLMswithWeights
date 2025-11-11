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
        tabBarLabelStyle: {
          fontSize: 12,
          fontWeight: '600',
          marginBottom: 4
        },
        tabBarStyle: {
          backgroundColor: '#ffffff',
          borderTopWidth: 1,
          borderTopColor: '#e5e7eb',
          height: 64,
          paddingVertical: 6
        },
        tabBarIcon: ({ color, size }) => (
          <Ionicons name={iconNameForRoute(route.name)} size={size} color={color} />
        )
      })}
    >
      <Tab.Screen
        name="Posts"
        component={PostListScreen}
        options={{
          tabBarLabel: 'Feed',
          tabBarTestID: 'feed-tab'
        }}
      />
      <Tab.Screen
        name="Map"
        component={PostMapScreen}
        options={{
          tabBarLabel: 'Map View',
          tabBarTestID: 'map-tab'
        }}
      />
    </Tab.Navigator>
  );
};

export default MainTabs;

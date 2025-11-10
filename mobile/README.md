# Demo Posts Map

This Expo-based React Native example demonstrates how to surface a map view of posts alongside the traditional list experience. The bottom tab navigator exposes two primary entry points:

- **Posts** – a feed-style list of cards rendered with mock post data.
- **Map** – a map visualisation that plots every post using its latitude and longitude.

## Key implementation details

- Post data lives in [`src/data/posts.ts`](src/data/posts.ts) and contains rich metadata plus geocoordinates.
- [`PostMapScreen`](src/screens/PostMapScreen.tsx) computes an appropriate initial region for the map and drops a marker per post.
- [`MainTabs`](src/navigation/MainTabs.tsx) wires the list and map screens into a bottom tab bar.

## Getting started

Install dependencies and run the Expo development server:

```bash
cd mobile
npm install
npm run start
```

Running on iOS or Android requires completing the standard Expo native setup for [`react-native-maps`](https://github.com/react-native-maps/react-native-maps#installation-and-setup).

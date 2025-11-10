import React, { useMemo } from 'react';
import { StyleSheet, View, Text } from 'react-native';
import MapView, { Marker } from 'react-native-maps';
import { posts } from '@data/posts';

const PostMapScreen: React.FC = () => {
  const region = useMemo(() => {
    if (posts.length === 0) {
      return undefined;
    }

    const latitudes = posts.map((post) => post.latitude);
    const longitudes = posts.map((post) => post.longitude);

    const minLat = Math.min(...latitudes);
    const maxLat = Math.max(...latitudes);
    const minLng = Math.min(...longitudes);
    const maxLng = Math.max(...longitudes);

    const latitude = (minLat + maxLat) / 2;
    const longitude = (minLng + maxLng) / 2;

    const latitudeDelta = Math.max(0.05, maxLat - minLat + 0.02);
    const longitudeDelta = Math.max(0.05, maxLng - minLng + 0.02);

    return { latitude, longitude, latitudeDelta, longitudeDelta };
  }, []);

  if (!region) {
    return (
      <View style={styles.emptyState}>
        <Text style={styles.emptyHeading}>No posts to show on the map</Text>
        <Text style={styles.emptyDescription}>
          Add new posts with location data to visualize them here.
        </Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <MapView style={StyleSheet.absoluteFill} initialRegion={region}>
        {posts.map((post) => (
          <Marker
            key={post.id}
            coordinate={{ latitude: post.latitude, longitude: post.longitude }}
            title={post.title}
            description={`${post.author} • ${new Date(post.publishedAt).toLocaleDateString()}`}
          />
        ))}
      </MapView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1
  },
  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 24,
    backgroundColor: '#f9fafb'
  },
  emptyHeading: {
    fontSize: 20,
    fontWeight: '600',
    color: '#111827',
    marginBottom: 8,
    textAlign: 'center'
  },
  emptyDescription: {
    fontSize: 14,
    color: '#6b7280',
    textAlign: 'center'
  }
});

export default PostMapScreen;

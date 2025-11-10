import React from 'react';
import { FlatList, View, StyleSheet, Text } from 'react-native';
import PostCard from '@components/PostCard';
import { posts } from '@data/posts';

const PostListScreen: React.FC = () => {
  return (
    <View style={styles.container}>
      <Text style={styles.heading}>Latest Posts</Text>
      <FlatList
        data={posts}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.listContent}
        renderItem={({ item }) => <PostCard post={item} />}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f3f4f6'
  },
  heading: {
    fontSize: 24,
    fontWeight: '700',
    marginTop: 16,
    marginHorizontal: 20,
    marginBottom: 12,
    color: '#111827'
  },
  listContent: {
    paddingHorizontal: 20,
    paddingBottom: 24
  }
});

export default PostListScreen;

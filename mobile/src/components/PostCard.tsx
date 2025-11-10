import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import type { Post } from '@data/posts';

type PostCardProps = {
  post: Post;
};

const PostCard: React.FC<PostCardProps> = ({ post }) => {
  return (
    <View style={styles.card}>
      <Text style={styles.title}>{post.title}</Text>
      <Text style={styles.meta}>
        By {post.author} • {new Date(post.publishedAt).toLocaleDateString()}
      </Text>
      <Text style={styles.excerpt}>{post.excerpt}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOpacity: 0.08,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 4 },
    elevation: 3
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
    color: '#111827',
    marginBottom: 6
  },
  meta: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 8
  },
  excerpt: {
    fontSize: 14,
    color: '#374151'
  }
});

export default PostCard;

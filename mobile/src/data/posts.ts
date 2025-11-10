export type Post = {
  id: string;
  title: string;
  excerpt: string;
  author: string;
  publishedAt: string;
  latitude: number;
  longitude: number;
};

export const posts: Post[] = [
  {
    id: '1',
    title: 'Sunrise Yoga in the Park',
    excerpt: 'Join us for an energizing morning session surrounded by nature.',
    author: 'Alex Carter',
    publishedAt: '2024-03-02T09:00:00.000Z',
    latitude: 37.7694,
    longitude: -122.4862
  },
  {
    id: '2',
    title: 'Local Farmers Market Finds',
    excerpt: 'A curated list of seasonal produce and artisan goods available this week.',
    author: 'Priya Desai',
    publishedAt: '2024-03-05T15:30:00.000Z',
    latitude: 37.8008,
    longitude: -122.4376
  },
  {
    id: '3',
    title: 'Coffee Crawl: Hidden Gems',
    excerpt: 'Discover new roasteries and cafés with unique brews across the city.',
    author: 'Jordan Lee',
    publishedAt: '2024-03-07T11:15:00.000Z',
    latitude: 37.7599,
    longitude: -122.4148
  },
  {
    id: '4',
    title: 'Street Art Walking Tour',
    excerpt: 'Explore vibrant murals and learn the stories behind the artists.',
    author: 'Maya Thompson',
    publishedAt: '2024-03-09T13:45:00.000Z',
    latitude: 37.7617,
    longitude: -122.4241
  }
];

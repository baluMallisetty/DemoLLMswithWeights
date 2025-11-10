module.exports = function (api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: [
      [
        'module-resolver',
        {
          extensions: ['.ts', '.tsx', '.js', '.json'],
          alias: {
            '@screens': './src/screens',
            '@components': './src/components',
            '@data': './src/data',
            '@navigation': './src/navigation'
          }
        }
      ],
      'react-native-reanimated/plugin'
    ]
  };
};

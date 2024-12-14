const { getDefaultConfig } = require('@react-native/metro-config');

module.exports = (async () => {
  const defaultConfig = await getDefaultConfig(__dirname);
  
  return {
    ...defaultConfig,
    transformer: {
      ...defaultConfig.transformer,
      babelTransformerPath: require.resolve('react-native-svg-transformer'),
      experimentalImportSupport: false,
      inlineRequires: true,
    },
    resolver: {
      ...defaultConfig.resolver,
      assetExts: defaultConfig.resolver.assetExts.filter((ext) => ext !== 'svg'),
      sourceExts: [...defaultConfig.resolver.sourceExts, 'svg', 'ts', 'tsx'],
      platforms: ['ios', 'android', 'web'],
      blockList: [
        /\.git\/.*/,
        /\.cache\/.*/,
        /node_modules\/.*\/node_modules\/react-native\/.*/,
      ],
    },
    watchFolders: [__dirname],
    resetCache: true,
    maxWorkers: 4,
    server: {
      port: 8081,
      enhanceMiddleware: (middleware) => {
        return (req, res, next) => {
          res.setTimeout(120000);
          return middleware(req, res, next);
        };
      }
    }
  };
})();

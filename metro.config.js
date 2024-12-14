const { getDefaultConfig } = require('@react-native/metro-config');

module.exports = (async () => {
  const config = await getDefaultConfig(__dirname);
  const { transformer, resolver } = config;

  return {
    transformer: {
      ...transformer,
      babelTransformerPath: require.resolve('react-native-svg-transformer')
    },
    resolver: {
      ...resolver,
      assetExts: resolver.assetExts.filter((ext) => ext !== 'svg'),
      sourceExts: [...resolver.sourceExts, 'svg'],
      blockList: [
        // Exclude python-related files that cause conflicts
        /\.pythonlibs\/.*/,
        /\.cache\/.*/
      ]
    },
    watchFolders: [__dirname],
    resetCache: true,
    maxWorkers: 2
  };
})();

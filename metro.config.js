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
      // Ensure we're not picking up duplicate modules from node_modules
      blacklistRE: /\.cache\/.*$/
    },
    watchFolders: [__dirname],
    // Exclude cache directories from Metro
    resetCache: true
  };
})();

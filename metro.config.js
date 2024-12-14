const { getDefaultConfig } = require('@react-native/metro-config');

module.exports = (async () => {
  const config = await getDefaultConfig(__dirname);
  const { transformer, resolver } = config;

  return {
    transformer: {
      ...transformer,
      babelTransformerPath: require.resolve('react-native-svg-transformer'),
      experimentalImportSupport: false,
      inlineRequires: true,
    },
    resolver: {
      ...resolver,
      assetExts: resolver.assetExts.filter((ext) => ext !== 'svg'),
      sourceExts: [...resolver.sourceExts, 'svg'],
      blockList: [
        /\.pythonlibs\/.*/,
        /\.cache\/.*/,
        /\.git\/.*/,
        /\.venv\/.*/
      ],
      extraNodeModules: new Proxy({}, {
        get: (target, name) => {
          return name in target ? target[name] : process.cwd() + '/node_modules/' + name;
        }
      })
    },
    watchFolders: [__dirname],
    resetCache: true,
    maxWorkers: 2,
    server: {
      port: 8081,
      enhanceMiddleware: (middleware) => {
        return (req, res, next) => {
          // Set higher timeout
          res.setTimeout(60000);
          return middleware(req, res, next);
        };
      }
    }
  };
})();

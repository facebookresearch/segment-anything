// development config
const { merge } = require("webpack-merge");
const commonConfig = require("./common");

module.exports = merge(commonConfig, {
  mode: "development",
  devServer: {
    hot: true, // enable HMR on the server
    open: true,
    // These headers enable the cross origin isolation state
    // needed to enable use of SharedArrayBuffer for ONNX 
    // multithreading. 
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "credentialless",
    },
  },
  devtool: "cheap-module-source-map",
});

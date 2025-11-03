import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  publicDir: resolve(__dirname, "public"), // serve static assets from /public
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:3000"
    },
    watch: {
      usePolling: true, // more aggressive watching
      interval: 100,    // ms
    }
  }
});
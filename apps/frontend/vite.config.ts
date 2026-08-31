import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  // Dev-only: pre-bundle agentation at server start so the first page load does
  // not trigger a mid-session re-optimize (stale dep chunk errors).
  optimizeDeps: mode === "development" ? { include: ["agentation"] } : undefined,
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8010",
        changeOrigin: true,
        // The transcript surface streams microphone PCM to a streaming ASR engine
        // over a WebSocket. Without this the upgrade request is silently dropped:
        // the socket never opens AND never errors, so a caller awaiting it waits
        // forever with nothing to report. Every other route worked, which made it
        // look like a UI bug rather than a proxy gap.
        ws: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
}));

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev the page runs on :5173 and proxies the API (run `python -m dronesim.api` on :8000).
export default defineConfig({
  plugins: [react()],
  // The bundle is ~1.2 MB because it contains three.js; the app is served locally, so code-splitting is not worth it.
  build: { chunkSizeWarningLimit: 1500 },
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});

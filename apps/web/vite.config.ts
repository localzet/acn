import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          charts: ["recharts"],
          flow: ["@xyflow/react"],
        },
      },
    },
  },
  server: {
    host: "0.0.0.0",
    port: 5173
  }
});

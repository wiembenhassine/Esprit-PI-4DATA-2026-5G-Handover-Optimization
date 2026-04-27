import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Dev proxy: /api → FastAPI gateway (only used with `npm run dev`)
    // In Docker, nginx handles this proxying instead
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api/, ''),
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    // Increase chunk warning limit (recharts is large)
    chunkSizeWarningLimit: 800,
  },
})

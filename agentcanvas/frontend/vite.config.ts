import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// BACKEND_URL overrides the dev-proxy target (same convention as
// /experiment:run's wrapped commands). Default stays the user's :8000.
const backend = process.env.BACKEND_URL || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: backend,
        changeOrigin: true,
      },
      '/ws': {
        target: backend.replace(/^http/, 'ws'),
        ws: true,
      },
    },
  },
})

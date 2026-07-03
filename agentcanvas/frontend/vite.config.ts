import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Backend the dev proxy forwards /api + /ws to. Override to pair this dev
// frontend with a non-default backend instance:
//   VITE_PROXY_TARGET=http://127.0.0.1:5175 npx vite --host
const proxyTarget = process.env.VITE_PROXY_TARGET || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: proxyTarget,
        changeOrigin: true,
      },
      '/ws': {
        target: proxyTarget.replace(/^http/, 'ws'),
        ws: true,
      },
    },
  },
})

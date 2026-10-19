import process from 'node:process'
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Vite dev server config. The proxy forwards every /api request to
// the FastAPI backend so the browser sees same-origin requests
// during development and CORS does not get in the way.
//
// Set VITE_API_URL to point at a non-default backend (e.g. when
// running on a remote box or a different port). Defaults to
// http://localhost:8000 to match dev.sh / startup-tmux.sh.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiUrl = env.VITE_API_URL || 'http://localhost:8000'
  return {
    plugins: [react(), tailwindcss()],
    server: {
      proxy: {
        '/api': apiUrl,
      },
    },
  }
})

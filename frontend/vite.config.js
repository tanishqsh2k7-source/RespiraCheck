import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,         // match the port expected by Flask CORS config
    open: true,         // auto-open browser on npm run dev
  },
})

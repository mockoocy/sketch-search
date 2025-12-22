import tailwindcss from "@tailwindcss/vite"
import react from '@vitejs/plugin-react'
import path from "path"
import { defineConfig } from 'vite'
// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react({
      babel: {
        plugins: [['babel-plugin-react-compiler']],
      },
    }),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    }
  },
  server: {
    proxy: {
      "/api/events": {
        target: "http://server:8000",
        configure: (proxy) => {
          proxy.on("proxyRes", (proxyRes) => {
            proxyRes.headers["cache-control"] = "no-cache"
          })
          proxy.on("error", (err) => {
            console.error("Proxy error:", err)
          })
          proxy.onError = (err, req, res) => {
            console.error("Proxy onError:", err)
            res.writeHead(500, {
              "Content-Type": "text/plain",
            })
            res.end("Something went wrong. And we are reporting a custom error message.")
          }
        },
      },
        '/api': {
          target: 'http://server:8000',
          configure: (proxy) => {
            proxy.on('error', (err) => {
              console.error('Proxy error:', err)
            })
          }
        }

      }
    }
})

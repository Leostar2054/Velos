import type { Config } from "tailwindcss";

export default {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { 950: "#070b18", 900: "#0b1020", 800: "#111a33", 700: "#1a2547", 600: "#273459" },
        cyan: { 300: "#67e8f9", 400: "#22d3ee", 500: "#06b6d4" },
        violet: { 400: "#a78bfa", 500: "#8b5cf6", 600: "#7c3aed" },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"],
      },
      boxShadow: { glow: "0 0 40px rgba(34, 211, 238, 0.15)" },
    },
  },
  plugins: [],
} satisfies Config;

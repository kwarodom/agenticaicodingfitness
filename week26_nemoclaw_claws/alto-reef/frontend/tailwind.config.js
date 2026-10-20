/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0a1729", panel: "#10213a", panel2: "#16304f", line: "rgba(120,200,255,0.18)",
        teal: "#3fd0c9", blue: "#4aa3ff", amber: "#f2b64c", magenta: "#ff5c8a", nv: "#76b900",
        ink: "#e6f1ff", muted: "#8fb0d0",
      },
      fontFamily: { sans: ["Inter", "ui-sans-serif", "system-ui"], mono: ["JetBrains Mono", "ui-monospace", "monospace"] },
      borderRadius: { md: "8px", lg: "10px" },
    },
  },
  plugins: [],
};

import type { NextConfig } from "next";

// Static export: the site is plain HTML/JS/CSS that can be hosted on any CDN or
// behind nginx; the contact form talks to the backend API via NEXT_PUBLIC_API_URL.
const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;

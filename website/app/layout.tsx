import type { Metadata } from "next";
import "./globals.css";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import Analytics from "@/components/Analytics";
import { SITE, SITE_URL } from "@/lib/site";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: { default: `${SITE.company.name} — ${SITE.company.tagline}`, template: `%s · ${SITE.company.name}` },
  description: SITE.company.description,
  keywords: ["quantitative phase imaging", "QPI", "ptychography", "label-free", "live-cell imaging", "drug discovery", "SPARK", "Velos QPI"],
  openGraph: {
    type: "website", siteName: SITE.company.name, title: SITE.company.name, description: SITE.company.description,
    images: [{ url: "/images/velos-qpi-concept.jpg", width: 1800, height: 1012, alt: "Velos QPI product concept" }],
  },
  twitter: { card: "summary_large_image" },
  robots: { index: true, follow: true },
  alternates: { canonical: "/" },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col">
        <Nav />
        <main className="flex-1">{children}</main>
        <Footer />
        <Analytics />
      </body>
    </html>
  );
}

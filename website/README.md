# Velos BioImaging website (Module 1)

Next.js 15 (App Router) + Tailwind CSS, exported as a static site (`out/`) that can be hosted on any CDN,
nginx (see `Dockerfile`) or object storage.

```bash
npm install
cp .env.example .env.local          # NEXT_PUBLIC_API_URL, NEXT_PUBLIC_SITE_URL, optional analytics ids
npm run dev                         # http://localhost:3000
npm run build                       # static export → out/
```

## Editing content without touching code
All copy, team members, navigation, contact details and the status note live in **`content/site.json`**.
Images live in `public/images/`. Changing text = edit the JSON, rebuild.

## Pages
`/` Home · `/product` Product (hardware, SPARK modes, platform features, targets) · `/team` · `/contact`
(form posts to `POST {NEXT_PUBLIC_API_URL}/api/v1/contact`, with honeypot; shows a mailto fallback on error).

## SEO / analytics
Per-page metadata, Open Graph image, canonical URLs, `sitemap.xml`, `robots.txt`.
Analytics loads only when `NEXT_PUBLIC_PLAUSIBLE_DOMAIN` or `NEXT_PUBLIC_GA_ID` is set at build time.

## Evidence (M2)
`../evidence/website/` — full-page screenshots at 1440 px and 390 px for all four pages, and the contact form
before/after submission with the lead stored in the backend.

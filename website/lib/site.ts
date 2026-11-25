import site from "@/content/site.json";

export type Site = typeof site;
export const SITE: Site = site;

// Contact address shown on the site: set at build time (NEXT_PUBLIC_CONTACT_EMAIL) so that no personal
// address has to live in the public repository; the block is hidden when it is empty.
export const CONTACT_EMAIL = process.env.NEXT_PUBLIC_CONTACT_EMAIL ?? site.company.email ?? "";
export const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? "https://velosbioimaging.example";
export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
export const PLAUSIBLE_DOMAIN = process.env.NEXT_PUBLIC_PLAUSIBLE_DOMAIN ?? "";
export const GA_ID = process.env.NEXT_PUBLIC_GA_ID ?? "";

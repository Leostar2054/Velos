import type { MetadataRoute } from "next";
import { SITE, SITE_URL } from "@/lib/site";

export const dynamic = "force-static";

export default function sitemap(): MetadataRoute.Sitemap {
  const now = new Date();
  return SITE.nav.map((i) => ({
    url: `${SITE_URL}${i.href === "/" ? "/" : `${i.href}/`}`,
    lastModified: now,
    changeFrequency: "monthly",
    priority: i.href === "/" ? 1 : 0.8,
  }));
}

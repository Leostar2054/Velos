#!/usr/bin/env node
/**
 * Website QA crawl (M3): every page × phone / tablet / desktop viewport.
 *
 *   node scripts/website_qa.js http://localhost:3100 out_dir [path/to/axe.min.js]
 *
 * Checks per page and viewport:
 *   - console errors, uncaught page errors, failed requests (HTTP >= 400)
 *   - horizontal overflow (document wider than the viewport) and the elements causing it
 *   - broken images, images without alt text, duplicate element ids
 *   - internal links that do not resolve (HTTP status)
 *   - accessibility (axe-core, WCAG 2.1 A/AA rules) when axe.min.js is given
 *   - title / meta description / lang / h1 count
 * Writes qa_report.json, qa_report.md and a screenshot per page × viewport.
 */
const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const BASE = process.argv[2] || "http://localhost:3100";
const OUT = process.argv[3] || "qa_out";
const AXE = process.argv[4];
const PAGES = ["/", "/product/", "/team/", "/contact/"];
const VIEWPORTS = [
  { name: "phone-390", width: 390, height: 844 },
  { name: "tablet-768", width: 768, height: 1024 },
  { name: "desktop-1440", width: 1440, height: 900 },
];

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const report = { base: BASE, date: new Date().toISOString(), results: [], links: {} };
  const linkStatus = {};
  for (const vp of VIEWPORTS) {
    const ctx = await browser.newContext({ viewport: { width: vp.width, height: vp.height } });
    for (const p of PAGES) {
      const page = await ctx.newPage();
      const consoleErrors = [], pageErrors = [], failed = [];
      page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });
      page.on("pageerror", (e) => pageErrors.push(String(e)));
      page.on("response", (r) => { if (r.status() >= 400) failed.push(`${r.status()} ${r.url()}`); });
      await page.goto(BASE + p, { waitUntil: "networkidle" });
      await page.waitForTimeout(300);
      const info = await page.evaluate(() => {
        const vw = document.documentElement.clientWidth;
        const overflow = document.documentElement.scrollWidth > vw + 1;
        const offenders = [];
        if (overflow) {
          for (const el of document.querySelectorAll("body *")) {
            const r = el.getBoundingClientRect();
            if (r.right > vw + 1 && r.width > 0) {
              offenders.push(`${el.tagName.toLowerCase()}${el.id ? "#" + el.id : ""}.${String(el.className).slice(0, 60)} right=${Math.round(r.right)}`);
              if (offenders.length >= 8) break;
            }
          }
        }
        const imgs = [...document.images];
        const ids = {};
        document.querySelectorAll("[id]").forEach((e) => (ids[e.id] = (ids[e.id] || 0) + 1));
        return {
          title: document.title,
          description: document.querySelector('meta[name="description"]')?.content || "",
          lang: document.documentElement.lang,
          h1: document.querySelectorAll("h1").length,
          overflow, scrollWidth: document.documentElement.scrollWidth, viewport: vw, offenders,
          images: imgs.length,
          brokenImages: imgs.filter((i) => i.complete && i.naturalWidth === 0).map((i) => i.currentSrc || i.src),
          missingAlt: imgs.filter((i) => !i.hasAttribute("alt")).map((i) => i.currentSrc || i.src),
          duplicateIds: Object.entries(ids).filter(([, n]) => n > 1).map(([id, n]) => `${id} ×${n}`),
          links: [...document.querySelectorAll("a[href]")].map((a) => a.getAttribute("href")),
          favicon: !!document.querySelector('link[rel~="icon"]'),
        };
      });
      let axe = null;
      if (AXE) {
        await page.addScriptTag({ path: AXE });
        axe = await page.evaluate(async () => {
          const r = await window.axe.run(document, { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"] } });
          return r.violations.map((v) => ({ id: v.id, impact: v.impact, help: v.help, nodes: v.nodes.length,
                                            targets: v.nodes.slice(0, 3).map((n) => n.target.join(" ")) }));
        });
      }
      for (const href of info.links) {
        if (href.startsWith("/") && !(href in linkStatus)) linkStatus[href] = null;
      }
      const shot = path.join(OUT, `${(p === "/" ? "home" : p.replace(/\//g, ""))}_${vp.name}.png`);
      await page.screenshot({ path: shot, fullPage: true });
      report.results.push({ page: p, viewport: vp.name, consoleErrors, pageErrors, failedRequests: failed,
                            ...info, links: undefined, axe, screenshot: path.basename(shot) });
      await page.close();
    }
    await ctx.close();
  }
  // internal links
  const ctx = await browser.newContext();
  for (const href of Object.keys(linkStatus)) {
    const r = await ctx.request.get(BASE + href.split("#")[0]);
    linkStatus[href] = r.status();
  }
  report.links = linkStatus;
  await browser.close();

  // summary
  const issues = [];
  for (const r of report.results) {
    const tag = `${r.page} @ ${r.viewport}`;
    if (r.consoleErrors.length) issues.push(`${tag}: console errors: ${r.consoleErrors.join(" | ")}`);
    if (r.pageErrors.length) issues.push(`${tag}: page errors: ${r.pageErrors.join(" | ")}`);
    if (r.failedRequests.length) issues.push(`${tag}: failed requests: ${r.failedRequests.join(", ")}`);
    if (r.overflow) issues.push(`${tag}: horizontal overflow (${r.scrollWidth}px > ${r.viewport}px): ${r.offenders.join("; ")}`);
    if (r.brokenImages.length) issues.push(`${tag}: broken images: ${r.brokenImages.join(", ")}`);
    if (r.missingAlt.length) issues.push(`${tag}: images without alt: ${r.missingAlt.join(", ")}`);
    if (r.duplicateIds.length) issues.push(`${tag}: duplicate ids: ${r.duplicateIds.join(", ")}`);
    if (r.h1 !== 1) issues.push(`${tag}: ${r.h1} <h1> elements`);
    if (!r.description) issues.push(`${tag}: no meta description`);
    if (!r.lang) issues.push(`${tag}: no lang attribute`);
    if (!r.favicon) issues.push(`${tag}: no favicon link`);
    for (const v of r.axe || []) issues.push(`${tag}: axe ${v.impact} — ${v.id}: ${v.help} (${v.nodes} nodes: ${v.targets.join(", ")})`);
  }
  for (const [href, st] of Object.entries(linkStatus)) if (st >= 400) issues.push(`link ${href} → HTTP ${st}`);
  report.issues = issues;
  fs.writeFileSync(path.join(OUT, "qa_report.json"), JSON.stringify(report, null, 1));
  const md = [`# Website QA report`, ``, `Base: ${BASE} · ${report.date}`, ``,
    `Pages: ${PAGES.join(", ")} · viewports: ${VIEWPORTS.map((v) => v.name).join(", ")} · internal links checked: ${Object.keys(linkStatus).length}`, ``,
    `| Page | Viewport | Console/page errors | Failed requests | Overflow | Broken img | No alt | Dup ids | axe violations |`,
    `|---|---|---|---|---|---|---|---|---|`,
    ...report.results.map((r) => `| ${r.page} | ${r.viewport} | ${r.consoleErrors.length + r.pageErrors.length} | ${r.failedRequests.length} | ${r.overflow ? "YES" : "no"} | ${r.brokenImages.length} | ${r.missingAlt.length} | ${r.duplicateIds.length} | ${r.axe ? r.axe.length : "n/a"} |`),
    ``, `## Issues (${issues.length})`, ``, ...(issues.length ? issues.map((i) => `- ${i}`) : ["None."]), ``];
  fs.writeFileSync(path.join(OUT, "qa_report.md"), md.join("\n"));
  console.log(md.slice(0, 22).join("\n"));
  console.log(`issues: ${issues.length}`);
})();

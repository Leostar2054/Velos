// Rasterise the SVG diagrams to PNG (2x) with Playwright: node scripts/render_diagrams.js <svgdir> <outdir>
const { chromium } = require('playwright'); const fs = require('fs'); const path = require('path');
(async () => {
  const [svgDir, outDir] = process.argv.slice(2); fs.mkdirSync(outDir, { recursive: true });
  const browser = await chromium.launch();
  for (const name of ['architecture', 'spark-workflow', 'validation-roadmap']) {
    const svg = fs.readFileSync(path.join(svgDir, name + '.svg'), 'utf8');
    const m = svg.match(/viewBox="0 0 (\d+) (\d+)"/); const w = +m[1], h = +m[2];
    const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 2 });
    await page.setContent(`<html><body style="margin:0;background:#0b1020">${svg}</body></html>`);
    await page.waitForTimeout(300);
    await page.screenshot({ path: path.join(outDir, name + '.png') });
    console.log('rendered', name, w, h);
    await page.close();
  }
  await browser.close();
})();

// Real end-to-end: pick an UNPROCESSED video from the dropdown, process it (YOLO runs on the server), see the result.
// Usage: node e2e/process_video.mjs [videoStem] [baseUrl] [screenshotDir]      (server must be running)
import { chromium } from "playwright-core";
import fs from "node:fs";

const stem = process.argv[2] ?? "testv3";
const base = process.argv[3] ?? "http://127.0.0.1:8000";
const shots = process.argv[4] ?? ".";
const chromePath = ["C:/Program Files/Google/Chrome/Application/chrome.exe", "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"].find((p) => fs.existsSync(p));
const browser = await chromium.launch({ executablePath: chromePath, args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 1500, height: 1500 } });
const errors = [];
page.on("console", (m) => m.type() === "error" && !/status of 4\d\d/.test(m.text()) && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(String(e)));
let failed = 0;
const check = (name, ok, detail = "") => { console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  — " + detail : ""}`); if (!ok) failed++; };

await page.goto(base, { waitUntil: "networkidle" });
await page.waitForSelector('[data-testid="run-select"]');
const options = await page.$$eval('[data-testid="run-select"] option', (o) => o.map((x) => x.textContent.trim()));
console.log("dropdown:", JSON.stringify(options));
check("dropdown lists every video in the project folder", options.length >= 3 && options.some((o) => o.startsWith(`${stem}.`)), options.join(" | "));
check("each entry shows whether it is processed", options.every((o) => /processed/.test(o)));

await page.selectOption('[data-testid="run-select"]', stem);
await page.waitForSelector('[data-testid="process-panel"], [data-testid="motor-graphs"]', { timeout: 15000 });
const needs = await page.$('[data-testid="process-panel"]');
if (needs) {
  check("unprocessed video shows the Process panel with a preview", !!(await page.$('[data-testid="preview"]')));
  await page.screenshot({ path: `${shots}/proc_panel.png`, clip: { x: 0, y: 0, width: 1500, height: 700 } });
  await page.click('[data-testid="process-btn"]');
  await page.waitForSelector('[data-testid="job-status"]', { timeout: 10000 });
  const t0 = Date.now();
  await page.waitForSelector('[data-testid="motor-graphs"]', { timeout: 420000 });
  console.log(`processed in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
} else {
  console.log("(already processed; skipping the processing step)");
}
await page.waitForSelector('[data-testid="hud"]');
await page.waitForTimeout(1500);
check("after processing the 3D scene, HUD and four motor graphs are shown", (await page.$$('[data-testid^="motor-graph-"]')).length === 4 && !!(await page.$('[data-testid="scene-canvas"], canvas')));
const sel = await page.$eval('[data-testid="run-select"]', (s) => s.options[s.selectedIndex].textContent.trim());
check("dropdown now marks it processed", /✓ processed/.test(sel), sel);
const other = await page.$('[data-testid="other-tracks"]');
check("other detected objects are listed", !!other, other ? (await other.textContent()).trim().slice(0, 120) : "none");
const plan = (await page.textContent('[data-testid="plan-note"]')).trim();
console.log("plan:", plan);
await page.fill('[data-testid="scrubber"]', "4");
await page.waitForTimeout(800);
await page.screenshot({ path: `${shots}/proc_result.png` });
// switching between videos repeatedly must not break the 3D scene (this used to throw "removeChild")
for (const target of ["testv2", stem, "testv2", stem]) {
  await page.selectOption('[data-testid="run-select"]', target);
  await page.waitForSelector('[data-testid="hud"]', { timeout: 20000 });
  await page.waitForTimeout(700);
}
check("switching videos back and forth keeps the scene working", (await page.$$('[data-testid^="motor-graph-"]')).length === 4);
const lbl = await page.$('[data-testid="clear-label"]');
check("clearance label is drawn as a normal overlay", !!lbl && /m$/.test((await lbl.textContent()) ?? ""), lbl ? (await lbl.textContent()).trim() : "missing");
check("no console errors", errors.length === 0, errors.slice(0, 3).join(" | "));
await browser.close();
console.log(failed ? `\n${failed} FAILED` : "\nALL CHECKS PASSED");
process.exit(failed ? 1 : 0);

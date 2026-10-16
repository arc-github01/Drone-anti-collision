// Checks the HUD attitude indicator against the drone's real pose in a tailwind (it leans back = nose UP).
// Usage: node e2e/hud_pitch.mjs [baseUrl] [screenshotDir]   (server must be running)
import { chromium } from "playwright-core";
import fs from "node:fs";

const base = process.argv[2] ?? "http://127.0.0.1:8000";
const shots = process.argv[3] ?? ".";
const chromePath = ["C:/Program Files/Google/Chrome/Application/chrome.exe", "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"].find((p) => fs.existsSync(p));
const browser = await chromium.launch({ executablePath: chromePath, args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
await page.goto(base, { waitUntil: "networkidle" });
await page.waitForSelector('[data-testid="hud"]');

const run = async (az, el) => {
  await page.fill('[data-testid="wind-speed"]', "8");
  await page.fill('[data-testid="wind-azimuth"]', String(az));
  await page.fill('[data-testid="wind-elevation"]', String(el));
  await page.click('[data-testid="simulate"]');
  await page.waitForFunction(() => !document.querySelector('[data-testid="wind-stale"]'), null, { timeout: 120000 });
  await page.fill('[data-testid="scrubber"]', "1");
  await page.waitForTimeout(600);
  const hud = (await page.textContent('[data-testid="hud"]')).replace(/\s+/g, " ");
  const pitch = +(hud.match(/Pitch ↑\s*(-?[\d.]+)°/) ?? [])[1];
  // horizon: the sky rect's y within the instrument (negative y = horizon moved down => more sky => nose up)
  const horizonLineY = await page.$eval('[data-testid="hud"] svg g[clip-path] g line', (l) => {
    const m = new DOMMatrixReadOnly(getComputedStyle(l.parentElement).transform);
    return m.f; // translateY applied to the horizon group
  });
  return { hud, pitch, horizonLineY };
};

let failed = 0;
const check = (name, ok, detail) => { console.log(`${ok ? "PASS" : "FAIL"}  ${name}  — ${detail}`); if (!ok) failed++; };

const tail = await run(0, 0); // tailwind: drag pushes the drone forward, so it leans BACK: nose up
await page.screenshot({ path: `${shots}/hud_tailwind.png`, clip: { x: 0, y: 40, width: 760, height: 220 } });
check("tailwind: pitch readout is positive (nose up)", tail.pitch > 5, `Pitch ↑ ${tail.pitch}°`);
check("tailwind: horizon moved DOWN (more sky)", tail.horizonLineY > 0, `horizon translateY ${tail.horizonLineY.toFixed(1)} px`);

const head = await run(180, 0); // headwind: leans FORWARD: nose down
await page.screenshot({ path: `${shots}/hud_headwind.png`, clip: { x: 0, y: 40, width: 760, height: 220 } });
check("headwind: pitch readout is negative (nose down)", head.pitch < -5, `Pitch ↑ ${head.pitch}°`);
check("headwind: horizon moved UP (more ground)", head.horizonLineY < 0, `horizon translateY ${head.horizonLineY.toFixed(1)} px`);

await browser.close();
console.log(failed ? `\n${failed} FAILED` : "\nALL CHECKS PASSED");
process.exit(failed ? 1 : 0);

// Headless-browser smoke test against a RUNNING server (python -m dronesim.api, serving frontend/dist).
// Usage: node e2e/smoke.mjs [baseUrl] [screenshotDir]
import { chromium } from "playwright-core";
import fs from "node:fs";

const base = process.argv[2] ?? "http://127.0.0.1:8000";
const shots = process.argv[3] ?? ".";
const chromePath = ["C:/Program Files/Google/Chrome/Application/chrome.exe", "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"].find((p) => fs.existsSync(p));
if (!chromePath) throw new Error("No Chrome/Edge found");

const failures = [];
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  — " + detail : ""}`);
  if (!ok) failures.push(name);
};

const browser = await chromium.launch({
  executablePath: chromePath,
  args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--autoplay-policy=no-user-gesture-required"],
});
const page = await browser.newPage({ viewport: { width: 1500, height: 1500 } });
const consoleErrors = [];
let expectedBad = 0; // the test deliberately sends one invalid request (422)
page.on("response", (r) => r.status() === 422 && expectedBad++);
page.on("console", (m) => m.type() === "error" && !/status of 422/.test(m.text()) && consoleErrors.push(m.text()));
page.on("pageerror", (e) => consoleErrors.push(String(e)));

await page.goto(base, { waitUntil: "networkidle" });
await page.waitForSelector('[data-testid="run-select"]');
await page.selectOption('[data-testid="run-select"]', "testv2"); // the scenarios below assume this video
await page.waitForSelector('[data-testid="motor-graphs"]', { timeout: 20000 });
await page.waitForTimeout(1500);

// 1. four graphs, directly below the 3D scene, full width of the left column
const graphs = await page.$$('[data-testid^="motor-graph-"]');
check("four motor graphs rendered", graphs.length === 4, `found ${graphs.length}`);
const sceneBox = await (await page.$('[data-testid="scene"]')).boundingBox();
const stackBox = await (await page.$('[data-testid="motor-graphs"]')).boundingBox();
check("graphs are directly below the 3D scene", stackBox.y >= sceneBox.y + sceneBox.height - 1 && stackBox.y - (sceneBox.y + sceneBox.height) < 40 && Math.abs(stackBox.x - sceneBox.x) < 2,
  `scene bottom ${Math.round(sceneBox.y + sceneBox.height)}, graphs top ${Math.round(stackBox.y)}`);
const gb = await Promise.all(graphs.map((g) => g.boundingBox()));
// DOM order is [M4 front-left, M1 front-right, M3 rear-left, M2 rear-right]: a 2 x 2 grid
const rowsOk = Math.abs(gb[0].y - gb[1].y) < 2 && Math.abs(gb[2].y - gb[3].y) < 2 && gb[2].y > gb[0].y + 100;
const colsOk = Math.abs(gb[0].x - gb[2].x) < 2 && Math.abs(gb[1].x - gb[3].x) < 2 && gb[1].x > gb[0].x + 100;
check("graphs form a 2 x 2 grid of equal rectangles", rowsOk && colsOk && gb.every((b) => Math.abs(b.width - gb[0].width) < 1 && Math.abs(b.height - gb[0].height) < 1),
  gb.map((b) => `(${Math.round(b.x)},${Math.round(b.y)} ${Math.round(b.width)}x${Math.round(b.height)})`).join(" "));
check("quadrants follow the drone layout (front motors on top)", (await page.$$eval('[data-testid^="motor-graph-"]', (els) => els.map((e) => e.getAttribute("data-testid")))).join() === "motor-graph-4,motor-graph-1,motor-graph-3,motor-graph-2");

// 2. the WebGL canvas really drew something (not blank)
const lit = await page.evaluate(() => {
  const c = document.querySelector("canvas");
  const t = document.createElement("canvas");
  t.width = 160; t.height = 90;
  const g = t.getContext("2d");
  g.drawImage(c, 0, 0, 160, 90);
  const d = g.getImageData(0, 0, 160, 90).data;
  const colors = new Set();
  for (let i = 0; i < d.length; i += 4) colors.add((d[i] >> 4) + "," + (d[i + 1] >> 4) + "," + (d[i + 2] >> 4));
  return colors.size;
});
check("3D canvas is rendering (many distinct colours)", lit > 12, `${lit} colour buckets`);

// 3. graph curves have real data and differ between motors
const paths = await page.$$eval('[data-testid^="motor-graph-"] path', (els) => els.map((e) => e.getAttribute("d") ?? ""));
check("each graph draws a curve", paths.length === 4 && paths.every((d) => d.length > 200), paths.map((d) => d.length).join(","));

// 4. scrubbing: time, playheads on all four graphs, HUD and readouts all move together
const readTime = async () => +((await page.textContent('[data-testid="time"]')).split("/")[0]);
const playheads = async () => page.$$eval('[data-testid^="motor-graph-"] .playhead', (els) => els.map((e) => +e.getAttribute("x1")));
const readouts = async () => page.$$eval('[data-testid^="motor-graph-"] .readout', (els) => els.map((e) => e.textContent));
await page.fill('[data-testid="scrubber"]', "0");
await page.waitForTimeout(300);
const ph0 = await playheads();
const status0 = await page.textContent('[data-testid="status"]');
await page.fill('[data-testid="scrubber"]', "8.4");
await page.waitForTimeout(400);
const t1 = await readTime();
const ph1 = await playheads();
check("scrubber sets the simulation time", Math.abs(t1 - 8.4) < 0.05, `time = ${t1}`);
check("all four playheads moved together", ph1.every((x, i) => x > ph0[i] + 50) && Math.max(...ph1) - Math.min(...ph1) < 0.5, `x: ${ph0[0].toFixed(0)} -> ${ph1.map((x) => x.toFixed(0)).join(",")}`);
const clr = await page.textContent('[data-testid="hud"]');
check("HUD shows clearance near the closest approach", /0\.3\d\d m/.test(clr), (clr.match(/Clearance(\S+)/) ?? [])[0] ?? clr.slice(0, 80));
await page.screenshot({ path: `${shots}/ui_t8.4.png` });

// clicking on a graph scrubs too
const box2 = await graphs[2].boundingBox();
await graphs[2].click({ position: { x: box2.width * 0.3, y: 40 } });
await page.waitForTimeout(300);
const t2 = await readTime();
check("clicking a graph scrubs the simulation", t2 > 1 && t2 < 12 && Math.abs(t2 - t1) > 0.5, `time = ${t2}`);

// 5. playback advances the clock from the recording
await page.fill('[data-testid="scrubber"]', "1");
await page.click('[data-testid="play"]');
await page.waitForTimeout(1500);
const tp = await readTime();
await page.click('[data-testid="play"]');
check("playback advances time", tp > 1.8, `time = ${tp}`);

// 6. video + detection telemetry present and synced
await page.fill('[data-testid="scrubber"]', "3");
await page.waitForTimeout(700);
const vtime = await page.$eval('[data-testid="video"]', (v) => v.currentTime);
check("video seeks to the synced time (reference 0.17 s + sim t)", Math.abs(vtime - 3.17) < 0.2, `video.currentTime = ${vtime.toFixed(2)}`);
const tel = await page.textContent('[data-testid="telemetry"]');
check("detection telemetry shows box size and TTC", /tracked/.test(tel) && /TTC|Time to contact/.test(tel) && /×/.test(tel), tel.replace(/\s+/g, " ").slice(0, 120));
await page.screenshot({ path: `${shots}/ui_t3.png` });

// 7. re-simulate through the UI
await page.selectOption('[data-testid="sim-panel"] select >> nth=0', "right");
await page.fill('[data-testid="sim-panel"] input >> nth=0', "2");
await page.click('[data-testid="simulate"]');
await page.waitForFunction(() => document.body.textContent.includes("unsaved re-simulation"), null, { timeout: 60000 });
const note = await page.textContent('[data-testid="plan-note"]');
check("re-simulate returns a new plan", /detour/i.test(note), note);
const dur = (await page.textContent('[data-testid="time"]')).split("/")[1];
check("new recording replaces the old one (different duration)", !dur.includes("14.72"), dur.trim());
await page.waitForTimeout(500);
await page.screenshot({ path: `${shots}/ui_resim.png` });

// 7b. wind: enter speed + 3D direction (azimuth AND elevation), re-simulate, see it everywhere
await page.fill('[data-testid="sim-panel"] input >> nth=0', "1.5");
await page.fill('[data-testid="wind-speed"]', "5");
await page.fill('[data-testid="wind-azimuth"]', "135");
await page.fill('[data-testid="wind-elevation"]', "30");
const readout = await page.textContent('[data-testid="wind-readout"]');
check("live wind vector readout matches 5 m/s @ az 135, el 30", /-3\.\d, 3\.\d, 2\.5 m\/s/.test(readout), readout.replace(/\s+/g, " ").slice(0, 110));
check("changing wind marks the result as stale", (await page.textContent('[data-testid="wind-stale"]')).includes("Re-simulate"));
await page.click('[data-testid="simulate"]');
await page.waitForFunction(() => !document.querySelector('[data-testid="wind-stale"]') && document.body.textContent.includes("unsaved re-simulation"), null, { timeout: 90000 });
await page.fill('[data-testid="scrubber"]', "2");
await page.waitForTimeout(500);
const hudWind = await page.textContent('[data-testid="hud-wind"]');
check("HUD shows the simulated wind and air-relative speed", /Wind5\.0m\/s/.test(hudWind.replace(/\s+/g, "")) && /Airspeed/.test(hudWind), hudWind.replace(/\s+/g, " ").slice(0, 90));
const planInfo = await page.textContent('[data-testid="wind-plan-info"]');
check("plan panel describes the wind and hover requirement", /Wind 5\.0 m\/s/.test(planInfo) && /elevation 30/.test(planInfo) && /Holding still/.test(planInfo), planInfo.replace(/\s+/g, " ").slice(0, 120));
check("flown with wind and still clear", /clear/.test(await page.textContent('[data-testid="plan-panel"]')) && !/COLLISION\s*[\d.]+°/.test((await page.textContent('[data-testid="plan-panel"]')).split("No avoidance")[0]));
await page.screenshot({ path: `${shots}/ui_wind.png` });

// wind input validation blocks the request
await page.fill('[data-testid="wind-elevation"]', "120");
check("out-of-range elevation is rejected in the form", (await page.textContent('[data-testid="wind-error"]')).includes("Elevation") && await page.isDisabled('[data-testid="simulate"]'));
await page.fill('[data-testid="wind-elevation"]', "30");
await page.click('[data-testid="wind-off"]');
await page.fill('[data-testid="sim-panel"] input >> nth=0', "");

// 8. unavoidable case is reported, not hidden
await page.fill('[data-testid="sim-panel"] input >> nth=0', "8");
await page.click('[data-testid="simulate"]');
await page.waitForSelector('[data-testid="unavoidable"]', { timeout: 60000 });
check("unavoidable scenario shows an explicit banner", true);
await page.screenshot({ path: `${shots}/ui_unavoidable.png` });

// 9. input error is surfaced
await page.fill('[data-testid="sim-panel"] input >> nth=0', "-3");
await page.click('[data-testid="simulate"]');
await page.waitForSelector('[data-testid="sim-error"]', { timeout: 20000 });
check("invalid input shows a server error message", true, (await page.textContent('[data-testid="sim-error"]')).slice(0, 90));

check("no console errors", consoleErrors.length === 0, consoleErrors.slice(0, 3).join(" | "));
await browser.close();
console.log(failures.length ? `\n${failures.length} FAILED: ${failures.join("; ")}` : "\nALL CHECKS PASSED");
process.exit(failures.length ? 1 : 0);

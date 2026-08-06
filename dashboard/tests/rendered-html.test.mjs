import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const pageUrl = new URL("../app/page.tsx", import.meta.url);

test("defines the local StormSignal operations surface", async () => {
  const page = await readFile(pageUrl, "utf8");
  assert.match(page, /Forecast surface/);
  assert.match(page, /AWS LIVE/);
  assert.match(page, /1 MIN/);
  assert.match(page, /5 MIN/);
  assert.match(page, /30 MIN/);
  assert.match(page, /probability.*0\.5/s);
});

test("keeps the API loopback-only and renders H3 polygons", async () => {
  const [page, packageJson] = await Promise.all([
    readFile(pageUrl, "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);
  assert.match(page, /http:\/\/127\.0\.0\.1:8765\/api\/state/);
  assert.match(page, /cellToBoundary/);
  assert.match(page, /h3-overlay/);
  assert.match(page, /createElementNS.*polygon/s);
  assert.match(packageJson, /"h3-js"/);
  assert.match(packageJson, /"maplibre-gl"/);
  assert.doesNotMatch(packageJson, /vinext|wrangler|cloudflare/i);
});

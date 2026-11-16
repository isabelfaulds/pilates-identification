const { chromium } = require('playwright');
const path = require('node:path');
const fs = require('node:fs/promises');
require('dotenv').config({ path: path.resolve(__dirname, '../../.env'), quiet: true });

const projectRoot = path.resolve(__dirname, '../..');

const { parse } = require('csv-parse/sync');
const catalogPath = path.join(projectRoot, 'output/lagree/micro-exercises.csv');

function filename(name) {
  const stem = name.trim().toLowerCase().replace(/\s+/g, '_')
    .replace(/[<>:"/\\|?*\x00-\x1f]/g, '_').replace(/[. ]+$/g, '');
  if (!stem || stem === '.' || stem === '..') throw new Error('Exercise name is empty or invalid');
  return `${stem}.mp4`;
}

async function main() {
  const rows = parse(await fs.readFile(catalogPath, 'utf8'), { columns: true, skip_empty_lines: true, bom: true });
  if (!rows.length || !Object.hasOwn(rows[0], 'training_data')) throw new Error('CSV must contain a training_data column');
  for (const row of rows) {
    if (!['0', '1'].includes(row.training_data.trim())) throw new Error(`training_data must be 0 or 1: ${row.name}`);
  }
  const videos = rows.filter(row => row.training_data.trim() === '1');
  console.log(`Selected ${videos.length} of ${rows.length} exercises from CSV.`);
  const force = process.argv.includes('--force');
  const listOnly = process.argv.includes('--list');
  const destinations = new Set();
  const pending = [];
  for (const video of videos) {
    const url = new URL(video.url);
    if (url.origin !== 'https://www.lagreeod.com' || !/^\/exercises\/[A-Za-z0-9]+$/.test(url.pathname)) {
      throw new Error('Expected a Lagree exercise URL');
    }
    const out = path.resolve(projectRoot, video.folder || 'training_data', filename(video.name));
    if (destinations.has(out)) throw new Error(`Duplicate destination: ${out}`);
    destinations.add(out);
    if (listOnly) { console.log(`${video.name} -> ${path.relative(projectRoot, out)}`); continue; }
    const existing = await fs.stat(out).catch(error => { if (error.code !== 'ENOENT') throw error; return null; });
    if (!force && existing?.isFile() && existing.size > 0) {
      console.log('Already downloaded:', path.relative(projectRoot, out));
      continue;
    }
    pending.push({...video, out});
  }
  if (!pending.length) return;
  const username = process.env.lagree_od_username;
  const password = process.env.lagree_od_password;
  if (!username || !password) throw new Error('Missing Lagree credentials in .env');
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  try {
    const context = await browser.newContext();
    const page = await context.newPage();
    const target = pending[0].url;
    await page.goto(target, { waitUntil: 'domcontentloaded' });
    console.log('Browser opened:', await page.title());
    // Submit the site's existing login form, preserving CSRF and device fields.
    await page.locator('#login-form input[name="email"]').evaluate((el, value) => { el.value = value; }, username);
    await page.locator('#login-form input[name="password"]').evaluate((el, value) => { el.value = value; }, password);
    const loginResult = new Promise((resolve, reject) => {
      page.route('**/login/validate_login', async route => {
        try {
          const response = await route.fetch();
          if (!response.ok()) throw new Error(`Login returned HTTP ${response.status()}`);
          const result = await response.json();
          await route.fulfill({ response });
          resolve(result);
        } catch (error) { reject(error); await route.abort().catch(() => {}); }
      });
    });
    await page.locator('#login-form').evaluate(form => form.requestSubmit());
    const result = await loginResult;
    if (!result.success) throw new Error('The site rejected login; check credentials or account verification.');
    console.log('Login succeeded.');
    await page.waitForTimeout(1500);
    let failures = 0;
    for (const video of pending) {
      try {
        await page.goto(video.url, { waitUntil: 'domcontentloaded' });
        await page.locator('.video-row .mvp-playlist-item[data-path], .video-row video').first().waitFor({ state: 'attached', timeout: 20000 });
        const sources = await page.locator('.video-row .mvp-playlist-item[data-path]').evaluateAll(els => els.flatMap(el => {
          try { return JSON.parse(el.dataset.path).map(source => source.mp4).filter(Boolean); }
          catch { return []; }
        }));
        const source = sources[0] || await page.locator('.video-row video, .video-row video source')
          .evaluateAll(els => els.map(el => el.currentSrc || el.src).find(src => /^https?:/.test(src)));
        if (!source) throw new Error('No downloadable video found in player');
        const download = await context.request.get(source, { headers: { Referer: video.url }, timeout: 120000 });
        try {
          if (!download.ok()) throw new Error(`Video returned HTTP ${download.status()}`);
          if (!(download.headers()['content-type'] || '').startsWith('video/')) throw new Error('Download response was not video');
          const bytes = await download.body();
          if (!bytes.length) throw new Error('Empty video response');
          await fs.mkdir(path.dirname(video.out), { recursive: true });
          // Rename only after the complete download is written.
          await fs.writeFile(`${video.out}.part`, bytes);
          await fs.rename(`${video.out}.part`, video.out);
          console.log('Downloaded:', path.relative(projectRoot, video.out), 'bytes:', bytes.length);
        } finally { await download.dispose(); }
      } catch (error) {
        failures++;
        console.error(`Failed (${video.name}):`, error.message.split('\n')[0]);
      }
    }
    if (failures) process.exitCode = 1;
  } finally { await browser.close(); }
}
main().catch(error => { console.error('Failed:', error.message.split('\n')[0]); process.exitCode = 1; });

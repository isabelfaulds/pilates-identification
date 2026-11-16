const { chromium } = require('playwright');
const path = require('node:path');
const fs = require('node:fs/promises');
require('dotenv').config({ path: path.resolve(__dirname, '../../.env'), quiet: true });

const target = 'https://www.lagreeod.com/exercises/micro';

async function main() {
  const username = process.env.lagree_od_username;
  const password = process.env.lagree_od_password;
  if (!username || !password) throw new Error('Missing Lagree credentials in .env');
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  try {
    const context = await browser.newContext();
    const page = await context.newPage();
    const media = new Set();
    page.on('response', response => {
      if (/\.(mp4|m3u8)(?:[?#]|$)/i.test(response.url())) media.add(response.url());
    });
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
    const firstResponse = page.waitForResponse(r => r.url().endsWith('/exercises/filter') && r.request().method() === 'POST');
    await page.goto(target, { waitUntil: 'domcontentloaded' });
    const initial = await firstResponse;
    const form = Object.fromEntries(new URLSearchParams(initial.request().postData()));
    let data = await initial.json();
    const records = new Map();
    let offset = 0;
    let pages = 0;
    while (true) {
      if (typeof data.view !== 'string') throw new Error('Unexpected catalog response');
      const batch = await page.evaluate(html => {
        const root = document.createElement('div');
        root.innerHTML = html;
        const text = (card, selector) => card.querySelector(selector)?.textContent.trim().replace(/\s+/g, ' ') || '';
        return [...root.querySelectorAll('.single-ajax-class')].map(card => {
          const href = card.querySelector('a.video-container')?.getAttribute('href');
          if (!href) throw new Error('Exercise link missing');
          const url = new URL(href, 'https://www.lagreeod.com/').href;
          return {
            id: url.split('/').pop(),
            name: text(card, '.video-ttl span'),
            duration: text(card, '.duration'),
            difficulty: text(card, '.thumb-difficulty'),
            machine: text(card, '.thumb-machine'),
            body_position: text(card, '.thumb-bposition').replace(/^Body position:\s*/, ''),
            url,
          };
        });
      }, data.view);
      const before = records.size;
      for (const row of batch) {
        if (!row.name) throw new Error('Exercise name missing');
        records.set(row.id, row);
      }
      pages++;
      console.log(`Page ${pages}: ${batch.length} exercises; ${records.size} unique total`);
      if (!data.show_more || data.show_more === '0' || data.show_more === 'false') break;
      if (!batch.length || records.size === before || pages >= 500) throw new Error('Pagination stalled; catalog incomplete');
      offset += batch.length;
      await page.waitForTimeout(250);
      const response = await context.request.post('https://www.lagreeod.com/exercises/filter', {
        form: {...form, start: String(offset)},
        headers: { Referer: target, 'X-Requested-With': 'XMLHttpRequest' },
      });
      if (!response.ok()) throw new Error(`Catalog page failed: HTTP ${response.status()}`);
      data = await response.json();
    }
    const rows = [...records.values()].sort((a,b) => a.name.localeCompare(b.name) || a.id.localeCompare(b.id));
    const groupRules = [
      [/bear/i, 'bear'],
      [/elevator/i, 'elevator lunge'],
      [/plank to pike/i, 'plank to pike'],
      [/spider kick/i, 'spider kick'],
      [/standing (?:inner|outer) thighs?/i, 'standing thigh'],
      [/side sweep/i, 'side sweep'],
      [/saw/i, 'saw'],
    ];
    for (const row of rows) row.group = groupRules.filter(([pattern]) => pattern.test(row.name)).map(([,group]) => group).join('; ');
    const dir = path.resolve(__dirname, '../../output/lagree');
    await fs.mkdir(dir, {recursive:true});
    // Keep manual selections from the CSV when refreshing the catalog.
    const { parse } = require('csv-parse/sync');
    let priorSelections = new Map();
    try {
      const previous = parse(await fs.readFile(path.join(dir, 'micro-exercises.csv'), 'utf8'), { columns: true, skip_empty_lines: true, bom: true });
      priorSelections = new Map(previous.map(row => [row.id, row.training_data || '0']));
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
    for (const row of rows) {
      const selected = priorSelections.get(row.id) || '0';
      if (!['0', '1'].includes(selected)) throw new Error(`Invalid training_data selection for ${row.id}`);
      row.training_data = Number(selected);
    }

    await fs.writeFile(path.join(dir, 'micro-exercises.json'), JSON.stringify({source:target,retrieved_at:new Date().toISOString(),count:rows.length,pages,complete:true,exercises:rows},null,2)+'\n');
    const columns = ['name','group','training_data','duration','difficulty','machine','body_position','url','id'];
    const csv = value => '"'+String(value).replaceAll('"','""')+'"';
    await fs.writeFile(path.join(dir, 'micro-exercises.csv'), [columns.map(csv).join(','), ...rows.map(row => columns.map(col => csv(row[col])).join(','))].join('\n')+'\n');
    const lines = rows.map((r,i) => `${i+1}. [${r.name}](${r.url}) — ${r.duration}; ${r.difficulty}; ${r.body_position}`);
    await fs.writeFile(path.join(dir, 'micro-exercises.md'), `# Micro exercise catalog\n\n${rows.length} exercises from ${target}. All ${pages} pages collected through the final show_more=false response.\n\n`+lines.join('\n')+'\n');
    console.log('Complete catalog saved:', dir);
  } finally { await browser.close(); }
}
main().catch(error => { console.error('Failed:', error.message.split('\n')[0]); process.exitCode = 1; });

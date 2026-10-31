async (page) => {
  const checks = [], errors = [], imagePosts = [];
  const check = (value, name) => { if (!value) throw Error(name); checks.push({name, passed: true}); };
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => { if (request.method() === 'POST' && request.url().endsWith('/api/image/detect')) imagePosts.push(request.url()); });
  let releaseOld;
  try {
    await page.goto('http://127.0.0.1:8005/?v=performance-regression#image');
    await page.reload();
    await page.setViewportSize({width: 1440, height: 1000});
    await page.evaluate(() => document.querySelector('#image-input').addEventListener('change', () => {
      window.qaDecodePending = document.querySelector('#detect-image').disabled && !state.imageReady;
    }, {once: true}));
    await page.locator('#image-input').setInputFiles('E:/AIGC_latest_20261011/output/playwright/resolution-fixtures/3840x2160-RGB.png');
    await page.locator('#detect-image').waitFor({state: 'visible'});
    await page.waitForFunction(() => state.imageReady);
    check(await page.evaluate(() => window.qaDecodePending), 'Detection stays disabled until image decoding completes');
    check((await page.locator('#preview-caption').textContent()).includes('3840 × 2160'), 'A valid 4K preview preserves dimensions');
    check(await page.locator('#detect-image').isEnabled(), 'A decoded 4K image can be analyzed');

    await page.locator('#image-input').setInputFiles({name: 'damaged.png', mimeType: 'image/png', buffer: Buffer.from('not a PNG image')});
    await page.locator('#image-result .error-state').waitFor();
    check((await page.locator('#image-result').textContent()).includes('图片无法读取'), 'A damaged image has an actionable error');
    check(await page.locator('#detect-image').isDisabled(), 'A damaged image cannot be submitted');
    check(await page.locator('#image-preview').isHidden() && await page.locator('#preview-caption').isHidden(), 'Old dimensions and preview are cleared after a decode error');
    check(await page.evaluate(() => state.image === null && !state.imageReady && !document.querySelector('#image-preview').dataset.objectUrl), 'Rejected image state and object URL are released');
    await page.screenshot({path: 'output/playwright/performance-corrupt-image.png', fullPage: true});

    await page.locator('#image-input').setInputFiles('E:/AIGC_latest_20261011/output/playwright/performance-fixtures/over-pixel-limit.png');
    await page.locator('#image-result .error-state').filter({hasText: '2500 万像素'}).waitFor();
    check(await page.locator('#detect-image').isDisabled(), 'The browser enforces the same pixel ceiling as the API');

    await page.locator('#image-input').setInputFiles('E:/AIGC_latest_20261011/output/playwright/resolution-fixtures/1x1-RGB.png');
    await page.waitForFunction(() => state.imageReady);
    check(await page.locator('#detect-image').isEnabled(), 'A valid 1px image still works after a rejected image');
    check((await page.locator('#preview-caption').textContent()).includes('1 × 1'), 'Recovered preview has no previous 4K dimensions');
    check(imagePosts.length === 0, 'Invalid inputs are rejected before any detection upload');

    const template = await (await page.request.get('http://127.0.0.1:8005/api/models')).json();
    const gate = new Promise(resolve => { releaseOld = resolve; });
    let routeCount = 0;
    await page.route('**/api/models', async route => {
      const index = ++routeCount;
      if (index === 1) await gate;
      await route.fulfill({json: {...template, scope: {...template.scope, image: index === 1 ? 'OLDER MODEL RESPONSE' : 'CURRENT MODEL RESPONSE'}}});
    });
    await page.locator('#model-info-button').click();
    await page.waitForFunction(() => document.querySelector('#model-dialog').open);
    await page.locator('#close-dialog').click();
    await page.locator('#model-info-button').click();
    await page.locator('#model-details').filter({hasText: 'CURRENT MODEL RESPONSE'}).waitFor();
    releaseOld();
    await page.waitForResponse(response => response.url().endsWith('/api/models') && response.status() === 200).catch(() => {});
    check((await page.locator('#model-details').textContent()).includes('CURRENT MODEL RESPONSE'), 'A delayed model response cannot replace a more recent dialog result');
    check(!(await page.locator('#model-details').textContent()).includes('OLDER MODEL RESPONSE'), 'Old model details are discarded');
    await page.locator('#dialog-done').click();
    await page.unroute('**/api/models');
    await page.setViewportSize({width: 390, height: 844});
    await page.locator('#model-info-button').click();
    await page.locator('#model-details .evaluation-summary').first().waitFor();
    check(await page.locator('#model-details .evaluation-summary').count() === 4, 'All four measured sample cohorts remain available');
    check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Updated model information fits the mobile viewport');
    await page.screenshot({path: 'output/playwright/performance-model-mobile.png', fullPage: true});
    await page.locator('#dialog-done').click();
    check(errors.length === 0, 'No uncaught browser errors');
    return JSON.stringify({status: 'passed', checks, errors, scope: 'Preview and metadata races; detection is tested separately with actual models.'});
  } catch (error) {
    return JSON.stringify({status: 'failed', checks, errors, error: String(error)});
  } finally {
    if (releaseOld) releaseOld();
    await page.unroute('**/api/models');
  }
}

async (page) => {
  const checks = [], errors = [], agentCalls = [];
  const check = (condition, name) => { if (!condition) throw Error(name); checks.push({name, passed: true}); };
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => { if (request.url().includes('/api/agent/')) agentCalls.push(request.url()); });
  await page.setViewportSize({width: 1440, height: 1000});
  await page.goto('http://127.0.0.1:8004/?v=upgrade20261026#image');
  await page.locator('#service-label').filter({hasText: '服务就绪'}).waitFor();
  check(await page.locator('.nav-item').count() === 4, 'Navigation contains exactly four direct workflows');
  check(await page.locator('#view-agent').count() === 0, 'Agent page and controls are removed');
  check(await page.locator('link[href*="agent.css"]').count() === 0, 'Agent stylesheet is removed');
  check((await page.locator('.input-note').first().textContent()).includes('任意正尺寸'), 'Image page describes arbitrary positive resolutions');
  const sample = await page.request.get('http://127.0.0.1:8004/static/assets/samples_2026.json');
  const samples = await sample.json();
  check(samples.length === 2 && samples.every(row => row.source_revision && row.split === 'validation'), '2026 examples retain source and validation-set provenance');
  await page.locator('.sample-button[data-sample="real"]').click();
  await page.locator('#image-preview').waitFor({state: 'visible'});
  check((await page.locator('#file-name').textContent()).includes('ntire-2026'), '2026 example selection loads the current image');
  check(await page.locator('#detect-image').isEnabled(), 'Example can be submitted for image analysis');
  await page.screenshot({path: 'output/playwright/no-agent-desktop.png', fullPage: true});
  for (const name of ['text', 'document', 'edit']) {
    await page.locator(`.nav-item[data-view="${name}"]`).click();
    check(await page.locator(`#view-${name}`).isVisible(), `${name} page remains accessible`);
  }
  await page.locator('#edit-input').fill('In order to improve 20 samples, the the team found no clear difference.');
  await page.locator('#run-edit').click();
  await page.locator('#edit-output').waitFor();
  check((await page.locator('#edit-output').textContent()).includes('20'), 'Rule editor remains usable after Agent removal');
  check(await page.locator('#run-llm-edit').isEnabled(), 'Local editing-preview control remains available');
  for (const path of ['/api/agent/run', '/api/agent/upload']) {
    const response = await page.request.post("http://127.0.0.1:8004" + path, {data: {text: 'example'}});
    check(response.status() === 404, `${path} is no longer exposed`);
  }
  const schema = await (await page.request.get('http://127.0.0.1:8004/openapi.json')).json();
  check(!Object.keys(schema.paths).some(path => path.includes('/agent/')), 'API documentation no longer advertises Agent endpoints');
  await page.goto('http://127.0.0.1:8004/?v=upgrade20261026#agent');
  await page.locator('#view-document').waitFor({state: 'visible'});
  check(page.url().endsWith('#document'), 'Old Agent bookmarks redirect to the document workflow');
  await page.setViewportSize({width: 390, height: 844});
  for (const name of ['image', 'text', 'document', 'edit']) {
    await page.locator(`.nav-item[data-view="${name}"]`).click();
    check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `${name} page fits a 390px screen`);
  }
  await page.screenshot({path: 'output/playwright/no-agent-mobile.png', fullPage: true});
  check(agentCalls.length === 0, 'The four workflows make no Agent API requests');
  check(errors.length === 0, 'No uncaught browser errors');
  return JSON.stringify({status: 'passed', checks, errors});
}

async (page) => {
  const observed = {};
  const original = 'The team reviewed 20 samples in order to check whether the results were consistent across the two groups.';
  await page.getByRole('button', {name:'文本检测', exact:true}).click();
  await page.locator('#text-input').fill(original);
  await page.getByRole('button', {name:'分析文本', exact:true}).click();
  await page.locator('#text-result .result-header').waitFor();
  await page.locator('#text-input').fill(original + ' A new sentence was added.');
  observed.completed_text_result_stays_after_input_change = await page.locator('#text-result .result-header').count() === 1;
  await page.getByRole('button', {name:'可信编辑', exact:true}).click();
  await page.locator('#edit-input').fill(original);
  await page.locator('#run-edit').click();
  await page.locator('#edit-output').waitFor();
  await page.locator('#edit-input').fill(original + ' This is a different version.');
  observed.completed_edit_stays_after_input_change = await page.locator('#use-edit').isVisible();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.evaluate(() => Object.defineProperty(navigator.clipboard, 'writeText', {configurable:true, value:() => Promise.reject(new DOMException('Clipboard denied in regression fixture','NotAllowedError'))}));
  await page.locator('#copy-edit').click();
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  observed.clipboard_denial_is_unhandled = errors.some(error => error.includes('Clipboard denied'));
  await page.getByRole('button', {name:'Agent 助手', exact:true}).click();
  await page.locator('#agent-document-input').setInputFiles({name:'previous.txt', mimeType:'text/plain', buffer:Buffer.from(original)});
  await page.locator('#agent-document-input').setInputFiles({name:'rejected.exe', mimeType:'application/octet-stream', buffer:Buffer.from('invalid')});
  observed.invalid_agent_document_keeps_previous_upload = await page.locator('#agent-document-name').isVisible() && (await page.locator('#agent-document-name').innerText()).includes('previous.txt');
  await page.locator('#agent-clear-document').click();
  await page.locator('#agent-text-file').setInputFiles({name:'unicode.txt', mimeType:'text/plain', buffer:Buffer.from('😀'.repeat(6000))});
  await page.getByText('文本超过 10,000 字符，请缩短后重试。', {exact:true}).waitFor();
  observed.valid_6000_codepoints_are_rejected_on_import = (await page.locator('#agent-input').inputValue()).length === 0;
  await page.getByRole('button', {name:'图像检测', exact:true}).click();
  const image = await (await page.request.get('http://127.0.0.1:8002/static/assets/sample-fake.png')).body();
  let arrive, release;
  const ready = new Promise(resolve => arrive = resolve), gate = new Promise(resolve => release = resolve);
  await page.route('**/static/assets/sample-real.png', async route => {arrive(); await gate; await route.fulfill({body:image, contentType:'image/png'});});
  try {
    await page.getByRole('button', {name:'真实类样本缩略图 真实类样本', exact:true}).click();
    await ready;
    await page.locator('#image-input').setInputFiles({name:'manual.png', mimeType:'image/png', buffer:image});
  } finally {release();}
  await page.getByText('cifake-real-sample.png', {exact:true}).waitFor();
  observed.delayed_sample_replaces_later_manual_selection = true;
  await page.unroute('**/static/assets/sample-real.png');
  await page.route('**/static/assets/sample-fake.png', route => route.fulfill({status:404, body:'missing sample'}));
  await page.getByRole('button', {name:'生成类样本缩略图 生成类样本', exact:true}).click();
  await page.getByText('cifake-fake-sample.png', {exact:true}).waitFor();
  observed.missing_sample_is_marked_ready = await page.getByText('图片已就绪', {exact:true}).isVisible();
  await page.unroute('**/static/assets/sample-fake.png');
  return {before_fix:true, observed};
}

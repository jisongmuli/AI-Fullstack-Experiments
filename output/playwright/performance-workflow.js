async (page) => {
  const base = 'http://127.0.0.1:8005', checks = [], errors = [], downloads = [];
  const check = (condition, name) => { if (!condition) throw Error(name); checks.push({name, passed: true}); };
  page.on('pageerror', error => errors.push(error.message));
  let jobId, reportId;
  try {
    await page.setViewportSize({width: 1440, height: 1000});
    await page.goto(base + '/?v=performance-workflow#image');
    await page.reload();
    check(await page.locator('.nav-item').count() === 4, 'Current service presents four workflows');
    await page.locator('#image-input').setInputFiles('E:/AIGC_latest_20261011/output/playwright/resolution-fixtures/3840x2160-RGB.png');
    await page.waitForFunction(() => document.querySelector('#image-preview').naturalWidth === 3840);
    check(await page.locator('#detect-image').isEnabled(), 'A full 4K image is accepted by the browser');
    const [imageResponse] = await Promise.all([
      page.waitForResponse(r => r.url().endsWith('/api/image/detect') && r.request().method() === 'POST'),
      page.locator('#detect-image').click()
    ]);
    const image = await imageResponse.json();
    check(imageResponse.status() === 200 && image.input_size.join('x') === '3840x2160', 'The server receives the original 4K dimensions');
    check(image.model_version.includes('ICML2026'), 'The image response uses DEAR 2026');
    await page.locator('#image-result .result-foot').waitFor();
    check(await page.locator('#image-result .model-score-list strong').count() === 1, 'A single selected image detector renders correctly');
    check(image.local_evidence.values.length === 9, 'Local sensitivity data remains available');
    await page.screenshot({path: 'output/playwright/performance-workflow-image-4k.png', fullPage: true});
    await page.locator('#image-input').setInputFiles('E:/AIGC_latest_20261011/output/playwright/resolution-fixtures/1x1-RGB.png');
    const [tinyResponse] = await Promise.all([
      page.waitForResponse(r => r.url().endsWith('/api/image/detect') && r.request().method() === 'POST'),
      page.locator('#detect-image').click()
    ]);
    const tiny = await tinyResponse.json();
    check(tinyResponse.status() === 200 && Number.isFinite(tiny.risk_score), 'A 1px image receives a valid analysis response');
    check(tiny.decision === 'uncertain', 'Minimal pixel information retains a review status');
    await page.locator('.nav-item[data-view="text"]').click();
    const chinese = '研究团队核对每个样本的来源、标签和处理过程，并将验证步骤保存为可复查的记录。' .repeat(4);
    await page.locator('#text-input').fill(chinese + '\n\nThe team reviewed each source record and compared the results before publication. 😀');
    const [textResponse] = await Promise.all([
      page.waitForResponse(r => r.url().endsWith('/api/text/detect') && r.request().method() === 'POST'),
      page.locator('#detect-text').click()
    ]);
    const text = await textResponse.json();
    await page.locator('#text-result .text-segments').waitFor();
    check(textResponse.status() === 200 && text.segments.length >= 2, 'Current language models return separate paragraph results');
    check(await page.locator('#text-result .text-segment').count() === text.segments.length, 'Paragraph styles survive Agent removal');
    await page.locator('.nav-item[data-view="document"]').click();
    await page.locator('#document-mode-text').click();
    const original = chinese + '\n\nThe team compared the input files and recorded every result before the final review. 😀';
    await page.locator('#document-text').fill(original);
    const [accepted] = await Promise.all([
      page.waitForResponse(r => r.url().endsWith('/api/document/jobs') && r.request().method() === 'POST'),
      page.locator('#document-detect').click()
    ]);
    jobId = (await accepted.json()).job_id;
    await page.locator('#document-result .document-exports').waitFor({timeout: 90000});
    const job = await (await page.request.get(base + '/api/document/jobs/' + jobId)).json();
    reportId = job.report_id;
    check(job.status === 'completed', 'A real Transformer document task completes through the interface');
    const report = await (await page.request.get(base + '/api/document/reports/' + reportId + '/json')).json();
    check(report.extracted_text === original, 'The report preserves all original Unicode text');
    check(await page.locator('#document-result button').filter({hasText: /Agent/}).count() === 0, 'Document exports have no Agent handoff action');
    const labels = ['带下划线的 Word', '标注 HTML / 打印 PDF', '完整结果 JSON'];
    for (const [index, label] of labels.entries()) {
      const [download] = await Promise.all([page.waitForEvent('download'), page.locator('#document-result').getByRole('button', {name: label, exact: true}).click()]);
      const extension = ['docx', 'html', 'json'][index];
      await download.saveAs('E:/AIGC_latest_20261011/output/playwright/performance-workflow-downloads/report.' + extension);
      downloads.push(download.suggestedFilename());
      check(download.suggestedFilename().endsWith('.' + extension), extension + ' export works with the current models');
    }
    await page.setViewportSize({width: 390, height: 844});
    check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'The modern document report fits a 390px viewport');
    await page.screenshot({path: 'output/playwright/performance-workflow-document-mobile.png', fullPage: true});
    await page.locator('#model-info-button').click();
    await page.locator('#model-details .evaluation-summary').first().waitFor();
    check(await page.locator('#model-details .evaluation-summary').count() === 4, 'Model information displays all four measured cohorts');
    check((await page.locator('#model-details').textContent()).includes('98.6%'), 'The model dialog displays the actual Chinese heldout measurement');
    check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Model measurements fit the mobile viewport');
    await page.screenshot({path: 'output/playwright/performance-workflow-model-info-mobile.png', fullPage: true});
    await page.locator('#dialog-done').click();
    check(errors.length === 0, 'No uncaught browser errors in the modern workflows');
    return JSON.stringify({status: 'passed', actual_models: true, checks, errors, downloads});
  } catch (error) {
    return JSON.stringify({status: 'failed', checks, errors, error: String(error), page: await page.locator('body').innerText()});
  } finally {
    if (jobId) await page.request.delete(base + '/api/document/jobs/' + jobId);
    if (reportId) await page.request.delete(base + '/api/document/reports/' + reportId);
  }
}

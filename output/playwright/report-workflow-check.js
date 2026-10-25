async (page) => {
  const checks = [], errors = [], downloadNames = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('download', download => downloadNames.push(download.suggestedFilename()));
  const check = (name, passed) => { if (!passed) throw new Error(name); checks.push({name, passed}); };
  const text = '小组在整理公开资料时，逐一检查文件格式、来源和内容完整性。每份记录都有稳定编号，重复项和缺失值分别统计。完成处理后，小组保留了原始记录和操作参数，方便复查本次实验。\n\nThe team checked the source files before preparing the data. Each record received a stable identifier, and missing values were counted separately.';
  const filename = '报告流程验证.txt', agentFilename = 'Agent报告验证.txt';
  await page.setViewportSize({width:1280, height:900});
  await page.getByRole('button', {name:'文档检测', exact:true}).click();
  await page.locator('#document-file').setInputFiles({name:filename, mimeType:'text/plain', buffer:Buffer.from(text)});
  const created = page.waitForResponse(response => response.url().endsWith('/api/document/jobs') && response.request().method() === 'POST');
  await page.getByRole('button', {name:'检测文档并标注', exact:true}).click();
  const acceptedJob = await (await created).json();
  await page.locator('#document-annotated-text').waitFor();
  const completedJob = await (await page.request.get(`http://127.0.0.1:8002/api/document/jobs/${acceptedJob.job_id}`)).json();
  const report = await (await page.request.get(`http://127.0.0.1:8002/api/document/reports/${completedJob.report_id}`)).json();
  check('Real document detection preserves the complete original', await page.locator('#document-annotated-text').innerText() === text);
  await page.getByRole('button', {name:`打开报告：${filename}`, exact:true}).waitFor();
  check('New reports appear in the current service list', true);
  const stored = await page.evaluate(() => localStorage.getItem('aigc.document.history.v1') || '');
  check('Unchecked browser history stores neither this report nor its text', !stored.includes(report.report_id) && !stored.includes(text));

  await page.reload();
  const openReport = page.getByRole('button', {name:`打开报告：${filename}`, exact:true});
  await openReport.click();
  await page.locator('#document-annotated-text').waitFor();
  check('A report can be restored after a page reload', await page.locator('#document-annotated-text').innerText() === text);
  await page.getByRole('button', {name:'Agent 助手', exact:true}).click();
  await page.locator('#agent-document-input').setInputFiles({name:agentFilename, mimeType:'text/plain', buffer:Buffer.from(text)});
  const agentResponse = page.waitForResponse(response => response.url().endsWith('/api/agent/upload'));
  await page.locator('#agent-form button').click();
  const agent = await (await agentResponse).json();
  check('Agent document analysis succeeds', Boolean(agent.findings?.document?.report_id));
  const agentId = agent.findings.document.report_id;
  await page.getByRole('button', {name:'文档检测', exact:true}).click();
  await page.getByRole('button', {name:`打开报告：${agentFilename}`, exact:true}).waitFor();
  check('Returning from Agent refreshes its report into the list', true);
  await openReport.click();
  await page.getByRole('button', {name:'粘贴长文', exact:true}).click();
  await page.locator('#document-text').fill(text);
  let arrived, release, deleteCalls = 0;
  const ready = new Promise(resolve => arrived = resolve), gate = new Promise(resolve => release = resolve);
  await page.route(`**/api/document/reports/${agentId}`, async route => {
    if (route.request().method() !== 'DELETE') { await route.continue(); return; }
    deleteCalls++;
    const response = await route.fetch(); arrived(); await gate;
    await route.fulfill({response});
  });
  try {
    const removeAgent = page.getByRole('button', {name:`删除报告：${agentFilename}`, exact:true});
    await removeAgent.click(); await ready;
    check('Deletion blocks conflicting analysis and restoration actions', await page.locator('#document-detect').isDisabled() && await openReport.isDisabled());
    check('Deletion does not offer an ambiguous cancel action', await page.locator('#document-cancel').isHidden());
    await removeAgent.dispatchEvent('click'); await openReport.dispatchEvent('click');
    check('Repeated deletion is suppressed', deleteCalls === 1);
  } finally {release();}
  await page.getByRole('button', {name:`删除报告：${agentFilename}`, exact:true}).waitFor({state:'detached'});
  await page.waitForFunction(() => !document.getElementById('document-mode-text').disabled);
  check('Deleting a different report preserves the open report', await page.locator('#document-result h3').innerText() === filename);
  await page.unroute(`**/api/document/reports/${agentId}`);

  const beforeError = downloadNames.length;
  await page.route(`**/api/document/reports/${report.report_id}/html`, route => route.fulfill({status:404, json:{detail:'Expired fixture'}}));
  await page.getByRole('button', {name:'标注 HTML / 打印 PDF', exact:true}).click();
  await page.getByText('报告已过期或已删除，请重新检测后下载。', {exact:true}).waitFor();
  check('Expired export shows a readable message without downloading an error file', downloadNames.length === beforeError);
  await page.unroute(`**/api/document/reports/${report.report_id}/html`);
  for (const [label, extension] of [['带下划线的 Word','docx'], ['标注 HTML / 打印 PDF','html'], ['完整结果 JSON','json']]) {
    const pending = page.waitForEvent('download');
    await page.getByRole('button', {name:label, exact:true}).click();
    const download = await pending;
    check(`Actual ${extension} export preserves the Chinese filename`, download.suggestedFilename() === `文档检测报告_报告流程验证.${extension}`);
    await download.saveAs(`output/playwright/report-downloads/verified.${extension}`);
    await page.waitForFunction(() => !document.getElementById('document-mode-text').disabled);
  }
  await page.route('**/api/document/reports', route => route.abort('connectionrefused'), {times:1});
  await page.getByRole('button', {name:'刷新列表', exact:true}).click();
  await page.getByText('报告列表暂未连接到服务，可点击刷新重试。', {exact:true}).waitFor();
  check('A failed list refresh preserves existing report entries', await openReport.isVisible());
  await page.getByRole('button', {name:'刷新列表', exact:true}).click();
  await page.waitForFunction(() => document.getElementById('document-reports-status').textContent === '');
  check('Report list recovers after a connection error', await openReport.isVisible());
  await page.screenshot({path:'output/playwright/reports-desktop.png', fullPage:true});
  await page.setViewportSize({width:390, height:844});
  check('The report workflow fits a 390px screen', await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.screenshot({path:'output/playwright/reports-mobile.png', fullPage:true});
  await page.setViewportSize({width:1280, height:900});
  await page.getByRole('button', {name:'删除本次报告', exact:true}).click();
  await page.getByText('报告已删除', {exact:true}).waitFor();
  await openReport.waitFor({state:'detached'});
  check('Deleting the open report removes its result and list entry', await page.locator('#document-annotated-text').count() === 0);
  check('No uncaught browser errors', errors.length === 0);
  return {status:'passed', checks, errors, downloadNames};
}

async (page) => {
  const checks=[], errors=[], downloads=[];
  page.on('pageerror',error=>errors.push(error.message));
  page.on('download',download=>downloads.push(download.suggestedFilename()));
  const check=(name,passed)=>{if(!passed)throw Error(name);checks.push({name,passed});};
  const base='http://127.0.0.1:8002';
  let firstArrive,firstRelease,secondArrive,secondRelease,polls=0,cancelled=false;
  const firstReady=new Promise(resolve=>firstArrive=resolve),firstGate=new Promise(resolve=>firstRelease=resolve);
  const secondReady=new Promise(resolve=>secondArrive=resolve),secondGate=new Promise(resolve=>secondRelease=resolve);
  const job={job_id:'race-job',filename:'粘贴的长文',status:'running',stage:'analyzing',completed:5,total:10,progress:52,revision:10,report_id:null,error:null};
  await page.route('**/api/document/jobs',route=>route.fulfill({status:route.request().method()==='POST'?202:200,json:route.request().method()==='POST'?job:{jobs:[{...job,revision:8,completed:1,progress:18}]}}));
  await page.route('**/api/document/jobs/race-job',async route=>{
    if(++polls===1){firstArrive();await firstGate;await route.fulfill({json:{...job,revision:8,completed:1,progress:18}});}
    else if(polls===2){secondArrive();await secondGate;await route.fulfill({json:job});}
    else await route.fulfill({json:{...job,status:cancelled?'cancelled':'cancel_requested',stage:cancelled?'cancelled':'analyzing',revision:cancelled?12:11}});
  });
  await page.route('**/api/document/jobs/race-job/cancel',route=>route.fulfill({json:{...job,status:'cancel_requested',revision:11}}));
  let reportId=null,jobId=null;
  try {
    await page.goto(base+'/?v=task-hardening#document');
    await page.reload();
    await page.getByRole('button',{name:'粘贴长文',exact:true}).click();
    await page.locator('#document-text').fill('The team reviewed the source files before preparing the data.');
    await page.locator('#document-detect').click();await firstReady;
    check('An active task has no history-removal action',await page.getByRole('button',{name:'移除任务记录：粘贴的长文',exact:true}).count()===0);
    const firstResponse=page.waitForResponse(response=>response.url().endsWith('/api/document/jobs/race-job'));
    firstRelease();await firstResponse;await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    check('An older progress response cannot move the progress bar backwards',await page.locator('#document-task-progress').evaluate(element=>element.value)===52);
    await secondReady;
    await page.getByRole('button',{name:'取消任务',exact:true}).click();
    await page.waitForFunction(()=>document.getElementById('document-background').disabled);
    const secondResponse=page.waitForResponse(response=>response.url().endsWith('/api/document/jobs/race-job'));
    secondRelease();await secondResponse;await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    check('A stale progress response cannot re-enable cancellation',await page.locator('#document-cancel').isDisabled());
    check('A stale progress response cannot re-enable background processing',await page.locator('#document-background').isDisabled());
    await page.locator('#document-refresh-jobs').click();
    await page.waitForFunction(()=>!document.getElementById('document-refresh-jobs').disabled);
    check('A stale collection refresh preserves the cancellation intent',await page.locator('#document-jobs').getByText('正在取消，请稍候…',{exact:true}).isVisible());
    cancelled=true;
    await page.getByText('任务已取消，可重新选择文档。',{exact:true}).waitFor();
    check('A terminal response exits cancellation mode',await page.locator('#document-mode-file').isEnabled());
    await page.unrouteAll({behavior:'wait'});
    await page.goto(base+'/?v=history-hardening#document');

    const text='The team reviewed the source files and counted missing values before preparing the data. Every record received a stable identifier.';
    const filename='记录移除验证.txt';
    await page.locator('#document-file').setInputFiles({name:filename,mimeType:'text/plain',buffer:Buffer.from(text)});
    const accepted=page.waitForResponse(response=>response.url().endsWith('/api/document/jobs')&&response.request().method()==='POST');
    await page.locator('#document-detect').click();jobId=(await(await accepted).json()).job_id;
    await page.locator('#document-annotated-text').waitFor();
    reportId=(await(await page.request.get(base+'/api/document/jobs/'+jobId)).json()).report_id;
    await page.getByRole('button',{name:'移除任务记录：'+filename,exact:true}).click();
    await page.getByRole('button',{name:'移除任务记录：'+filename,exact:true}).waitFor({state:'detached'});
    check('History removal deletes only task metadata',(await page.request.get(base+'/api/document/jobs/'+jobId)).status()===404);
    check('The generated report survives task-history removal',(await page.request.get(base+'/api/document/reports/'+reportId)).status()===200);
    check('History removal preserves the report currently being read',await page.locator('#document-annotated-text').innerText()===text);
    await page.getByRole('button',{name:'交给 Agent 汇总',exact:true}).click();
    const word=page.getByRole('button',{name:'Word 报告',exact:true}).first();
    await page.route(`**/api/document/reports/${reportId}/docx`,route=>route.fulfill({status:503,json:{detail:'导出暂时不可用，请重试。'}}),{times:1});
    await word.click();
    await page.locator('.agent-download-status').first().getByText('导出暂时不可用，请重试。',{exact:true}).waitFor();
    check('Agent export failures show inline feedback without an error download',downloads.length===0&&await word.isEnabled());
    const docxReady=page.waitForEvent('download');await word.click();const docx=await docxReady;
    await docx.saveAs('output/playwright/hardening-downloads/agent-report.docx');
    check('A failed Agent download can be retried with the Chinese filename',docx.suggestedFilename()==='文档检测报告_记录移除验证.docx');
    await page.getByRole('button',{name:'文档检测',exact:true}).click();
    await page.getByRole('button',{name:'交给 Agent 汇总',exact:true}).click();
    const html=page.getByRole('button',{name:'HTML / 打印 PDF',exact:true});
    let arrive,release,calls=0;
    const ready=new Promise(resolve=>arrive=resolve),gate=new Promise(resolve=>release=resolve);
    await page.route(`**/api/document/reports/${reportId}/html`,async route=>{calls++;const response=await route.fetch();arrive();await gate;await route.fulfill({response});});
    const before=downloads.length,htmlReady=page.waitForEvent('download');
    try{await html.nth(0).click();await ready;await html.nth(1).click();check('Same-report downloads share one HTTP transfer across Agent cards',calls===1);}finally{release();}
    const htmlDownload=await htmlReady;await htmlDownload.saveAs('output/playwright/hardening-downloads/agent-report.html');
    await page.waitForFunction(()=>Array.from(document.querySelectorAll('.agent-report-download')).every(button=>!button.disabled));
    check('A shared download starts only one native file download',downloads.length===before+1);
    await page.unroute(`**/api/document/reports/${reportId}/html`);
    await page.request.delete(base+'/api/document/reports/'+reportId);
    const count=downloads.length;await word.click();
    await page.locator('.agent-download-status').first().getByText('报告已过期或已删除，请重新检测后下载。',{exact:true}).waitFor();
    check('An expired Agent report produces a readable error and no JSON download',downloads.length===count&&await word.isEnabled());
    await page.screenshot({path:'output/playwright/agent-download-error.png',fullPage:true});
    await page.setViewportSize({width:390,height:844});
    check('Agent download feedback fits a 390px screen',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    check('Agent report actions do not cause horizontal chat scrolling',await page.locator('#chat-messages').evaluate(element=>element.scrollWidth<=element.clientWidth));
    await page.screenshot({path:'output/playwright/agent-download-error-mobile.png',fullPage:true});
    check('No uncaught browser errors',errors.length===0);
    return {status:'passed',checks,errors,downloads};
  } finally {
    firstRelease();secondRelease();await page.unrouteAll({behavior:'wait'});
    if(reportId)await page.request.delete(base+'/api/document/reports/'+reportId);
    if(jobId)await page.request.delete(base+'/api/document/jobs/'+jobId);
  }
}

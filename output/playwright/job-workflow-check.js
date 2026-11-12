async (page) => {
  const base='http://127.0.0.1:8004', checks=[], errors=[], ids=[];
  page.on('pageerror',error=>errors.push(error.message));
  const check=(name,passed)=>{if(!passed)throw Error(name);checks.push({name,passed});};
  const paragraph='The team checked the source files before preparing the data. Each record received a stable identifier, and missing values were counted separately.';
  const text=(paragraph+'\n\n').repeat(100);
  async function start(name,body=text){
    await page.locator('#document-file').setInputFiles({name,mimeType:'text/plain',buffer:Buffer.from(body)});
    const response=page.waitForResponse(item=>item.url().endsWith('/api/document/jobs')&&item.request().method()==='POST');
    await page.locator('#document-detect').click();
    const accepted=await response, job=await accepted.json(); ids.push(job.job_id);
    check(`Task ${name} is accepted immediately`,accepted.status()===202);
    return job;
  }
  try {
    await page.goto(base+'/?v=jobs20261025#document');
    await page.setViewportSize({width:1280,height:900});
    const cancelled=await start('取消进度验证.txt');
    await page.waitForFunction(()=>document.getElementById('document-task-progress')?.value>10);
    check('Actual worker progress is visible',await page.locator('#document-task-progress').getAttribute('aria-valuetext')!==null);
    await page.getByRole('button',{name:'取消任务',exact:true}).click();
    check('Cancellation prevents a conflicting background action',await page.locator('#document-background').isHidden()||await page.locator('#document-background').isDisabled());
    await page.getByText('任务已取消，可重新选择文档。',{exact:true}).waitFor();
    const cancelState=await(await page.request.get(base+'/api/document/jobs/'+cancelled.job_id)).json();
    check('Cancel stops the server task and leaves no report',cancelState.status==='cancelled'&&!cancelState.report_id);
    check('Cancellation restores the processing slot',(await(await page.request.get(base+'/api/runtime')).json()).processing.active===0);

    const background=await start('后台恢复验证.txt');
    await page.locator('#document-background').click();
    await page.getByText('任务已转入后台',{exact:true}).waitFor();
    check('Background processing leaves document controls available',await page.locator('#document-mode-file').isEnabled());
    check('The background worker remains active',(await(await page.request.get(base+'/api/runtime')).json()).processing.active===1);
    await page.reload();
    const resume=page.getByRole('button',{name:'恢复任务：后台恢复验证.txt',exact:true});
    await resume.waitFor();
    check('Reload recovers an in-flight task without its source file',await page.locator('#document-file').evaluate(element=>element.files.length)===0);
    await resume.click();
    await page.waitForFunction(()=>document.getElementById('document-task-progress')?.value>10);
    check('A resumed task displays its existing progress',await page.locator('#document-task-progress').getAttribute('aria-valuetext')?.then(Boolean));
    await page.screenshot({path:'output/playwright/jobs-progress-desktop.png',fullPage:true});
    await page.setViewportSize({width:390,height:844});
    check('Task controls fit a 390px mobile screen',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    await page.screenshot({path:'output/playwright/jobs-progress-mobile.png',fullPage:true});
    await page.setViewportSize({width:1280,height:900});
    await page.locator('#document-annotated-text').waitFor({timeout:20000});
    check('A background task completes with the full original',await page.locator('#document-annotated-text').innerText()===text);
    const done=await(await page.request.get(base+'/api/document/jobs/'+background.job_id)).json();
    check('Completed metadata links to an existing report',done.status==='completed'&&Boolean(done.report_id));
    await page.reload();
    await page.getByRole('button',{name:'查看任务报告：后台恢复验证.txt',exact:true}).click();
    await page.locator('#document-annotated-text').waitFor();
    check('The completed task can reopen its report',await page.locator('#document-annotated-text').innerText()===text);
    const bad=await start('损坏任务验证.docx','broken archive');
    await page.waitForFunction(()=>!document.getElementById('document-mode-file').disabled);
    const failed=await(await page.request.get(base+'/api/document/jobs/'+bad.job_id)).json();
    check('Bad documents show a failed task with a readable error',failed.status==='failed'&&failed.error.status_code===422);
    check('Task failure preserves the report already being reviewed',await page.locator('#document-annotated-text').innerText()===text);
    check('No uncaught browser errors',errors.length===0);
    return {status:'passed',fixture:'deterministic slow model on localhost:8004',checks,errors};
  } finally {
    for(const id of ids){
      const response=await page.request.get(base+'/api/document/jobs/'+id);
      if(!response.ok())continue;
      const job=await response.json();
      if(!['completed','failed','cancelled'].includes(job.status))await page.request.post(base+'/api/document/jobs/'+id+'/cancel');
      if(job.report_id)await page.request.delete(base+'/api/document/reports/'+job.report_id);
    }
  }
}

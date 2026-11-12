async (page) => {
  let release, arrive, cancelSent=false, polls=0;
  const gate=new Promise(resolve=>release=resolve), ready=new Promise(resolve=>arrive=resolve);
  const job={job_id:'race-job',filename:'粘贴的长文',status:'running',stage:'analyzing',completed:1,total:10,progress:18,report_id:null,error:null};
  const downloads=[];
  page.on('download',download=>downloads.push(download.suggestedFilename()));
  await page.route('**/api/document/jobs',route=>route.fulfill({status:route.request().method()==='POST'?202:200,json:route.request().method()==='POST'?job:{jobs:[{...job,status:cancelSent?'cancel_requested':'running'}]}}));
  await page.route('**/api/document/jobs/race-job',async route=>{if(++polls===1){arrive();await gate;await route.fulfill({json:job});}else await route.fulfill({json:{...job,status:'cancel_requested'}});});
  await page.route('**/api/document/jobs/race-job/cancel',route=>{cancelSent=true;return route.fulfill({json:{...job,status:'cancel_requested'}});});
  await page.goto('http://127.0.0.1:8002/?v=race-before#document');
  await page.getByRole('button',{name:'粘贴长文',exact:true}).click();
  await page.locator('#document-text').fill('The team reviewed the source files before preparing the data.');
  await page.locator('#document-detect').click();await ready;
  await page.getByRole('button',{name:'取消任务',exact:true}).click();
  await page.waitForFunction(()=>document.getElementById('document-background').disabled);
  const stale=page.waitForResponse(response=>response.url().endsWith('/api/document/jobs/race-job'));
  release();await stale;await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  const result={before_fix:true,background_reenabled_by_stale_poll:await page.locator('#document-background').isEnabled(),cancel_reenabled_by_stale_poll:await page.locator('#document-cancel').isEnabled()};
  await page.locator('#document-background').click();
  await page.unrouteAll({behavior:'wait'});
  await page.goto('http://127.0.0.1:8002/?v=agent-download-before#agent');
  await page.evaluate(()=>window.dispatchEvent(new CustomEvent('aigc:document-result',{detail:{report_id:'missing-report',filename:'过期下载验证.txt',document_summary:{highlighted_total_percent:0,scored_character_percent:100},detection:{segments:[]}}})));
  const href=await page.getByRole('link',{name:'Word 报告',exact:true}).getAttribute('href');
  result.expired_agent_export_status=(await page.request.get('http://127.0.0.1:8002'+href)).status();
  result.inline_agent_download_error=await page.locator('.agent-download-status').count()>0;
  result.native_download_attempts=downloads.length;
  return result;
}

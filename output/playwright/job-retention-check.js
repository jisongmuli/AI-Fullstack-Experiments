async (page) => {
  const old={job_id:'old-active',filename:'最早仍在处理的任务.txt',status:'running',stage:'analyzing',completed:5,total:100,progress:14,report_id:null,error:null};
  const jobs=Array.from({length:19},(_,index)=>({...old,job_id:`completed-${index}`,filename:`已结束任务${index}.txt`,status:'completed',stage:'completed',progress:100,report_id:`report-${index}`}));
  jobs.push(old);
  await page.route('**/api/document/jobs',route=>route.fulfill({json:{jobs}}));
  try {
    await page.goto('http://127.0.0.1:8002/?v=job-priority-check#document');
    await page.reload();
    await page.getByRole('button',{name:'恢复任务：最早仍在处理的任务.txt',exact:true}).waitFor();
    const count=await page.locator('#document-jobs .document-job-row').count();
    if(count!==8)throw Error('The task panel did not preserve its bounded display');
    return {status:'passed',checks:[{name:'An older active task stays visible ahead of finished jobs',passed:true},{name:'The task panel keeps a bounded list of eight rows',passed:true}]};
  } finally {await page.unroute('**/api/document/jobs');}
}

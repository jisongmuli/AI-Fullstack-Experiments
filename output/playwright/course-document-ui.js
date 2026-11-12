async page => {
  const checks=[],errors=[];
  const check=(value,name)=>{if(!value)throw Error(name);checks.push({name,passed:true})};
  page.on('pageerror',error=>errors.push(error.message));
  let report;
  try{
    await page.goto('http://127.0.0.1:8006/?v=course-doc#agent');
    await page.reload();
    await page.setViewportSize({width:390,height:844});
    const original='研究团队核对样本来源和标签，保存训练、验证与测试的划分记录。模型结果经过逐项检查，报告保留原文与模型版本。'.repeat(3)+' 😀';
    await page.locator('#agent-document').setInputFiles({name:'课程联调.txt',mimeType:'text/plain',buffer:Buffer.from(original,'utf-8')});
    await page.locator('#agent-instruction').fill('检查这份材料并保留原文标注');
    const [response]=await Promise.all([page.waitForResponse(r=>r.url().endsWith('/api/agent/run')&&r.request().method()==='POST',{timeout:180000}),page.locator('#agent-submit').click()]);
    const data=await response.json();
    check(response.status()===200&&data.planner.tools[0]==='document_detection','A real local LLM selects document detection from an uploaded file');
    report=data.tool_calls[0].result.report_id;
    check(Boolean(report)&&data.tool_calls[0].http_status===200,'The real document API produces a stored report');
    await page.locator('#agent-result .open-document-report').click();
    await page.locator('#document-result .document-exports').waitFor();
    check(await page.locator('.nav-item[data-view="document"]').getAttribute('class').then(value=>value.includes('active')),'Agent results link to the document workflow');
    const result=await (await page.request.get('http://127.0.0.1:8006/api/document/reports/'+report+'/json')).json();
    check(result.extracted_text===original,'The Agent report retains the exact Unicode source');
    for(const [label,extension] of [['带下划线的 Word','docx'],['标注 HTML / 打印 PDF','html'],['完整结果 JSON','json']]){
      const [download]=await Promise.all([page.waitForEvent('download'),page.locator('#document-result').getByRole('button',{name:label,exact:true}).click()]);
      await download.saveAs('E:/AIGC_latest_20261011/output/playwright/course-downloads/agent-report.'+extension);
      check(download.suggestedFilename().endsWith('.'+extension),'Agent document export: '+extension);
    }
    check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Agent-linked document results fit the mobile viewport');
    await page.screenshot({path:'output/playwright/course-agent-document-mobile.png',fullPage:true});
    check(errors.length===0,'No uncaught errors in the Agent document handoff');
    return JSON.stringify({status:'passed',checks,errors,actual_models:true});
  }catch(error){return JSON.stringify({status:'failed',checks,errors,error:String(error)})}
  finally{if(report)await page.request.delete('http://127.0.0.1:8006/api/document/reports/'+report)}
}

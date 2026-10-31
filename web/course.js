(() => {
  'use strict';
  const byId = id => document.getElementById(id), esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const percent = value => Number.isFinite(value) ? (100*value).toFixed(2)+'%' : '—';
  const names = {model_info:'模型信息',course_overview:'课程实验',text_detection:'文本检测',image_detection:'图像检测',document_detection:'文档检测',clarity_edit:'英文编辑'};
  let records = null, busy = false, history = [], turns = [];
  let exampleRequest = null;
  async function get(path, options){const response=await fetch(path, options);const data=await response.json();if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:`请求失败（${response.status}）`);return data}
  async function loadCourse(){
    try{
      records=await get('/api/course');
      byId('course-stages').innerHTML=records.stages.map(stage=>`<article class="course-stage"><span class="course-step">实验 ${stage.number} · ${stage.ready?'已就绪':'待准备'}</span><h3>${esc(stage.name)}</h3><p>${esc(stage.description)}</p></article>`).join('');
      byId('course-records').textContent=JSON.stringify(records,null,2);
      const image=records.records['experiment2-image.json'],text=records.records['experiment2-text.json'],agent=records.records['agent/finetune-metrics.json'];
      const rows=[];
      for(const [label,value] of [['课程图像 · 80 张测试图',image],['课程中文 · 600 条测试文本',text]])if(value)rows.push(`<tr><td>${label}</td><td>${percent(value.trained.accuracy)}</td><td>${percent(value.optimized.accuracy)}</td><td>${value.trained.latency.median_ms.toFixed(3)} → ${value.optimized.latency.median_ms.toFixed(3)} ms</td></tr>`);
      if(agent)rows.push(`<tr><td>Agent 工具选择 · ${agent.after.samples} 条保留指令</td><td>${percent(agent.before.accuracy)}</td><td>${percent(agent.after.accuracy)}</td><td>LoRA · ${agent.steps} 步</td></tr>`);
      byId('course-measurements').innerHTML=`<h3>本轮实测对照</h3><table><thead><tr><th>样本与指标</th><th>训练基线 / 微调前</th><th>优化 / 微调后</th><th>单次推理中位数 / 训练</th></tr></thead><tbody>${rows.join('')}</tbody></table><p class="document-helper">图像和文本按 0.5 阈值统计。图像压缩展示速度、体积与识别表现的取舍；工具选择按指令与计划是否匹配统计。</p>`;
    }catch(error){byId('course-stages').textContent='实验记录暂不可用：'+error.message}
  }
  function setCourseProfile(){byId('model-profile').value='course-optimized';byId('model-profile').dispatchEvent(new Event('change'))}
  async function loadExample(kind){
    const status=byId('course-example-status');
    if(state.imageBusy||state.textBusy){status.textContent='检测正在进行，请完成后再载入课程样例。';return}
    if(exampleRequest)exampleRequest.abort();
    const request=new AbortController();exampleRequest=request;
    const selection=state.imageSelectionVersion,text=byId('text-input').value,profile=byId('model-profile').value;
    const buttons=[byId('course-try-image'),byId('course-try-text')];
    buttons.forEach(button=>{button.disabled=true});status.textContent='正在载入课程样例…';
    try{
      let value;
      if(kind==='image'){
        const response=await fetch('/api/course/example/image',{signal:request.signal});
        if(!response.ok)throw Error('课程样例图片暂不可用，请稍后重试。');
        value=new File([await response.blob()],'course-test-example.png',{type:'image/png'});
      }else value=(await get('/api/course/example/text',{signal:request.signal})).text;
      if(request.signal.aborted||exampleRequest!==request)return;
      if(state.view!=='course'||state.imageBusy||state.textBusy||state.imageSelectionVersion!==selection||byId('text-input').value!==text||byId('model-profile').value!==profile){
        status.textContent='输入或检测方案已更新，可重新点击载入样例。';return;
      }
      status.textContent='';setCourseProfile();
      if(kind==='image'){setImage(value);setView('image')}
      else{byId('text-input').value=value;updateText();setView('text')}
    }catch(error){if(error.name!=='AbortError'&&exampleRequest===request)status.textContent=error.message}
    finally{if(exampleRequest===request){exampleRequest=null;buttons.forEach(button=>{button.disabled=false});if(request.signal.aborted)status.textContent=''}}
  }
  function renderTurns(){
    byId('agent-result').innerHTML=turns.map(turn=>`<article class="agent-turn"><h3>${esc(turn.instruction)}</h3><div class="agent-answer">${esc(turn.answer)}</div><p class="agent-planner">${esc(turn.model)} · 大模型选择：${turn.planner.tools.map(tool=>esc(names[tool]||tool)).join(' → ')||'补充输入'} · ${esc(turn.profile==='course'?'课程优化方案':'增强方案')}</p>${turn.tool_calls.map(call=>`<section class="agent-tool"><h4>${esc(names[call.tool]||call.tool)} · ${call.status==='ok'?'完成':'未完成'}</h4><p>${esc(call.summary)}</p><small>${esc(call.path)} · HTTP ${call.http_status} · ${Math.round(call.duration_ms)} ms</small>${call.result.report_id?`<p><button type="button" class="secondary-button open-document-report" data-report-id="${esc(call.result.report_id)}">打开文档报告</button></p>`:''}${call.tool==='clarity_edit'&&call.result.edited_text?`<p>${esc(call.result.edited_text)}</p>`:''}<details><summary>查看 API 返回</summary><pre class="course-json">${esc(JSON.stringify(call.result,null,2))}</pre></details></section>`).join('')}</article>`).join('');
  }
  function updateBusy(value){busy=value;byId('agent-submit').disabled=value;byId('agent-clear').disabled=value;byId('agent-form').querySelectorAll('input,textarea,select,[data-agent-prompt]').forEach(node=>{node.disabled=value})}
  document.addEventListener('DOMContentLoaded',()=>{
    loadCourse();get('/api/agent/status').then(data=>{byId('agent-model-status').textContent=data.available?'Qwen3.5 + LoRA 已准备':'模型尚未准备'}).catch(()=>{byId('agent-model-status').textContent='暂未连接助手服务'});
    byId('course-open-agent').onclick=()=>setView('agent');
    byId('course-try-image').onclick=()=>loadExample('image');
    byId('course-try-text').onclick=()=>loadExample('text');
    window.addEventListener('aigc:view-changed',event=>{if(event.detail.view!=='course'&&exampleRequest)exampleRequest.abort()});
    byId('course-download').onclick=()=>{if(!records)return;const url=URL.createObjectURL(new Blob([JSON.stringify(records,null,2)],{type:'application/json'}));const anchor=document.createElement('a');anchor.href=url;anchor.download='五次实验记录_20261027.json';anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
    document.querySelectorAll('[data-agent-prompt]').forEach(button=>{button.onclick=()=>{byId('agent-instruction').value=button.dataset.agentPrompt}});
    byId('agent-clear').onclick=()=>{if(busy)return;history=[];turns=[];byId('agent-form').reset();byId('agent-result').innerHTML='<div class="empty-result"><strong>对话与附件已清空</strong><p>可以开始新的请求。</p></div>'};
    byId('agent-form').onsubmit=async event=>{
      event.preventDefault();if(busy)return;
      const instruction=byId('agent-instruction').value.trim(),text=byId('agent-text').value,image=byId('agent-image').files[0],documentFile=byId('agent-document').files[0];
      if(!instruction)return;
      const target=byId('agent-result');
      if(Array.from(text).length>10000){target.textContent='正文超过 10,000 字符，请缩短后重试。';return}
      if((image&&image.size>25*1048576)||(documentFile&&documentFile.size>20*1048576)){target.textContent='附件超过单文件大小限制，请缩小后重试。';return}
      if((image?.size||0)+(documentFile?.size||0)+new Blob([text]).size>34*1048576){target.textContent='本次附件与正文总量过大，请分次提交。';return}
      const data=new FormData();data.append('instruction',instruction);data.append('text',text);data.append('profile',byId('agent-profile').value);data.append('history',JSON.stringify(history.slice(-6)));if(image)data.append('image',image);if(documentFile)data.append('document',documentFile);
      updateBusy(true);target.innerHTML='<div class="busy-state"><span class="spinner"></span>大模型正在选择工具、调用接口并整理结果…</div>';
      try{
        const result=await get('/api/agent/run',{method:'POST',body:data});
        turns=[{...result,instruction},...turns].slice(0,6);history.push({role:'user',content:instruction.slice(0,1000)},{role:'assistant',content:result.answer.slice(0,1000)});history=history.slice(-6);renderTurns();
      }catch(error){renderTurns();target.insertAdjacentHTML('afterbegin',`<div class="error-state">${esc(error.message)}</div>`)}
      finally{updateBusy(false)}
    };
  });
})();

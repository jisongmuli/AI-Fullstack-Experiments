const state={view:"image",image:null,agentImage:null,agentDocument:null,agentBusy:false,mode:"clarity",models:null};
document.addEventListener("DOMContentLoaded",()=>{document.getElementById("run-llm-edit").onclick=runLLMEdit});
const $=id=>document.getElementById(id);
const icon=()=>window.lucide?.createIcons();
const escapeHtml=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"})[c]);
const fmtScore=value=>Math.round(Math.max(0,Math.min(1,Number(value)||0))*100);
const bytes=n=>n<1024?`${n} B`:n<1048576?`${(n/1024).toFixed(1)} KB`:`${(n/1048576).toFixed(1)} MB`;
function previewReason(result){if(result.status==="outside_scope")return"当前只支持 20 至 320 字符的英文段落。";const reason=String(result.reason||"");const issues=[];if(reason.includes("distinct edit"))issues.push("未产生有效改写");if(reason.includes("commentary"))issues.push("输出包含解释文字");if(reason.includes("unexpectedly long"))issues.push("输出长度异常");if(reason.includes("meaning safeguard"))issues.push("原意表面检查未通过");if(reason.includes("identifier"))issues.push("字母数字标识发生变化");return issues.length?`${issues.join("；")}。请保留原文并人工核对。`:"候选改写仅供参考；使用前请逐句核对事实和原意。"}
const api=async(path,options={})=>{const response=await fetch(path,options);let data;try{data=await response.json()}catch{data={detail:`HTTP ${response.status}`}}if(!response.ok)throw Error(typeof data.detail==="string"?data.detail:JSON.stringify(data.detail||data));return data};
const jsonPost=(path,payload)=>api(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
function setView(view){if(!["image","text","document","edit","agent"].includes(view))return;history.replaceState(null,"",`#${view}`);state.view=view;document.querySelectorAll(".nav-item").forEach(el=>el.classList.toggle("active",el.dataset.view===view));document.querySelectorAll(".view").forEach(el=>el.classList.toggle("active",el.id===`view-${view}`));$("page-title").textContent={image:"图像检测",text:"文本检测",document:"文档检测",edit:"可信编辑",agent:"Agent 工具工作流"}[view]||"内容可信";}
function setBusy(button,target,message){button.disabled=true;target.innerHTML=`<div class="busy-state"><span class="spinner"></span>${escapeHtml(message)}</div>`;}
function showError(target,error){target.innerHTML=`<div class="error-state">${escapeHtml(error.message||error)}</div><div class="empty-result"><i data-lucide="circle-alert"></i><strong>暂未生成结果</strong><p>请检查输入后重试。</p></div>`;icon()}
function decisionClass(decision){const value=String(decision||"").toLowerCase();if(value.includes("uncertain")||value.includes("abstain")||value.includes("无法"))return"";return value.includes("high")||value.includes("ai")||value.includes("生成")?"high":value.includes("low")||value.includes("human")||value.includes("真实")?"low":""}
function decisionLabel(decision){const value=String(decision||"").toLowerCase();if(value.includes("abstain")||value.includes("uncertain")||value.includes("无法")||value.includes("复核"))return"无法可靠判断";if(value.includes("high")||value.includes("ai")||value.includes("生成"))return"模型证据偏向生成";if(value.includes("low")||value.includes("human")||value.includes("真实"))return"模型证据偏向非生成";return"需人工复核"}
function evidenceList(value){if(Array.isArray(value))return value.map(item=>typeof item==="string"?item:item.advice||item.description||item.reason||JSON.stringify(item));if(typeof value==="string")return[value];return[]}
function provenanceHtml(value){
  if(!value||typeof value!=="object")return"";
  const labels={trusted:"本地核验受信任",valid_but_untrusted:"凭证完整，签发者未受信任",invalid:"凭证核验失败",invalid_or_unreadable:"凭证无法核验",credential_absent:"未发现内容凭证",unverified:"凭证状态不明",unavailable:"核验组件不可用",not_checked:"未核验"};
  const status=String(value.status||"unverified"),label=labels[status]||labels.unverified;
  const actions=Array.isArray(value.declared_actions)?value.declared_actions:[];
  const sources=Array.isArray(value.declared_source_types)?value.declared_source_types:[];
  const failures=Array.isArray(value.failure_codes)?value.failure_codes:[];
  return`<div class="provenance-card"><strong>内容凭证 · ${escapeHtml(label)}</strong><p>${escapeHtml(value.explanation||"")}</p>${actions.length?`<p>凭证声明的操作：${actions.map(escapeHtml).join("、")}</p>`:""}${sources.length?`<p>凭证声明的来源类型：${sources.map(escapeHtml).join("、")}</p>`:""}${failures.length?`<p>核验代码：${failures.map(escapeHtml).join("、")}</p>`:""}<small>凭证状态与像素模型分数分开呈现；缺少凭证不表示 AI 生成。</small></div>`;
}
function localEvidenceHtml(value){
  if(!value||typeof value!=="object")return"";
  const rows=value.grid_rows,cols=value.grid_cols,values=value.values;
  if(!Number.isInteger(rows)||!Number.isInteger(cols)||rows<1||cols<1||rows>16||cols>16||!Array.isArray(values)||values.length!==rows*cols)return"";
  if(!values.every(item=>typeof item==="number"&&Number.isFinite(item)))return"";
  const noStableResponse=values.every(item=>item<=.05);
  const tiles=values.map(item=>{
    const strength=Math.max(0,Math.min(1,item));
    const opacity=strength>.05?Math.min(.72,.06+strength*.66):0;
    return`<span class="local-evidence-tile" style="background-color:rgba(228,70,43,${opacity.toFixed(3)})"></span>`;
  }).join("");
  const explanation=typeof value.meaning==="string"?value.meaning:typeof value.explanation==="string"?value.explanation:"将图像划分为区域，比较遮挡前后模型输出的变化。";
  const method=value.method==="blur_occlusion_3x3"?"九宫格模糊遮挡":typeof value.method==="string"?value.method:"局部扰动分析";
  const toggle=noStableResponse?"":'<button type="button" class="local-evidence-toggle" aria-pressed="true">隐藏热图</button>';
  const legend=noStableResponse?"":'<div class="local-evidence-legend"><span>低</span><span class="local-evidence-gradient" aria-hidden="true"></span><span>高</span></div>';
  const empty=noStableResponse?'<p>本次遮挡试验没有稳定的局部正向响应；下方显示原图，不绘制热图。</p>':"";
  return`<div class="local-evidence-card"><div class="local-evidence-heading"><strong>图像局部敏感度</strong>${toggle}</div><div class="local-evidence-frame"><img class="local-evidence-image" alt="本次上传图片，局部分析详见下方文字"><div class="local-evidence-grid ${noStableResponse?"is-empty":""}" aria-hidden="true" style="grid-template-columns:repeat(${cols},1fr);grid-template-rows:repeat(${rows},1fr)">${tiles}</div></div>${legend}${empty}<p>${escapeHtml(explanation)}</p><small>方法：${escapeHtml(method)}。颜色表示模型对该区域变化的敏感度，不是 AI 像素、生成区域真值或 AI 所占比例。</small></div>`;
}
function attachLocalEvidenceImage(container,file){
  if(!file)return;
  container.querySelectorAll(".local-evidence-image").forEach(img=>{
    const objectUrl=URL.createObjectURL(file);
    img.addEventListener("load",()=>URL.revokeObjectURL(objectUrl),{once:true});
    img.addEventListener("error",()=>URL.revokeObjectURL(objectUrl),{once:true});
    img.src=objectUrl;
  });
}
function renderDetection(target,data,kind,file=null){
  const score=data.risk_score??data.score??data.ai_score;
  const hasScore=Number.isFinite(score);
  const percent=hasScore?fmtScore(score):0;
  const decision=data.decision??data.status??"uncertain";
  const cls=decisionClass(decision);
  const evidence=evidenceList(data.evidence||data.signals||data.reasons);
  const caveats=evidenceList(data.limitations||data.warnings);
  const version=data.model_version||data.version||"未提供";
  const provisional=["experimental_cross_source","low_resolution_experimental","unverified_platform","english_no_valid_external_benchmark"].includes(data.applicability);
  const imageCandidate=kind==="image"&&hasScore&&String(decision).toLowerCase().includes("uncertain");
  const badge=imageCandidate?"结论未校准":decisionLabel(decision);
  const caption=hasScore?(kind==="image"?data.applicability==="low_resolution_experimental"?"低分辨率探索性候选输出；不是 AI 率或来源概率":"候选模型分数；不是 AI 率或来源概率":provisional?"候选模型输出，未经目标平台校准":"模型风险分数，非作者身份概率"):"未给出可解释的模型分数";
  const modelScores=Array.isArray(data.model_scores)?data.model_scores.filter(item=>item&&Number.isFinite(item.score)):[];
  const breakdown=modelScores.length?`<div class="detail-section"><h4>候选模型分数</h4><div class="model-score-list">${modelScores.map(item=>`<div><span>${escapeHtml(item.model||"模型")}</span><strong>${fmtScore(item.score)}/100</strong></div>`).join("")}</div>${Number.isFinite(data.model_disagreement)?`<p>模型分歧：${fmtScore(data.model_disagreement)}/100。分数为统计输出，不能作为 AI 来源概率。</p>`:""}</div>`:"";
  target.innerHTML=`<div class="result-header"><div><small>检测结果</small><h3>${kind==="image"?"图片模式分析":"文本模式分析"}</h3></div><span class="decision ${cls}">${escapeHtml(badge)}</span></div>${kind==="image"&&hasScore?'<div class="risk-label">模型风险指数（实验性）</div>':""}<div class="risk-number">${hasScore?percent:"—"}<span>${hasScore?"/ 100":""}</span></div><div class="risk-caption">${caption}</div><div class="risk-track"><div class="risk-fill ${cls}" style="width:${percent}%"></div></div>${breakdown}${kind==="image"?localEvidenceHtml(data.local_evidence)+provenanceHtml(data.provenance):""}<div class="detail-section"><h4>可核对的依据</h4>${evidence.length?`<ul>${evidence.map(x=>`<li>${escapeHtml(x)}</li>`).join("")}</ul>`:`<p>当前模型未提供逐项证据；请结合输入质量理解分数。</p>`}</div><div class="detail-section"><h4>适用范围与限制</h4>${caveats.length?`<ul>${caveats.map(x=>`<li>${escapeHtml(x)}</li>`).join("")}</ul>`:`<p>当前模型尚未覆盖这一来源，无法提供可靠来源结论。</p>`}</div><div class="result-foot">模型版本：${escapeHtml(version)} · 模型输出不证明内容来源</div>`;
  if(kind==="image")attachLocalEvidenceImage(target,file);
  icon();
}
function segmentResultHtml(data){
  const segments=Array.isArray(data.segments)?data.segments:[];
  if(!segments.length)return"";
  const summary=data.segment_summary||{};
  const highlighted=summary.highlighted_character_percent;
  const scored=summary.scored_character_percent;
  const coverage=Number.isFinite(highlighted)?`<div class="segment-coverage"><strong>${highlighted.toFixed(1)}%</strong><span>已评分字符中处于高亮段落的比例</span></div>`:"";
  const evaluated=Number.isFinite(scored)?`<p class="finding-note">可评分字符覆盖：${scored.toFixed(1)}%。高亮覆盖率不是 AI 撰写比例，也不能证明作者身份。</p>`:`<p class="finding-note">段落分数不是 AI 撰写比例，也不能证明作者身份。</p>`;
  return`<div class="agent-segments"><h4>段落标注</h4>${coverage}${evaluated}${segments.map((segment,index)=>{
    const flagged=segment.highlighted===true;
    const score=Number.isFinite(segment.risk_score)?` · 模型分数 ${fmtScore(segment.risk_score)}/100`:" · 未给分";
    const label=flagged?"疑似生成特征":!Number.isFinite(segment.risk_score)?"未评分 / 需复核":segment.decision==="uncertain"?"分数不确定 / 需复核":"未高亮";
    const reason=Array.isArray(segment.evidence)?segment.evidence[0]:null;
    return`<div class="agent-segment ${flagged?"flagged":""}"><strong>段落 ${index+1} · ${label}</strong><span class="segment-score">${score}</span><p>${escapeHtml(segment.text||"")}</p>${reason?`<small>${escapeHtml(reason)}</small>`:""}</div>`;
  }).join("")}</div>`;
}
function validateImage(file){if(!file)return"请选择图片";if(!["image/png","image/jpeg"].includes(file.type))return"仅支持 JPG 或 PNG";if(file.size>25*1048576)return"文件不能超过 25 MB";return null}
function setImage(file){const error=validateImage(file);if(error){clearImage();showError($("image-result"),error);return}state.image=file;$("agent-use-image").disabled=false;const preview=$("image-preview"),caption=$("preview-caption"),dropzone=$("image-dropzone");if(preview.dataset.objectUrl)URL.revokeObjectURL(preview.dataset.objectUrl);preview.dataset.objectUrl=URL.createObjectURL(file);preview.onload=()=>{const side=Math.min(preview.naturalWidth,preview.naturalHeight),small=preview.naturalWidth===32&&preview.naturalHeight===32;dropzone.classList.toggle("pixel-preview",small);const scope=small?" · 放大预览（CIFAKE 教学模型）":side<200?" · 不评分":side<512?" · 仅探索性候选分数":"";caption.textContent=`原图 ${preview.naturalWidth} × ${preview.naturalHeight} 像素${scope}`;caption.hidden=false};preview.src=preview.dataset.objectUrl;preview.hidden=false;dropzone.classList.add("has-image");$("file-meta").hidden=false;$("file-name").textContent=file.name;$("file-size").textContent=bytes(file.size);$("detect-image").disabled=false;$("image-result").innerHTML=`<div class="empty-result"><i data-lucide="scan-line"></i><strong>图片已就绪</strong><p>点击分析图片查看模型输出。</p></div>`;updateAgentInput();icon()}
function clearImage(){state.image=null;$("agent-use-image").checked=false;$("agent-use-image").disabled=true;const preview=$("image-preview");if(preview.dataset.objectUrl)URL.revokeObjectURL(preview.dataset.objectUrl);preview.removeAttribute("src");preview.hidden=true;$("preview-caption").hidden=true;$("image-input").value="";$("image-dropzone").classList.remove("has-image","pixel-preview");$("file-meta").hidden=true;$("detect-image").disabled=true;$("image-result").innerHTML=`<div class="empty-result"><i data-lucide="scan-line"></i><strong>等待图片</strong><p>选择图片后查看模型输出、适用范围与判断状态。</p></div>`;updateAgentInput();icon()}
async function useSample(kind){try{const response=await fetch(`/static/assets/sample-${kind}.png`);const blob=await response.blob();setImage(new File([blob],`cifake-${kind}-sample.png`,{type:"image/png"}))}catch(error){showError($("image-result"),error)}}
async function detectImage(){const button=$("detect-image"),target=$("image-result"),file=state.image;setBusy(button,target,"正在分析图片…");try{const body=new FormData();body.append("file",file);const result=await api("/api/image/detect",{method:"POST",body});if(state.image===file)renderDetection(target,result,"image",file)}catch(error){if(state.image===file)showError(target,error)}finally{button.disabled=!state.image}}
function updateText(){const value=$("text-input").value,length=Array.from(value).length;$("text-count").textContent=`${length} / 10,000 字符`;const valid=Boolean(value.trim())&&length<=10000;$("detect-text").disabled=!valid;$("text-hint").textContent=length>10000?"长于 10,000 字符，请使用文档检测":valid?"可开始分析；短段落可能无法评分":"请输入文本"}
async function detectText(){const button=$("detect-text"),target=$("text-result"),text=$("text-input").value;setBusy(button,target,"正在分析文本…");try{const result=await jsonPost("/api/text/detect",{text});renderDetection(target,result,"text");if(window.AIGCDocument)target.append(window.AIGCDocument.renderInline(text,result.segments,{showUnscored:true}));target.insertAdjacentHTML("beforeend",segmentResultHtml(result))}catch(error){showError(target,error)}finally{updateText()}}
function updateEdit(){const length=$("edit-input").value.trim().length;$("edit-count").textContent=`${length} 字符`;$("run-edit").disabled=length<20;$("run-llm-edit").disabled=length<20}
function renderEdit(target,data){const edited=data.edited_text??data.revised_text??data.text??"";const suggestions=evidenceList(data.suggestions||data.changes);const checks=Object.entries(data.meaning_check?.checks||{}).map(([name,passed])=>`${name}: ${passed?"通过":"需核对"}`);const applied=data.applied_edits||[];target.innerHTML=`<div class="result-header"><div><small>编辑建议</small><h3>改写预览</h3></div><span class="decision">请作者确认</span></div><div class="edit-output" id="edit-output">${escapeHtml(edited||"未生成改写文本")}</div><div class="result-actions"><button class="secondary-button" id="copy-edit" type="button">复制建议</button><button class="secondary-button" id="use-edit" type="button">放入原文框</button></div><div class="detail-section"><h4>修改与建议</h4><p>自动修改 ${applied.length} 处；人工建议 ${suggestions.length} 条。</p>${suggestions.length?`<ul>${suggestions.map(s=>`<li>${escapeHtml(s)}</li>`).join("")}</ul>`:"<p>未发现预设规则覆盖的问题，请仍逐句核对。</p>"}</div><div class="detail-section"><h4>原意保留检查</h4>${checks.length?`<ul>${checks.map(s=>`<li>${escapeHtml(s)}</li>`).join("")}</ul>`:"<p>数字、姓名、引文和事实仍需作者核对。</p>"}<p>${escapeHtml(data.meaning_check?.limitation||"表面检查不证明语义等价。")}</p></div><div class="result-foot">编辑结果不用于规避检测；确认后再使用。</div>`;$("copy-edit").onclick=()=>navigator.clipboard.writeText(edited);$("use-edit").onclick=()=>{$("edit-input").value=edited;updateEdit()}}
async function runEdit(){const button=$("run-edit"),target=$("edit-result"),text=$("edit-input").value;setBusy(button,target,"正在生成建议…");try{renderEdit(target,await jsonPost("/api/text/edit",{text}))}catch(error){showError(target,error)}finally{button.disabled=text.trim().length<20}}
async function runLLMEdit(){const button=$("run-llm-edit"),target=$("edit-result"),value=$("edit-input").value;setBusy(button,target,"正在生成大模型预览…");try{const result=await jsonPost("/api/text/edit/llm-preview",{text:value});const accepted=result.status==="review_candidate";target.innerHTML=`<div class="result-header"><div><small>大模型编辑预览</small><h3>${accepted?"待作者核对":"未提供可用改写"}</h3></div><span class="decision">${accepted?"人工确认":"已拦截"}</span></div>${accepted?`<div class="edit-output">${escapeHtml(result.candidate)}</div>`:""}<div class="detail-section"><h4>检查结果</h4><p>${escapeHtml(previewReason(result))}</p><p>模型建议不能证明原意保留，也不会自动替换原文。</p></div>`}catch(error){showError(target,error)}finally{button.disabled=value.trim().length<20}}
function addMessage(role,content){const intro=document.querySelector(".chat-intro");if(intro)intro.remove();const el=document.createElement("div");el.className=`message ${role}`;el.innerHTML=`<span class="speaker">${role==="user"?"你":"Agent"}</span><div class="bubble">${escapeHtml(content)}</div>`;$("chat-messages").append(el);$("chat-messages").scrollTop=$("chat-messages").scrollHeight;return el}
function renderTrace(data){const calls=data.tool_calls||[];const names={image_detection:"图像检测",text_detection:"文本检测",clarity_edit:"可信编辑",llm_edit_preview:"大模型编辑预览"};$("tool-trace").innerHTML=calls.length?calls.map(call=>`<div class="trace-item"><strong>${escapeHtml(names[call.tool]||call.tool||"工具")}</strong><span>${call.status==="ok"?"完成":"不可用"} · ${escapeHtml(call.duration_ms??"—")} ms</span></div>`).join(""):`<p class="muted">此轮没有工具调用记录。</p>`}
function fileToBase64(file){return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result).split(",")[1]);reader.onerror=()=>reject(Error("无法读取图片"));reader.readAsDataURL(file)})}
function agentDocumentHtml(report){
  const summary=report.document_summary||{},rate=summary.highlighted_total_percent;
  const pct=value=>Number.isFinite(value)?`${value.toFixed(1)}%`:"未评分";
  const id=typeof report.report_id==="string"?report.report_id:"";
  const segments=Array.isArray(report.detection?.segments)?report.detection.segments:[];
  const samples=segments.filter(item=>item.highlighted===true).slice(0,3);
  const base=`/api/document/reports/${encodeURIComponent(id)}`;
  return`<div class="agent-finding"><strong>文档检测 · ${escapeHtml(report.filename||"文档")}</strong><span class="finding-score">${pct(rate)}</span><span>AI 特征覆盖率：高风险段字符占全文非空白字符的比例</span><span class="finding-note">不是实际 AI 撰写比例。可评分覆盖 ${pct(summary.scored_character_percent)} · ${escapeHtml(summary.high_risk_segments??0)} 个高风险片段</span>${Number.isFinite(summary.mean_segment_score)?`<span>可评分片段模型均分 ${fmtScore(summary.mean_segment_score)}/100</span>`:""}${samples.map(item=>`<p class="document-agent-excerpt"><u>${escapeHtml(Array.from(item.text||"").slice(0,220).join(""))}${Array.from(item.text||"").length>220?"…":""}</u></p>`).join("")}${summary.mixed_signals?'<span class="finding-note">段落信号存在明显差异，未提供整篇来源分数。</span>':""}${id?`<div class="result-actions"><button type="button" class="secondary-button open-document-report" data-report-id="${escapeHtml(id)}">完整下划线标注</button><a class="secondary-button" href="${base}/docx" download>Word 报告</a><a class="secondary-button" href="${base}/html" download>HTML / 打印 PDF</a></div>`:""}<span class="finding-note">报告暂存在本地内存；过期或重启后请重新检测。</span></div>`;
}
function agentResultHtml(data){
  const findings=Object.entries(data.findings||{}).map(([name,value])=>{
    const result=value&&typeof value==="object"?value:{};
    if(name==="document")return agentDocumentHtml(result);
    const hasScore=Number.isFinite(result.risk_score);
    const score=hasScore?`${fmtScore(result.risk_score)}/100`:"未给分";
    const provisional=["experimental_cross_source","low_resolution_experimental","unverified_platform","english_no_valid_external_benchmark"].includes(result.applicability);
    const note=provisional?"未经目标平台校准的候选分数":"模型风险分数，不代表 AI 所占比例";
    const modelScores=Array.isArray(result.model_scores)?result.model_scores.filter(item=>item&&Number.isFinite(item.score)):[];
    const components=modelScores.length?`<span class="finding-note">${modelScores.map(item=>`${escapeHtml(item.model||"模型")} ${fmtScore(item.score)}/100`).join(" · ")}</span>`:"";
    const title=name==="image"?"图像模型风险指数（实验性）":name==="text"?"文本模型风险分数":`${escapeHtml(name)} 模型风险分数`;
    const decisionText=name==="image"&&hasScore&&String(result.decision).toLowerCase().includes("uncertain")?"模型筛查分数 · 结论未校准":decisionLabel(result.decision);
    return`<div class="agent-finding"><strong>${title}</strong><span class="finding-score">${score}</span><span>${escapeHtml(decisionText)}</span><span class="finding-note">${note}</span>${components}<span>${escapeHtml(evidenceList(result.evidence||result.signals)[0]||"")}</span>${name==="image"?localEvidenceHtml(result.local_evidence)+provenanceHtml(result.provenance):""}${name==="text"?segmentResultHtml(result):""}</div>`;
  }).join("");
  const edit=data.editing_result,llm=data.llm_preview;
  const status={no_input:"未提交内容",insufficient_evidence:"证据不足",mixed_signals:"检测信号不一致",review_recommended:"模型证据已汇总"};
  const explanation={no_input:"未提交图像或文本。",insufficient_evidence:"至少一个检测工具没有给分，当前无法形成完整结论。",mixed_signals:"图像与文本工具的结论不同；当前没有经过验证的跨模态融合模型。",review_recommended:"图像、文本分别展示候选分数和局部证据；分数不能解释为 AI 内容比例或来源结论。"};
  const scored=Object.entries(data.findings||{}).filter(([,result])=>Number.isFinite(result?.risk_score)).map(([name])=>name);
  const partial=data.decision==="insufficient_evidence"&&scored.length;
  const title=partial?scored.length===1&&scored[0]==="image"?"图像模型结果":scored.length===1&&scored[0]==="text"?"文本模型结果":"部分模型有分数":status[data.decision]||"模型证据摘要";
  const summary=partial?"已显示可用模型的候选分数；其他模态未提交或无法评分。分数不能证明内容来源。":explanation[data.decision]||data.explanation||"";
  return`<div class="agent-summary"><strong>${escapeHtml(title)}</strong><p>${escapeHtml(summary)}</p></div><div class="agent-findings">${findings||"未运行检测工具"}${edit?`<div class="agent-finding"><strong>可信编辑</strong><span>${escapeHtml(edit.applied_edits?.length||0)} 处自动修改，${escapeHtml(edit.suggestions?.length||0)} 条人工建议</span></div>`:""}${llm?`<div class="agent-finding"><strong>大模型编辑预览</strong><span>${llm.status==="review_candidate"?"候选改写，需人工核对":llm.status==="outside_scope"?"超出预览范围":"输出未通过检查"}</span>${llm.status==="review_candidate"?`<span>${escapeHtml(llm.candidate)}</span>`:""}</div>`:""}</div>`;
}
function setAgentImage(file){
  const error=validateImage(file);
  if(error){clearAgentImage();addMessage("assistant",error);return}
  state.agentImage=file;
  $("agent-image-name").textContent=`${file.name} · ${bytes(file.size)}`;
  $("agent-image-name").hidden=false;
  $("agent-clear-image").hidden=false;
  $("agent-use-image").checked=false;
  updateAgentInput();
}
function clearAgentImage(){state.agentImage=null;$("agent-image-input").value="";$("agent-image-name").hidden=true;$("agent-clear-image").hidden=true;updateAgentInput()}
function updateAgentInput(){const length=Array.from($("agent-input").value).length;const file=state.agentImage||($("agent-use-image").checked?state.image:null);const conflict=Boolean(state.agentDocument&&$("agent-input").value.trim());$("agent-text-count").textContent=conflict?"文档和粘贴文本请分别提交":`${length} / 10,000 字符${length>10000?" · 长文请上传为文档":""}`;$("agent-text-count").classList.toggle("over-limit",length>10000||conflict);$("agent-form").querySelector("button").disabled=state.agentBusy||length>10000||conflict||(!$("agent-input").value.trim()&&!file&&!state.agentDocument)}
function setAgentDocument(file){
  if(!file)return;
  if(!/\.(docx|pdf|txt|md)$/i.test(file.name)||file.size>20*1048576){addMessage("assistant","文档支持 DOCX、PDF、TXT、MD，最大 20 MB。");return}
  state.agentDocument=file;
  $("agent-document-name").textContent=`${file.name} · ${bytes(file.size)}`;
  $("agent-document-name").hidden=false;$("agent-clear-document").hidden=false;updateAgentInput();
}
function clearAgentDocument(){state.agentDocument=null;$("agent-document-input").value="";$("agent-document-name").hidden=true;$("agent-clear-document").hidden=true;updateAgentInput()}
async function importAgentText(file){
  if(!file)return;
  try{
    if(!/\.(txt|md)$/i.test(file.name))throw Error("仅支持 TXT 或 Markdown 文本文件。");
    if(file.size>100000)throw Error("文本文件超过 100 KB，请选择较短的文件。");
    const content=new TextDecoder("utf-8",{fatal:true}).decode(await file.arrayBuffer());
    if(content.length>10000)throw Error("文本超过 10,000 字符，请缩短后重试。");
    $("agent-input").value=content;
    $("agent-input").focus();
    updateAgentInput();
  }catch(error){addMessage("assistant",error.message||"无法导入文本文件。")}
  finally{$("agent-text-file").value=""}
}
async function sendAgent(message){
  const file=state.agentImage||($("agent-use-image").checked?state.image:null);
  const documentFile=state.agentDocument;
  if((!message.trim()&&!file&&!documentFile)||Array.from(message).length>10000||state.agentBusy||(documentFile&&message.trim()))return;
  const preview=message.length>160?`${message.slice(0,160)}…`:message;
  const userMessage=addMessage("user",`${file?`[${file.name}] `:""}${documentFile?`[${documentFile.name}] `:""}${preview||"开始检测附件"}`);
  if(file){const url=URL.createObjectURL(file);const image=document.createElement("img");image.className="agent-preview";image.alt=`已上传图片：${file.name}`;image.onload=()=>URL.revokeObjectURL(url);image.onerror=()=>URL.revokeObjectURL(url);image.src=url;userMessage.querySelector(".bubble").append(image)}
  const pending=addMessage("assistant","正在调用工具…");
  state.agentBusy=true;
  updateAgentInput();
  try{
    const payload=new FormData();
    payload.append("text",message);
    payload.append("edit",String($("agent-use-edit").checked));
    payload.append("llm_preview",String($("agent-use-llm").checked));
    if(file)payload.append("file",file);
    if(documentFile)payload.append("document",documentFile);
    const data=await api("/api/agent/upload",{method:"POST",body:payload});
    pending.querySelector(".bubble").innerHTML=agentResultHtml(data);
    attachLocalEvidenceImage(pending,file);
    if(message&&data.findings?.text&&window.AIGCDocument)pending.querySelector(".bubble").append(window.AIGCDocument.renderInline(message,data.findings.text.segments));
    renderTrace(data);
  }catch(error){pending.querySelector(".bubble").textContent=`请求失败：${error.message}`}
  finally{state.agentBusy=false;updateAgentInput()}
}
async function showModelInfo(){const dialog=$("model-dialog");dialog.showModal();try{const data=await api("/api/models");state.models=data;const scope=data.scope||{};const versions=data.manifest||{};$("model-details").innerHTML=`<p>${data.available?"模型已加载":"模型暂不可用"}${data.reason?`：${escapeHtml(data.reason)}`:""}</p><h3>图像检测</h3><p>${escapeHtml(scope.image||"适用范围未提供")}</p><h3>文本检测</h3><p>${escapeHtml(scope.text||"适用范围未提供")}</p><h3>可信编辑</h3><p>${escapeHtml(scope.editing||"适用范围未提供")}</p><h3>模型清单</h3><p>${escapeHtml(JSON.stringify(versions))}</p><p>所有检测结果都需要人工复核。</p>`}catch(error){$("model-details").innerHTML=`<p class="muted">模型信息暂不可用：${escapeHtml(error.message)}</p>`}}
async function checkHealth(){try{await api("/health");$("service-dot").className="status-dot ready";$("service-label").textContent="服务已连接"}catch{$("service-dot").className="status-dot error";$("service-label").textContent="服务未连接"}}
document.addEventListener("DOMContentLoaded",()=>{icon();document.querySelectorAll(".nav-item").forEach(el=>el.onclick=()=>setView(el.dataset.view));$("image-input").onchange=event=>setImage(event.target.files[0]);$("clear-image").onclick=clearImage;const drop=$("image-dropzone");["dragenter","dragover"].forEach(name=>drop.addEventListener(name,event=>{event.preventDefault();drop.classList.add("dragging")}));["dragleave","drop"].forEach(name=>drop.addEventListener(name,event=>{event.preventDefault();drop.classList.remove("dragging")}));drop.addEventListener("drop",event=>setImage(event.dataTransfer.files[0]));document.querySelectorAll(".sample-button").forEach(el=>el.onclick=()=>useSample(el.dataset.sample));$("detect-image").onclick=detectImage;$("text-input").oninput=updateText;$("detect-text").onclick=detectText;$("edit-input").oninput=updateEdit;$("run-edit").onclick=runEdit;$("agent-input").oninput=updateAgentInput;$("agent-use-image").onchange=()=>{if($("agent-use-image").checked)clearAgentImage();updateAgentInput()};$("agent-form").onsubmit=event=>{event.preventDefault();sendAgent($("agent-input").value)};$("model-info-button").onclick=showModelInfo;$("close-dialog").onclick=()=>$("model-dialog").close();$("dialog-done").onclick=()=>$("model-dialog").close();updateAgentInput();checkHealth()});
document.addEventListener("DOMContentLoaded",()=>{
  $("agent-image-input").onchange=event=>{if(event.target.files[0])setAgentImage(event.target.files[0])};
  $("agent-clear-image").onclick=clearAgentImage;
  $("agent-text-file").onchange=event=>importAgentText(event.target.files[0]);
  if($("agent-document-input"))$("agent-document-input").onchange=event=>setAgentDocument(event.target.files[0]);
  if($("agent-clear-document"))$("agent-clear-document").onclick=clearAgentDocument;
});
window.addEventListener("aigc:document-result",event=>{
  const report=event.detail;
  if(!report||typeof report.report_id!=="string"||!report.document_summary)return;
  setView("agent");
  const message=addMessage("assistant","");
  message.querySelector(".bubble").innerHTML=agentDocumentHtml(report);
  $("tool-trace").innerHTML='<p class="muted">展示已完成的文档检测结果，本次没有重新调用模型。</p>';
});
document.addEventListener("click",event=>{
  const reportButton=event.target.closest?.(".open-document-report");
  if(reportButton){setView("document");window.dispatchEvent(new CustomEvent("aigc:open-document-report",{detail:{report_id:reportButton.dataset.reportId}}));return}
  const button=event.target.closest?.(".local-evidence-toggle");
  if(!button)return;
  const card=button.closest(".local-evidence-card");
  const hidden=card.classList.toggle("is-original-only");
  button.setAttribute("aria-pressed",String(!hidden));
  button.textContent=hidden?"显示热图":"隐藏热图";
});

function openLinkedView(){const view=location.hash.slice(1);if(["image","text","document","edit","agent"].includes(view))setView(view)}
document.addEventListener("DOMContentLoaded",openLinkedView);
window.addEventListener("hashchange",openLinkedView);

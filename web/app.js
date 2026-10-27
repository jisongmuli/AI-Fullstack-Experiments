const state={view:"image",image:null,imageReady:false,imageBusy:false,imageSelectionVersion:0,textBusy:false,textResultSource:null,editBusy:false,editResultSource:null,healthBusy:false,healthTimer:null,mode:"clarity",models:null,modelInfoVersion:0};
document.addEventListener("DOMContentLoaded",()=>{document.getElementById("run-llm-edit").onclick=runLLMEdit});
const $=id=>document.getElementById(id);
const icon=()=>window.lucide?.createIcons();
const escapeHtml=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"})[c]);
const fmtScore=value=>Math.round(Math.max(0,Math.min(1,Number(value)||0))*100);
const pct=value=>`${Number(value).toFixed(1)}%`;
const bytes=n=>n<1024?`${n} B`:n<1048576?`${(n/1024).toFixed(1)} KB`:`${(n/1048576).toFixed(1)} MB`;
function previewReason(result){if(result.status==="outside_scope")return"当前只支持 20 至 320 字符的英文段落。";const reason=String(result.reason||"");const issues=[];if(reason.includes("distinct edit"))issues.push("未产生有效改写");if(reason.includes("commentary"))issues.push("输出包含解释文字");if(reason.includes("unexpectedly long"))issues.push("输出长度异常");if(reason.includes("meaning safeguard"))issues.push("原意表面检查未通过");if(reason.includes("identifier"))issues.push("字母数字标识发生变化");return issues.length?`${issues.join("；")}。请保留原文并人工核对。`:"候选改写仅供参考；使用前请逐句核对事实和原意。"}
const api=async(path,options={})=>{const response=await fetch(path,options);let data;try{data=await response.json()}catch{data={detail:`HTTP ${response.status}`}}if(!response.ok)throw Error(typeof data.detail==="string"?data.detail:JSON.stringify(data.detail||data));return data};
const jsonPost=(path,payload)=>api(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
function setView(view){if(!["image","text","document","edit","course","agent"].includes(view))return;history.replaceState(null,"",`#${view}`);state.view=view;document.querySelectorAll(".nav-item").forEach(el=>el.classList.toggle("active",el.dataset.view===view));document.querySelectorAll(".view").forEach(el=>el.classList.toggle("active",el.id===`view-${view}`));$("page-title").textContent={image:"图像检测",text:"文本检测",document:"文档检测",edit:"可信编辑",course:"课程实验",agent:"大模型助手"}[view]||"内容可信";window.dispatchEvent(new CustomEvent("aigc:view-changed",{detail:{view}}));}
function detectionProfile(){const value=$("model-profile")?.value||"advanced";return{course:value.startsWith("course-"),stage:value==="course-trained"?"trained":"optimized"}}
function updateProfileControl(){if($("model-profile"))$("model-profile").disabled=state.imageBusy||state.textBusy}
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
  const badge=imageCandidate?"需要复核":decisionLabel(decision);
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
  return`<div class="text-segments"><h4>段落标注</h4>${coverage}${evaluated}${segments.map((segment,index)=>{
    const flagged=segment.highlighted===true;
    const score=Number.isFinite(segment.risk_score)?` · 模型分数 ${fmtScore(segment.risk_score)}/100`:" · 未给分";
    const label=flagged?"疑似生成特征":!Number.isFinite(segment.risk_score)?"未评分 / 需复核":segment.decision==="uncertain"?"分数不确定 / 需复核":"未高亮";
    const reason=Array.isArray(segment.evidence)?segment.evidence[0]:null;
    return`<div class="text-segment ${flagged?"flagged":""}"><strong>段落 ${index+1} · ${label}</strong><span class="segment-score">${score}</span><p>${escapeHtml(segment.text||"")}</p>${reason?`<small>${escapeHtml(reason)}</small>`:""}</div>`;
  }).join("")}</div>`;
}
function validateImage(file){if(!file)return"请选择图片";if(!["image/png","image/jpeg"].includes(file.type))return"仅支持 JPG 或 PNG";if(file.size>25*1048576)return"文件不能超过 25 MB";return null}
function setImage(file){
  const selection=++state.imageSelectionVersion,error=validateImage(file);
  if(error){clearImage();showError($("image-result"),error);return}
  state.image=file;state.imageReady=false;
  const preview=$("image-preview"),caption=$("preview-caption"),dropzone=$("image-dropzone");
  if(preview.dataset.objectUrl)URL.revokeObjectURL(preview.dataset.objectUrl);
  preview.dataset.objectUrl=URL.createObjectURL(file);
  preview.hidden=true;caption.hidden=true;dropzone.classList.remove("pixel-preview");
  $("detect-image").disabled=true;
  $("image-result").innerHTML='<div class="busy-state"><span class="spinner"></span>正在读取图片…</div>';
  preview.onload=()=>{
    if(selection!==state.imageSelectionVersion||state.image!==file)return;
    const width=preview.naturalWidth,height=preview.naturalHeight;
    if(!width||!height||width*height>25000000){clearImage();showError($("image-result"),Error("图片超过 2500 万像素，请缩小后重试。"));return}
    state.imageReady=true;preview.hidden=false;
    dropzone.classList.toggle("pixel-preview",width===32&&height===32);
    caption.textContent=`原图 ${width} × ${height} 像素${Math.min(width,height)<200?" · 小尺寸图片":""}`;caption.hidden=false;
    $("detect-image").disabled=state.imageBusy;
    $("image-result").innerHTML='<div class="empty-result"><i data-lucide="scan-line"></i><strong>图片已就绪</strong><p>点击分析图片查看模型输出。</p></div>';icon();
  };
  preview.onerror=()=>{if(selection===state.imageSelectionVersion){clearImage();showError($("image-result"),Error("图片无法读取，请选择有效的 JPG 或 PNG 文件。"))}};
  preview.src=preview.dataset.objectUrl;dropzone.classList.add("has-image");
  $("file-meta").hidden=false;$("file-name").textContent=file.name;$("file-size").textContent=bytes(file.size);
}
function clearImage(){++state.imageSelectionVersion;state.image=null;state.imageReady=false;const preview=$("image-preview");preview.onload=null;preview.onerror=null;if(preview.dataset.objectUrl)URL.revokeObjectURL(preview.dataset.objectUrl);delete preview.dataset.objectUrl;preview.removeAttribute("src");preview.hidden=true;$("preview-caption").hidden=true;$("image-input").value="";$("image-dropzone").classList.remove("has-image","pixel-preview");$("file-meta").hidden=true;$("detect-image").disabled=true;$("image-result").innerHTML=`<div class="empty-result"><i data-lucide="scan-line"></i><strong>等待图片</strong><p>选择图片后查看模型输出、适用范围与判断状态。</p></div>`;icon()}
async function useSample(kind){
  const version=++state.imageSelectionVersion;
  try{
    const response=await fetch(`/static/assets/sample-${kind}-2026.jpg`);
    if(!response.ok)throw Error(`示例图片加载失败（${response.status}），请重新选择图片。`);
    const blob=await response.blob();
    if(version===state.imageSelectionVersion)setImage(new File([blob],`ntire-2026-${kind}.jpg`,{type:"image/jpeg"}));
  }catch(error){if(version===state.imageSelectionVersion){clearImage();showError($("image-result"),error)}}
}
async function detectImage(){
  if(state.imageBusy||!state.image||!state.imageReady)return;
  const button=$("detect-image"),target=$("image-result"),file=state.image;
  state.imageBusy=true;setBusy(button,target,"正在分析图片…");
  updateProfileControl();
  try{const body=new FormData();body.append("file",file);const profile=detectionProfile(),path=profile.course?`/api/course/image/detect?stage=${profile.stage}`:"/api/image/detect";const result=await api(path,{method:"POST",body});if(state.image===file)renderDetection(target,result,"image",file)}
  catch(error){if(state.image===file)showError(target,error)}
  finally{state.imageBusy=false;button.disabled=!state.image||!state.imageReady;updateProfileControl()}
}
function inputUpdated(target){target.innerHTML='<div class="empty-result"><i data-lucide="file-pen-line"></i><strong>输入已更新</strong><p>请重新提交，查看当前文本的结果。</p></div>';icon()}
function updateText(){const value=$("text-input").value,length=Array.from(value).length;if(state.textResultSource!==null&&state.textResultSource!==value){state.textResultSource=null;inputUpdated($("text-result"))}$("text-count").textContent=`${length} / 10,000 字符`;const valid=Boolean(value.trim())&&length<=10000;$("detect-text").disabled=state.textBusy||!valid;$("text-hint").textContent=length>10000?"长于 10,000 字符，请使用文档检测":state.textBusy?"分析中；修改文本后请重新分析":valid?"可开始分析；短段落可能无法评分":"请输入文本"}
async function detectText(){
  const button=$("detect-text"),target=$("text-result"),text=$("text-input").value;
  if(state.textBusy||!text.trim()||Array.from(text).length>10000)return;
  state.textBusy=true;state.textResultSource=null;setBusy(button,target,"正在分析文本…");updateText();
  updateProfileControl();
  try{const profile=detectionProfile(),result=await jsonPost(profile.course?"/api/course/text/detect":"/api/text/detect",profile.course?{text,stage:profile.stage}:{text});if($("text-input").value!==text)return;renderDetection(target,result,"text");if(window.AIGCDocument&&Array.isArray(result.segments))target.append(window.AIGCDocument.renderInline(text,result.segments,{showUnscored:true}));target.insertAdjacentHTML("beforeend",segmentResultHtml(result));state.textResultSource=text}
  catch(error){if($("text-input").value===text)showError(target,error)}
  finally{state.textBusy=false;if($("text-input").value!==text)inputUpdated(target);updateText();updateProfileControl()}
}
function updateEdit(){if(state.editResultSource!==null&&state.editResultSource!==$("edit-input").value){state.editResultSource=null;inputUpdated($("edit-result"))}const length=Array.from($("edit-input").value).length,valid=Array.from($("edit-input").value.trim()).length>=20&&length<=10000;$("edit-count").textContent=`${length} / 10,000 字符`;$("run-edit").disabled=state.editBusy||!valid;$("run-llm-edit").disabled=state.editBusy||!valid}
function renderEdit(target,data){const edited=data.edited_text??data.revised_text??data.text??"";const suggestions=evidenceList(data.suggestions||data.changes);const checks=Object.entries(data.meaning_check?.checks||{}).map(([name,passed])=>`${name}: ${passed?"通过":"需核对"}`);const applied=data.applied_edits||[];target.innerHTML=`<div class="result-header"><div><small>编辑建议</small><h3>改写预览</h3></div><span class="decision">请作者确认</span></div><div class="edit-output" id="edit-output">${escapeHtml(edited||"未生成改写文本")}</div><div class="result-actions"><button class="secondary-button" id="copy-edit" type="button">复制建议</button><button class="secondary-button" id="use-edit" type="button">放入原文框</button></div><div class="detail-section"><h4>修改与建议</h4><p>自动修改 ${applied.length} 处；人工建议 ${suggestions.length} 条。</p>${suggestions.length?`<ul>${suggestions.map(s=>`<li>${escapeHtml(s)}</li>`).join("")}</ul>`:"<p>未发现预设规则覆盖的问题，请仍逐句核对。</p>"}</div><div class="detail-section"><h4>原意保留检查</h4>${checks.length?`<ul>${checks.map(s=>`<li>${escapeHtml(s)}</li>`).join("")}</ul>`:"<p>数字、姓名、引文和事实仍需作者核对。</p>"}<p>${escapeHtml(data.meaning_check?.limitation||"表面检查不证明语义等价。")}</p></div><div class="result-foot">编辑结果不用于规避检测；确认后再使用。</div>`;const copy=$("copy-edit");copy.onclick=async()=>{try{await navigator.clipboard.writeText(edited);copy.textContent="已复制";copy.title="";setTimeout(()=>{if(copy.isConnected)copy.textContent="复制建议"},2000)}catch{copy.textContent="复制失败，请手动复制";copy.title="请在上方建议文本中选择并复制。"}};$("use-edit").onclick=()=>{$("edit-input").value=edited;updateEdit()}}
async function runEdit(){return runEditing(false)}
async function runLLMEdit(){return runEditing(true)}
async function runEditing(llm){
  const text=$("edit-input").value,target=$("edit-result");
  if(state.editBusy||Array.from(text.trim()).length<20||Array.from(text).length>10000)return;
  state.editBusy=true;state.editResultSource=null;updateEdit();setBusy($(llm?"run-llm-edit":"run-edit"),target,llm?"正在生成大模型预览…":"正在生成建议…");
  try{
    const result=await jsonPost(llm?"/api/text/edit/llm-preview":"/api/text/edit",{text});
    if($("edit-input").value!==text)return;
    state.editResultSource=text;
    if(!llm){renderEdit(target,result);return}
    const accepted=result.status==="review_candidate";
    target.innerHTML=`<div class="result-header"><div><small>大模型编辑预览</small><h3>${accepted?"待作者核对":"未提供可用改写"}</h3></div><span class="decision">${accepted?"人工确认":"已拦截"}</span></div>${accepted?`<div class="edit-output">${escapeHtml(result.candidate)}</div>`:""}<div class="detail-section"><h4>检查结果</h4><p>${escapeHtml(previewReason(result))}</p><p>请逐句核对原意和事实，确认后使用。</p></div>`;
  }catch(error){if($("edit-input").value===text)showError(target,error)}
  finally{state.editBusy=false;if($("edit-input").value!==text)inputUpdated(target);updateEdit()}
}
async function showModelInfo(){
  const dialog=$("model-dialog"),version=++state.modelInfoVersion;
  if(!dialog.open)dialog.showModal();
  $("model-details").innerHTML='<p class="muted">正在读取模型信息…</p>';
  try{
    const data=await api("/api/models",{signal:AbortSignal.timeout(8000)});
    if(version!==state.modelInfoVersion)return;
    state.models=data;
    const scope=data.scope||{},results=Array.isArray(data.evaluation_summary)?data.evaluation_summary:[];
    const rates=results.map(row=>`<div class="evaluation-summary"><strong>${escapeHtml(row.name||"样本评测")}</strong><p>${escapeHtml(row.samples)} 条样本 · 准确率 ${pct(100*row.accuracy)} · 误报率 ${pct(100*row.false_positive_rate)} · 漏报率 ${pct(100*row.false_negative_rate)}</p></div>`).join("");
    $("model-details").innerHTML=`<p>${data.available?(data.loaded?"检测组件已初始化":"模型文件齐备，首次检测时按需加载"):"模型暂不可用"}${data.reason?`：${escapeHtml(data.reason)}`:""}</p><h3>图像检测</h3><p>${escapeHtml(scope.image||"")}</p><h3>文本检测</h3><p>${escapeHtml(scope.text||"")}</p>${rates?`<h3>近期标注样本实测</h3>${rates}<p>表内按 0.5 阈值统计；检测页保留 0.2–0.8 的复核区间。完整记录保存在项目评测目录。</p>`:""}<h3>可信编辑</h3><p>${escapeHtml(scope.editing||"")}</p><details><summary>模型版本与参数</summary><p class="model-manifest">${escapeHtml(JSON.stringify(data.manifest||{},null,2))}</p></details>`;
  }catch(error){if(version===state.modelInfoVersion)$("model-details").innerHTML=`<p class="muted">模型信息暂不可用：${escapeHtml(error.message)}</p>`}
}
async function checkHealth(){
  if(state.healthBusy)return;
  clearTimeout(state.healthTimer);state.healthBusy=true;
  try{
    const data=await api("/api/runtime",{signal:AbortSignal.timeout(4000)}),tasks=data.processing;
    $("service-dot").className="status-dot ready";
    $("service-label").textContent=tasks.active?`处理中 ${tasks.active} / ${tasks.limit}`:data.agent?.active?"助手处理中":"服务就绪";
    $("service-label").title=`已完成 ${tasks.completed} 项请求 · 暂存报告 ${data.reports.count} / ${data.reports.limit} 份`;
  }catch{$("service-dot").className="status-dot error";$("service-label").textContent="服务未连接";$("service-label").title="请检查本地服务是否已启动"}
  finally{state.healthBusy=false;if(!document.hidden)state.healthTimer=setTimeout(checkHealth,3000)}
}
document.addEventListener("visibilitychange",()=>{clearTimeout(state.healthTimer);if(!document.hidden)checkHealth()});
document.addEventListener("DOMContentLoaded",()=>{
  icon();
  document.querySelectorAll(".nav-item").forEach(el=>el.onclick=()=>setView(el.dataset.view));
  $("image-input").onchange=event=>setImage(event.target.files[0]);
  $("clear-image").onclick=clearImage;
  const drop=$("image-dropzone");
  ["dragenter","dragover"].forEach(name=>drop.addEventListener(name,event=>{event.preventDefault();drop.classList.add("dragging")}));
  ["dragleave","drop"].forEach(name=>drop.addEventListener(name,event=>{event.preventDefault();drop.classList.remove("dragging")}));
  drop.addEventListener("drop",event=>setImage(event.dataTransfer.files[0]));
  document.querySelectorAll(".sample-button").forEach(el=>el.onclick=()=>useSample(el.dataset.sample));
  $("detect-image").onclick=detectImage;
  $("text-input").oninput=updateText;$("detect-text").onclick=detectText;
  $("edit-input").oninput=updateEdit;$("run-edit").onclick=runEdit;
  $("model-info-button").onclick=showModelInfo;
  $("model-profile").onchange=()=>{state.textResultSource=null;inputUpdated($("text-result"));if(state.imageReady)inputUpdated($("image-result"))};
  $("close-dialog").onclick=()=>$("model-dialog").close();
  $("dialog-done").onclick=()=>$("model-dialog").close();
  checkHealth();
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

function openLinkedView(){const view=location.hash.slice(1);if(["image","text","document","edit","course","agent"].includes(view))setView(view)}
document.addEventListener("DOMContentLoaded",openLinkedView);
window.addEventListener("hashchange",openLinkedView);

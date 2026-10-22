'use strict';
const $ = (s, e=document) => e.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const state = {user:null, projects:[], project:null, sample:null, job:null, view:'home', dirty:false, cap:null};
const statusName = {queued:'等待分析',running:'正在分析',completed:'分析完成',failed:'未完成',cancelled:'已取消'};
const actionName = {accept:'接受预测',correct:'修正几何',reject:'拒绝预测',no_defect:'无缺陷',artifact:'图像伪影',unknown:'无法判断'};
let canvasState = null, pollTimer = null, toastTimer, evolutionPlayer = null;
let evolutionAutoplayUsed = false;
const date = t => new Date(t*1000).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'});
function toast(text){$('#toast').textContent=text;$('#toast').style.display='block';clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').style.display='none',5000);}
async function api(path, options={}){
  const headers = {...options.headers};
  if(options.method && options.method!=='GET') headers['X-CSRF-Token']=state.user?.csrf || '';
  if(options.body && typeof options.body==='object' && !(options.body instanceof Blob)){headers['Content-Type']='application/json';options.body=JSON.stringify(options.body);}
  const r=await fetch('/api'+path,{...options,headers,credentials:'same-origin'});
  let data;try{data=await r.json();}catch{throw new Error('服务器暂时未响应，请稍后重试。已保存记录不会丢失。');}
  if(!r.ok){if(r.status===401 && path!=='/login'){state.user=null;loginView();}throw new Error(typeof data.detail==='string'?data.detail:'请检查输入内容');}
  return data;
}
function guarded(fn){return async e=>{try{await fn(e);}catch(err){toast(err.message);}};}
function leave(){if(state.dirty && !confirm('当前标注尚未保存，确定离开吗？'))return false;state.dirty=false;return true;}
function modal(html){const d=$('#modal');d.innerHTML=html;d.showModal();d.querySelectorAll('[data-close]').forEach(b=>b.onclick=()=>d.close());return d;}
function closeModal(){$('#modal').close();}
function markDirty(){state.dirty=true;const el=$('#save-state');if(el){el.className='muted';el.textContent='有未保存的修改';}}
const brand = '<div class="brand"><span class="brandmark">✧</span>PhyEvo-Mic</div>';

// Curated historical milestones, not a live research job or per-user activity feed.
// Evidence mapping lives in docs/platform_zh.md; no experiment assets are fetched here.
const evolutionStages = [
 {name:'物理起点', en:'PHYSICS', title:'先让异常，有物理依据。', body:'从晶格、频谱、相位与对称关系出发，定位点、线和区域异常，形成可检查的物理证据。', kept:'物理分析与证据协议', action:'观察结构 → 定位异常', x:55,y:145},
 {name:'专家反馈', en:'EXPERT FEEDBACK', title:'把专家判断，留成可用经验。', body:'将难例、几何修正与审核结论关联保存，让研究者的判断成为可复用的监督与研究线索。', kept:'难例与版本化审核', action:'发现分歧 → 保存修正', x:175,y:45},
 {name:'学生协同', en:'STUDENT LEARNING', title:'学习新的场景，保留物理优势。', body:'以专家监督训练学生，比较物理、学生及组合表现，研究不同能力适合被调用的条件。', kept:'学生模型与协同经验', action:'专家监督 → 能力互补', x:425,y:45},
 {name:'自主研究', en:'PROGRAM EVOLUTION', title:'从研究提案，走向可执行改进。', body:'根据诊断提出具体程序，在隔离环境中执行，并用固定协议评价；结果反馈进入下一轮研究。', kept:'可执行提案与实验反馈', action:'提出机制 → 执行 → 评价', x:545,y:145},
 {name:'能力积累', en:'CAPABILITY BANK', title:'新能力加入，旧能力留下。', body:'将新分析分支和采用规则一起评价。通过的更新纳入能力库，未通过的尝试不替换已有系统。', kept:'分析分支与采用规则', action:'整体评价 → 接受或拒绝', x:425,y:245},
 {name:'状态继承', en:'STATE INHERITANCE', title:'让已验证的能力，参与下一轮。', body:'后续程序实际调用已接纳的父节点，继承算法与数值状态；把一次有效改进沉淀成可继续使用的能力。', kept:'父状态与研究经验', action:'继承状态 → 再次调用', x:175,y:245},
];

function evolutionMarkup(expanded=false){
 const edges=evolutionStages.map((s,i)=>{const n=evolutionStages[(i+1)%6];return `<path data-orbit="${i}" d="M${s.x} ${s.y} L${n.x} ${n.y}"/>`;}).join('');
 const spokes=evolutionStages.map((s,i)=>`<path class="evo-spoke" data-spoke="${i}" d="M300 145 Q${(s.x+300)/2} ${145+(s.y-145)*.2} ${s.x} ${s.y}"/>`).join('');
 const nodes=evolutionStages.map((s,i)=>`<g class="evo-node" data-node="${i}" transform="translate(${s.x} ${s.y})"><circle class="evo-halo" r="27"/><circle class="evo-node-ring" r="18"/><circle class="evo-dot" r="4"/><text y="40" text-anchor="middle">${s.name}</text></g>`).join('');
 return `<section class="evolution ${expanded?'evolution-expanded':''}" data-playing="false" data-stage="0" aria-label="PhyEvo-Mic 历史进化路线回放">
   <div class="evo-top"><span class="evo-eyebrow">PhyEvo-Mic · RECURSIVE SELF-IMPROVEMENT</span><span class="evo-history-label"><span></span>历史演进回放</span></div>
   <div class="evo-main"><div class="evo-intro"><h1>${expanded?'每一轮探索，<br>留下下一轮的能力。':'从物理证据出发，<br>让分析能力逐步进化。'}</h1><p>研究交互轨迹与专家反馈汇成经验，<br class="desktop-break">在研究批次中探索、验证、继承。</p><div class="evo-objects"><span>分析方法</span><span>学生模型</span><span>采用规则</span><span>研究经验</span></div>${expanded?'<div class="evo-history-note">六个研发节点 · 一条能力积累路线</div>':'<div class="evo-actions"><button id="new-project" class="evo-primary">＋ 创建研究项目</button><button id="explore-evolution" class="evo-secondary">展开进化路线 ↗</button></div>'}</div>
   <div class="evo-visual" aria-hidden="true"><div class="evo-visual-caption"><span>EVOLUTION MAP</span><span class="evo-map-index">01 / 06</span></div><svg viewBox="0 0 600 300" class="evo-network"><defs><radialGradient id="evo-core-light"><stop stop-color="#50d8bc" stop-opacity=".25"/><stop offset="1" stop-color="#50d8bc" stop-opacity="0"/></radialGradient><linearGradient id="evo-edge" x2="1" y2="1"><stop stop-color="#50dab9"/><stop offset="1" stop-color="#8f9fff"/></linearGradient></defs><circle cx="300" cy="145" r="130" fill="url(#evo-core-light)"/><g class="evo-grid"><path d="M120 145H480 M300 16V277 M190 45L410 245 M190 245L410 45"/><ellipse cx="300" cy="145" rx="116" ry="52"/><ellipse cx="300" cy="145" rx="70" ry="108"/></g><g class="evo-orbit">${edges}</g><g class="evo-spokes">${spokes}</g><g class="evo-core"><path d="M300 91L347 118V172L300 199L253 172V118Z"/><path class="evo-core-inner" d="M300 103L337 124V166L300 187L263 166V124Z"/><text x="300" y="145" text-anchor="middle">RSI</text><text class="evo-core-caption" x="300" y="165" text-anchor="middle">PhyEvo-Mic</text></g><g class="evo-particles"><circle cx="222" cy="105" r="2"/><circle cx="380" cy="184" r="2"/><circle cx="300" cy="63" r="2"/><circle cx="300" cy="229" r="2"/></g>${nodes}</svg><div class="evo-visual-bottom"><span>OBSERVE</span><span>EXPLORE</span><span>INHERIT</span></div></div></div>
   <div class="evo-story"><div class="evo-stage-number">01</div><div class="evo-story-copy"><div class="evo-stage-en">PHYSICS</div><h2 class="evo-stage-title">${evolutionStages[0].title}</h2><p class="evo-stage-body">${evolutionStages[0].body}</p></div><div class="evo-retained"><span>这一阶段沉淀</span><strong class="evo-stage-kept">${evolutionStages[0].kept}</strong><small class="evo-stage-action">${evolutionStages[0].action}</small></div></div>
   <div class="evo-timeline" role="group" aria-label="选择历史研发阶段">${evolutionStages.map((s,i)=>`<button data-evo-stage="${i}" class="evo-step ${i===0?'selected':''}" aria-pressed="${i===0}"><span>0${i+1}</span><strong>${s.name}</strong><i><b></b></i></button>`).join('')}</div>
   <div class="evo-controls"><div class="evo-buttons"><button data-evo-toggle aria-label="播放进化路线">▶ 播放</button><button data-evo-replay aria-label="从头重播进化路线">↻ 重播</button><span class="evo-play-status">已暂停</span></div><span class="evo-footnote">已有研发记录的视觉回放 · 不触发当前项目升级</span></div>
 </section>`;
}

function mountEvolution(){
 evolutionPlayer?.dispose();
 const root=$('.evolution');if(!root)return;
 const media=matchMedia('(prefers-reduced-motion: reduce)');
 const duration=5200, total=duration*evolutionStages.length;
 let elapsed=0, last=0, raf=0, playing=false, disposed=false, active=-1;
 const render=()=>{
   const index=Math.min(evolutionStages.length-1,Math.floor(elapsed/duration));
   root.dataset.playing=String(playing);root.dataset.stage=String(index);
   root.dataset.progress=(elapsed/total).toFixed(4);
   const toggle=$('[data-evo-toggle]',root);toggle.textContent=playing?'Ⅱ 暂停':'▶ 播放';toggle.setAttribute('aria-label',playing?'暂停进化路线':'播放进化路线');
   $('.evo-play-status',root).textContent=elapsed>=total?'回放结束':playing?'回放中':'已暂停';
   root.querySelectorAll('[data-evo-stage]').forEach((button,i)=>{
     button.classList.toggle('selected',i===index);button.classList.toggle('visited',i<index);
     button.setAttribute('aria-pressed',String(i===index));
     $('b',button).style.width=`${Math.min(1,Math.max(0,(elapsed-i*duration)/duration))*100}%`;
   });
   if(active===index)return;active=index;const s=evolutionStages[index];
   $('.evo-stage-number',root).textContent=String(index+1).padStart(2,'0');
   $('.evo-map-index',root).textContent=`0${index+1} / 06`;
   for(const [cls,value] of [['en',s.en],['title',s.title],['body',s.body],['kept',s.kept],['action',s.action]])$('.evo-stage-'+cls,root).textContent=value;
   root.querySelectorAll('[data-node]').forEach((el,i)=>{el.classList.toggle('active',i===index);el.classList.toggle('retained',i<index);});
   root.querySelectorAll('[data-spoke]').forEach((el,i)=>el.classList.toggle('active',i===index));
   root.querySelectorAll('[data-orbit]').forEach((el,i)=>el.classList.toggle('visited',i<index));
 };
 const frame=now=>{
   if(disposed||!playing)return;
   if(document.hidden||!root.isConnected){pause();return;}
   // A queued frame timestamp can precede the click's performance.now().
   elapsed=Math.min(total,elapsed+Math.max(0,Math.min(100,now-last)));last=now;
   if(elapsed>=total)playing=false;
   render();if(playing)raf=requestAnimationFrame(frame);
 };
 function pause(){playing=false;cancelAnimationFrame(raf);render();}
 function play(restart=false){if(restart||elapsed>=total)elapsed=0;playing=true;last=performance.now();cancelAnimationFrame(raf);render();raf=requestAnimationFrame(frame);}
 $('[data-evo-toggle]',root).onclick=()=>playing?pause():play();
 $('[data-evo-replay]',root).onclick=()=>{if(media.matches){elapsed=0;pause();}else play(true);};
 root.querySelectorAll('[data-evo-stage]').forEach((b,i)=>b.onclick=()=>{elapsed=i*duration;pause();});
 const visibility=()=>{if(document.hidden)pause();};
 const reduced=()=>{if(media.matches)pause();};
 document.addEventListener('visibilitychange',visibility);media.addEventListener('change',reduced);
 const observer=new IntersectionObserver(entries=>{if(!entries[0].isIntersecting&&playing)pause();},{threshold:0});observer.observe(root);
 evolutionPlayer={dispose(){disposed=true;cancelAnimationFrame(raf);observer.disconnect();document.removeEventListener('visibilitychange',visibility);media.removeEventListener('change',reduced);root.dataset.playing='false';}};
 render();
 if(!media.matches&&!evolutionAutoplayUsed){evolutionAutoplayUsed=true;play();}
}

function loginView(){
 evolutionPlayer?.dispose();evolutionPlayer=null;
 clearInterval(pollTimer);state.dirty=false;canvasState=null;
 $('#app').innerHTML=`<main class="login"><section class="login-story">${brand}<div><div class="eyebrow">MICROSCOPY · EVIDENCE · EVOLUTION</div><h1>让每一次观察，<br>成为下一次发现。</h1><p>从显微图像到物理证据，连接研究者、专家与持续改进的分析框架。</p><div class="lattice" aria-hidden="true"></div></div><div class="login-footer muted">您的图像属于您。分享由您决定。</div></section><section class="login-box"><form class="login-form" id="login-form"><div class="eyebrow">PRIVATE RESEARCH WORKSPACE</div><h2>进入研究工作台</h2><p class="muted">使用邀请人提供的个人激活码登录。<br>无需注册，也不需要额外安装软件。</p><label for="code">个人激活码</label><input id="code" type="password" autocomplete="off" placeholder="PEM.…" required maxlength="256"><label class="muted"><input id="show-code" type="checkbox"> 显示激活码</label><button class="primary" type="submit">进入我的工作空间 →</button><div class="error" id="login-error" role="alert"></div><div class="privacy"><p>◉ 每个账号拥有独立、私有的项目空间</p><p>◉ 只有您明确邀请的专家才能查看共享项目</p><p>◉ 刷新或退出登录，不会丢失已保存的记录</p></div></form></section></main>`;
 $('#show-code').onchange=e=>$('#code').type=e.target.checked?'text':'password';
 $('#login-form').onsubmit=async e=>{e.preventDefault();const b=$('button.primary');b.disabled=true;$('#login-error').textContent='';try{state.user=await api('/login',{method:'POST',body:{code:$('#code').value.trim()}});$('#code').value='';await boot();}catch(err){$('#login-error').textContent=err.message;}finally{b.disabled=false;}};
}
function shell(view){
 evolutionPlayer?.dispose();evolutionPlayer=null;
 clearInterval(pollTimer);canvasState?.observer?.disconnect();canvasState=null;state.view=view;
 const nav=[['home','▦','我的项目'],['review','◎','专家审核'],['rsi','↗','RSI 进化路线'],['version','◇','框架与版本']];if(state.user.role==='admin')nav.push(['accounts','♙','账号管理']);
 $('#app').innerHTML=`<div class="shell"><aside class="sidebar">${brand}<nav class="nav" aria-label="主导航">${nav.map(([id,i,n])=>`<button data-nav="${id}" class="${view===id?'active':''}"><span class="nav-icon">${i}</span>${n}</button>`).join('')}</nav><div class="side-foot"><span class="badge good">● 独立研究空间</span><p class="muted">项目默认私有。<br>共享前，专家不可见。</p></div></aside><main class="main"><header class="topbar"><div class="muted">研究工作台 <span> / ${esc(nav.find(n=>n[0]===view)?.[2] || '图像分析')}</span></div><div class="row"><span class="avatar">${esc(state.user.name.slice(0,1))}</span><span class="user-name tiny">${esc(state.user.name)}</span><button class="text tiny" id="logout">退出</button></div></header><div class="content" id="content"></div></main></div>`;
 document.querySelectorAll('[data-nav]').forEach(b=>b.onclick=guarded(async()=>{if(!leave())return;location.hash='';if(b.dataset.nav==='home'||b.dataset.nav==='review')await home(b.dataset.nav);else if(b.dataset.nav==='rsi')await rsiView();else if(b.dataset.nav==='accounts')await accountView();else versionView();}));
 $('#logout').onclick=guarded(async()=>{if(!leave())return;await api('/logout',{method:'POST'});state.user=null;location.hash='';loginView();});
}
async function boot(){state.cap=await api('/capabilities');state.projects=await api('/projects');const match=location.hash.match(/^#project=([0-9a-f]{32})(?:&sample=([0-9a-f]{32}))?$/);if(match&&state.projects.some(p=>p.id===match[1]))await openProject(match[1],match[2]);else await home();}
async function home(view='home'){
 state.projects=await api('/projects');shell(view);const shared=view==='review';const list=state.projects.filter(p=>shared?p.owner_id!==state.user.id:p.owner_id===state.user.id);
 $('#content').innerHTML=`<section class="welcome"><div><div class="eyebrow">${shared?'EXPERT COLLABORATION':'YOUR RESEARCH, CONNECTED'}</div><h1>${shared?'每一次复核，都让证据更清晰。':'从一张原图，开始新的发现。'}</h1><p class="muted">${shared?'这里只显示已明确授权给您的项目。审核结论按您的独立身份保存。':'上传原图、查看异常候选、保存判断。所有工作在您的独立项目中完成。'}</p><div class="workflow"><span>01 原始图像</span>→<span>02 物理证据</span>→<span>03 专家反馈</span>→<span>04 改进研究</span></div></div>${!shared?'<div><button class="primary" id="new-project">＋ 创建项目</button></div>':''}</section><div class="section-title"><h2>${shared?'邀请我审核的项目':'我的项目'}</h2><span class="muted">${list.length} 个项目 · ${shared?'明确授权访问':'默认仅自己可见'}</span></div>${list.length?`<div class="projects">${list.map(p=>`<button class="project-card" data-project="${p.id}"><div class="row between" style="width:100%"><span class="project-icon">▧</span><span class="badge">${shared?'受邀审核':'私有项目'}</span></div><h3>${esc(p.name)}</h3><span class="muted">${shared?'来自 '+esc(p.owner_name)+' · ':''}${date(p.created)}</span><span class="link">打开工作台 →</span></button>`).join('')}</div>`:`<div class="empty"><h3>${shared?'尚未收到项目邀请':'您的工作空间已准备好'}</h3><p>${shared?'项目所有者需在「邀请专家」中选择您的账号。没有邀请时，您不能查看他人的图像。':'点击「创建项目」，上传第一张灰度显微图像。'}</p></div>`}`;
 $('#new-project')?.addEventListener('click',createProjectDialog);document.querySelectorAll('[data-project]').forEach(b=>b.onclick=guarded(()=>openProject(b.dataset.project)));
 if(!shared){
   $('.welcome').outerHTML=evolutionMarkup(false);
   $('#new-project').onclick=createProjectDialog;
   $('#explore-evolution').onclick=guarded(()=>rsiView());
   mountEvolution();
 }
}
function createProjectDialog(){const d=modal(`<h2>创建私有项目</h2><p class="muted">先给这组图像一个名字。之后可以选择邀请专家。</p><form id="create-project"><label for="project-name">项目名称</label><input id="project-name" maxlength="100" placeholder="例如：样品 A · 界面观察" required><div class="actions"><button type="button" data-close>取消</button><button class="primary">创建并打开</button></div></form>`);$('#create-project').onsubmit=guarded(async e=>{e.preventDefault();const p=await api('/projects',{method:'POST',body:{name:$('#project-name').value}});d.close();await openProject(p.id);});$('#project-name').focus();}

async function openProject(pid,sid){
 state.project=await api('/projects/'+pid);state.sample=state.project.samples.find(s=>s.id===sid)||state.project.samples[0]||null;
 location.hash=`project=${pid}${state.sample?'&sample='+state.sample.id:''}`;shell('home');
 const own=state.project.owner_id===state.user.id;
 $('#content').innerHTML=`<div class="work-head row between"><div><div class="eyebrow">${own?'PRIVATE PROJECT':'SHARED FOR REVIEW'}</div><h1>${esc(state.project.name)}</h1></div><div class="row">${own?'<button id="share">邀请专家</button><button id="export">导出审核记录</button>':''}<button id="back">← 项目</button></div></div><div class="workbench"><aside class="panel sample-panel"><h3>项目图像 <span class="muted">${state.project.samples.length}</span></h3>${state.project.samples.map(s=>`<button data-sample="${s.id}" class="sample ${s.id===state.sample?.id?'active':''}"><strong>${esc(s.name)}</strong><span class="muted">${s.width} × ${s.height} px · ${s.jobs.length?esc(statusName[s.jobs[0].status]):'待分析'}</span></button>`).join('')}${own?'<button class="primary" id="upload">＋ 上传图像</button>':''}</aside><div id="viewer-area" class="grow"></div><aside class="panel review-panel" id="review-area"></aside></div>`;
 $('#back').onclick=guarded(()=>leave()&&home());$('#share')?.addEventListener('click',guarded(shareDialog));$('#export')?.addEventListener('click',()=>download('/api/projects/'+pid+'/export','审核记录.json'));
 $('#upload')?.addEventListener('click',uploadDialog);document.querySelectorAll('[data-sample]').forEach(b=>b.onclick=guarded(()=>leave()&&openProject(pid,b.dataset.sample)));
 if(!state.sample){$('#viewer-area').innerHTML='<div class="empty"><h3>等待第一张图像</h3><p>支持二维灰度 PNG、TIFF、JPEG。<br>保留原始像素和文件，显示对比度调整不改变分析输入。</p></div>';$('#review-area').innerHTML='<h3>如何开始？</h3><div class="steps"><div class="step"><div><h3>上传原图</h3><p>请勿上传已画圈或截图后的彩色图。</p></div></div><div class="step"><div><h3>选择任务</h3><p>根据研究问题选择点缺陷、界面等。</p></div></div><div class="step"><div><h3>检查与保存</h3><p>预测是候选，您的判断会单独保存。</p></div></div></div>';return;}
 state.job=state.sample.jobs[0]?await api('/jobs/'+state.sample.jobs[0].id):null;
 await renderViewer();renderReview();if(state.job&&['queued','running'].includes(state.job.status))pollJob();
}
function uploadDialog(){const d=modal(`<h2>上传原始图像</h2><p class="muted">二维灰度图 · 单张 ≤ 20 MiB · 最多 419 万像素。多层 TIFF 请先选择并导出单层。</p><form id="upload-form"><div class="dropzone"><label for="file">选择图像文件</label><input id="file" type="file" accept=".png,.tif,.tiff,.jpg,.jpeg" required></div><p class="muted">原图仅用于您的项目，不会自动加入训练数据或公开数据集。</p><div class="actions"><button type="button" data-close>取消</button><button class="primary" id="upload-submit">上传到私有项目</button></div><div id="upload-error" class="error"></div></form>`);$('#upload-form').onsubmit=async e=>{e.preventDefault();const f=$('#file').files[0];if(!f)return;const b=$('#upload-submit');b.disabled=true;b.textContent='正在上传…';try{const s=await api('/projects/'+state.project.id+'/samples?name='+encodeURIComponent(f.name),{method:'POST',body:f,headers:{'Content-Type':'application/octet-stream'}});d.close();await openProject(state.project.id,s.id);toast('原图已保存，请选择任务开始分析');}catch(err){$('#upload-error').textContent=err.message;b.disabled=false;b.textContent='重试上传';}};}
async function shareDialog(){const experts=await api('/experts');const shares=state.project.shares;const d=modal(`<h2>明确共享给专家</h2><p class="muted">授权后，所选专家可以查看本项目的全部图像、结果及审核历史，并提交审核；不能替您上传、运行分析或再次共享。撤销后立即停止新访问，已下载的副本无法远程收回。</p>${experts.length?experts.map(e=>`<div class="account-row"><div><strong>${esc(e.name)}</strong><div class="muted">专家账号 · ${e.id.slice(0,8)}</div></div><button data-expert="${e.id}" data-active="${shares.some(s=>s.expert_id===e.id&&s.active)?'0':'1'}">${shares.some(s=>s.expert_id===e.id&&s.active)?'撤销授权':'授权审核'}</button></div>`).join(''):'<div class="notice">目前没有专家账号。请管理员在「账号管理」中创建专家激活码。</div>'}<div class="actions"><button data-close>完成</button></div>`);d.querySelectorAll('[data-expert]').forEach(b=>b.onclick=guarded(async()=>{await api('/projects/'+state.project.id+'/share',{method:'POST',body:{expert_id:b.dataset.expert,active:b.dataset.active==='1'}});state.project=await api('/projects/'+state.project.id);d.close();await shareDialog();}));}
function loadImage(url){return new Promise((resolve,reject)=>{const i=new Image();i.onload=()=>resolve(i);i.onerror=()=>reject(new Error('图像加载失败，可能是会话过期或共享已撤销，请刷新页面'));i.src=url;});}
async function renderViewer(){
 const s=state.sample, j=state.job;
 $('#viewer-area').innerHTML=`<section class="panel viewer"><div class="viewer-head row between"><strong class="tiny">${esc(s.name)}</strong><span class="muted">${s.width} × ${s.height} px · 原始坐标</span></div><div class="toolbar"><button id="fit">适合窗口</button><button id="one">1:1</button><button id="minus" aria-label="缩小">−</button><span id="zoom" class="tiny">100%</span><button id="plus" aria-label="放大">＋</button><button id="full">全屏</button><span class="divider"></span><button data-tool="pan" class="active">拖动</button><button data-tool="point">点</button><button data-tool="line">线</button><button data-tool="polygon">区域</button><button id="finish">完成区域</button><button id="undo">撤销</button></div><div class="canvas-wrap" id="canvas-wrap"><canvas id="canvas" aria-label="显微图像，可拖动、缩放并添加审核几何"></canvas><span class="canvas-hint">滚轮缩放 · 拖动看图 · 按住 B 暂看原图</span></div><div class="layers"><label><input id="prediction" type="checkbox" ${j?.status==='completed'?'':'disabled'}><span class="legend-dot amber"></span>预测轮廓</label><label><input id="heatmap" type="checkbox" ${j?.status==='completed'?'':'disabled'}>分数图</label><label><input id="annotations" type="checkbox" checked><span class="legend-dot teal"></span>我的标注</label></div><div class="viewer-foot">默认先看原图。琥珀色是模型候选，青绿色是您的修正；缩放不改变保存坐标。显示对比度经过拉伸，不改变分析输入。</div></section><section class="panel jobs"><div class="row between"><h3>分析任务</h3><span class="badge ${j?.status==='completed'?'good':''}" id="job-status">${j?statusName[j.status]:'尚未分析'}</span></div>${state.project.owner_id===state.user.id?`<div class="row"><select id="task" aria-label="分析任务" class="grow">${Object.entries(state.cap.tasks).map(([k,v])=>`<option value="${k}" ${j?.task===k?'selected':''}>${v}</option>`).join('')}</select><button id="run" class="primary" ${j&&['queued','running'].includes(j.status)?'disabled':''}>开始分析</button></div>`:''}<p id="job-message" class="muted" style="margin:12px 0 0">${j?.error?esc(j.error):j?.status==='completed'?'结果已保存。模型候选需要结合原图判断，不是元素身份鉴定。':'任务提交后在后台运行；退出页面不会取消任务。'}</p>${j&&['queued','running'].includes(j.status)&&state.project.owner_id===state.user.id?'<button class="text" id="cancel">取消本次分析</button>':''}${j?.status==='completed'?'<button class="text link" id="report">下载本次分析 JSON 报告 →</button>':''}${s.jobs.length>1?`<details><summary class="muted">历史分析 (${s.jobs.length})</summary>${s.jobs.map(j=>`<button class="text tiny" data-job="${j.id}">${date(j.created)} · ${esc(state.cap.tasks[j.task])} · ${statusName[j.status]}</button>`).join('')}</details>`:''}</section>`;
 const image=await loadImage('/api/samples/'+s.id+'/preview');const outline=j?.status==='completed'?await loadImage('/api/jobs/'+j.id+'/artifacts/outline.png'):null;const heat=j?.status==='completed'?await loadImage('/api/jobs/'+j.id+'/artifacts/heatmap.png'):null;
 const last=j?.feedback.filter(f=>f.user_id===state.user.id).at(-1);
 canvasState={image,outline,heat,geometry:structuredClone(last?.geometry||[]),history:j?.feedback||[],pending:[],tool:'pan',scale:1,x:0,y:0,pointer:null,seen:!!last?.prediction_seen,raw:false};
 if(j?.feedback.some(f=>f.user_id!==state.user.id)){
   const latest=Object.values(j.feedback.filter(f=>f.user_id!==state.user.id).reduce((acc,f)=>{acc[f.user_id]=f;return acc;},{}));
   const label=document.createElement('label');label.innerHTML=`查看修正 <select id="annotation-source" aria-label="查看谁的几何标注"><option value="">我的标注</option>${latest.map(f=>`<option value="${f.id}">${esc(f.reviewer_name)} · 第 ${f.revision} 版</option>`).join('')}</select>`;$('.layers').appendChild(label);
   $('#annotations').parentElement.lastChild.textContent='几何标注';$('.viewer-foot').textContent='默认先看原图。琥珀色是模型候选，青绿色是当前选中的审核修正；查看他人记录不会修改您的标注。坐标始终对应原图。';
 }
 setupCanvas();
 $('#run')?.addEventListener('click',guarded(async()=>{if(!leave())return;$('#run').disabled=true;try{await api('/samples/'+s.id+'/jobs',{method:'POST',body:{task:$('#task').value}});await openProject(state.project.id,s.id);}catch(e){$('#run').disabled=false;throw e;}}));
 $('#cancel')?.addEventListener('click',guarded(async()=>{await api('/jobs/'+j.id+'/cancel',{method:'POST'});toast('已请求取消；原图保留');}));$('#report')?.addEventListener('click',()=>download('/api/jobs/'+j.id+'/artifacts/summary.json','分析报告.json'));
 document.querySelectorAll('[data-job]').forEach(b=>b.onclick=guarded(async()=>{if(!leave())return;clearInterval(pollTimer);state.job=await api('/jobs/'+b.dataset.job);await renderViewer();renderReview();if(['running','queued'].includes(state.job.status))pollJob();}));
}
function setupCanvas(){
 const c=$('#canvas'), wrap=$('#canvas-wrap'), v=canvasState;
 const dimensions=()=>({w:wrap.clientWidth,h:wrap.clientHeight});
 function fit(){const {w,h}=dimensions();v.scale=Math.min(w/v.image.width,h/v.image.height)*.94;v.x=(w-v.image.width*v.scale)/2;v.y=(h-v.image.height*v.scale)/2;draw();}
 function zoom(f,px=wrap.clientWidth/2,py=wrap.clientHeight/2){const old=v.scale;v.scale=Math.max(.04,Math.min(30,v.scale*f));v.x=px-(px-v.x)*v.scale/old;v.y=py-(py-v.y)*v.scale/old;draw();}
 function draw(){if(canvasState!==v||!c.isConnected)return;const {w,h}=dimensions(),dpr=window.devicePixelRatio||1;c.width=Math.round(w*dpr);c.height=Math.round(h*dpr);const ctx=c.getContext('2d');ctx.scale(dpr,dpr);ctx.fillStyle='#252e32';ctx.fillRect(0,0,w,h);ctx.save();ctx.translate(v.x,v.y);ctx.scale(v.scale,v.scale);ctx.imageSmoothingEnabled=v.scale<2;ctx.drawImage(v.image,0,0);if(!v.raw){if($('#heatmap').checked&&v.heat){ctx.globalAlpha=.42;ctx.drawImage(v.heat,0,0);ctx.globalAlpha=1;}if($('#prediction').checked&&v.outline)ctx.drawImage(v.outline,0,0);if($('#annotations').checked){ctx.lineWidth=1.7/v.scale;ctx.strokeStyle='#65d6b9';ctx.fillStyle='#65d6b922';const selected=$('#annotation-source')?.value,geometry=selected?v.history.find(f=>f.id===selected)?.geometry||[]:v.geometry;for(const g of geometry)drawGeometry(ctx,g,v.scale);if(!selected&&v.pending.length){ctx.setLineDash([4/v.scale,4/v.scale]);drawGeometry(ctx,{type:v.tool,points:v.pending},v.scale);}}}ctx.restore();$('#zoom').textContent=Math.round(v.scale*100)+'%';}
 v.draw=draw;v.observer=new ResizeObserver(()=>{if(c.isConnected&&canvasState===v)draw();else v.observer.disconnect();});v.observer.observe(wrap);
 $('#fit').onclick=fit;$('#one').onclick=()=>zoom(1/v.scale);$('#minus').onclick=()=>zoom(.8);$('#plus').onclick=()=>zoom(1.25);$('#full').onclick=()=>document.fullscreenElement?document.exitFullscreen():wrap.requestFullscreen();
 for(const id of ['prediction','heatmap','annotations'])$('#'+id).onchange=()=>{if($('#prediction').checked)v.seen=true;draw();};
 if($('#annotation-source'))$('#annotation-source').onchange=()=>{v.tool='pan';v.pending=[];document.querySelectorAll('[data-tool]').forEach(x=>x.classList.toggle('active',x.dataset.tool==='pan'));wrap.classList.remove('draw');draw();};
 document.querySelectorAll('[data-tool]').forEach(b=>b.onclick=()=>{if(b.dataset.tool!=='pan'&&state.job?.status!=='completed'){toast('请先完成分析，再提交带坐标的反馈');return;}v.tool=b.dataset.tool;v.pending=[];document.querySelectorAll('[data-tool]').forEach(x=>x.classList.toggle('active',x===b));wrap.classList.toggle('draw',v.tool!=='pan');if(v.tool!=='pan'&&$('#decision')){if($('#annotation-source'))$('#annotation-source').value='';$('#decision').value='correct';markDirty();}draw();});
 function finish(){if(v.tool==='polygon'&&v.pending.length>=3){v.geometry.push({type:'polygon',points:v.pending});v.pending=[];markDirty();draw();}}
 $('#finish').onclick=finish;$('#undo').onclick=()=>{if(v.pending.length)v.pending.pop();else v.geometry.pop();markDirty();draw();};
 c.onwheel=e=>{e.preventDefault();const r=c.getBoundingClientRect();zoom(Math.exp(-e.deltaY*.0015),e.clientX-r.left,e.clientY-r.top);};
 c.onpointerdown=e=>{const r=c.getBoundingClientRect();v.pointer={id:e.pointerId,x:e.clientX,y:e.clientY,startx:e.clientX,starty:e.clientY};c.setPointerCapture(e.pointerId);if(v.tool!=='pan'){const p=[(e.clientX-r.left-v.x)/v.scale,(e.clientY-r.top-v.y)/v.scale];if(p[0]>=0&&p[1]>=0&&p[0]<v.image.width&&p[1]<v.image.height){if(v.tool==='point'){v.geometry.push({type:'point',points:[p]});markDirty();}else{v.pending.push(p);if(v.tool==='line'&&v.pending.length===2){v.geometry.push({type:'line',points:v.pending});v.pending=[];}markDirty();}draw();}}};
 c.onpointermove=e=>{if(v.pointer?.id===e.pointerId&&v.tool==='pan'){v.x+=e.clientX-v.pointer.x;v.y+=e.clientY-v.pointer.y;v.pointer.x=e.clientX;v.pointer.y=e.clientY;draw();}};c.onpointerup=()=>v.pointer=null;c.onpointercancel=()=>v.pointer=null;fit();
}
function drawGeometry(ctx,g,scale){const p=g.points;if(!p.length)return;ctx.beginPath();if(g.type==='point'){ctx.arc(p[0][0],p[0][1],6/scale,0,2*Math.PI);}else{ctx.moveTo(...p[0]);for(const q of p.slice(1))ctx.lineTo(...q);if(g.type==='polygon'&&p.length>2){ctx.closePath();ctx.fill();}}ctx.stroke();for(const q of p){ctx.beginPath();ctx.arc(q[0],q[1],2/scale,0,2*Math.PI);ctx.stroke();}}
function renderReview(){
 const j=state.job, ready=j?.status==='completed', last=j?.feedback.filter(f=>f.user_id===state.user.id).at(-1);
 $('#review-area').innerHTML=`<div class="eyebrow">REVIEW & FEEDBACK</div><h3>我的判断</h3>${ready?`<div class="metric"><span>模型候选区域</span><strong>${j.summary.detected_region_count ?? '—'}</strong></div><p class="muted">先看原图，再按需打开预测。没有轮廓不等于没有缺陷。</p><label for="decision">审核结论</label><select id="decision">${Object.entries(actionName).map(([k,v])=>`<option value="${k}" ${(last?.action||'unknown')===k?'selected':''}>${v}</option>`).join('')}</select><div id="decision-help" class="muted" style="margin-top:9px"></div><label for="note">备注（可选）</label><textarea id="note" maxlength="3000" placeholder="例如：右上区域需要复核；模型漏掉了这一条界面。">${esc(last?.note||'')}</textarea><button class="primary" id="save">保存我的判断</button><div id="save-state" class="${last?'success':'muted'}" style="margin-top:12px">${last?'已持久化第 '+last.revision+' 版 · '+date(last.created):'尚未保存。请看到保存成功提示后再离开。'}</div><div class="notice" style="margin-top:17px;font-size:11px;padding:10px">标注坐标始终基于原始图像。每次保存形成独立版本，不覆盖其他人的意见。</div><details style="margin-top:20px"><summary class="muted">审核历史 · ${j.feedback.length} 次</summary>${j.feedback.map(f=>`<div class="history-entry"><strong>${esc(f.reviewer_name)} · ${esc(actionName[f.action])}</strong><div class="muted">第 ${f.revision} 版 · ${date(f.created)}</div>${f.note?`<div>${esc(f.note)}</div>`:''}</div>`).join('')||'<p class="muted">暂无记录</p>'}</details>`:'<p class="muted">分析完成后，可以查看物理异常候选，保存您的判断，或使用「点、线、区域」修正位置。</p><div class="notice">已提交的分析在后台运行。无需一直停留在页面。</div>'}`;
 if(!ready)return;
 function help(){const v=$('#decision').value;$('#decision-help').textContent={accept:'请先打开预测轮廓，确认候选位置和范围。',correct:'使用图像上方的点、线或区域工具，画出完整的修正几何。',reject:'模型预测不可靠；此结论不等于图中没有缺陷。',no_defect:'您认为该图没有需要标注的目标缺陷。',artifact:'记录成像、重建或扫描等伪影。',unknown:'现有图像证据不足，暂时无法确定。'}[v];}
 help();$('#decision').onchange=()=>{help();markDirty();};$('#note').oninput=markDirty;
 $('#save').onclick=guarded(async()=>{if(canvasState.pending.length){toast('还有未完成的线或区域，请先完成或撤销');return;}const b=$('#save');b.disabled=true;try{const result=await api('/jobs/'+j.id+'/feedback',{method:'POST',body:{action:$('#decision').value,note:$('#note').value,geometry:$('#decision').value==='correct'?canvasState.geometry:[],prediction_seen:canvasState.seen,request_id:crypto.randomUUID()}});state.dirty=false;state.job=await api('/jobs/'+j.id);if($('#decision').value!=='correct'){canvasState.geometry=[];canvasState.draw();}renderReview();toast('已持久化第 '+result.revision+' 版，刷新或退出后仍会保留');}finally{b.disabled=false;}});
}
function pollJob(){clearInterval(pollTimer);const id=state.job.id;pollTimer=setInterval(async()=>{try{const j=await api('/jobs/'+id);if(!$('#job-status')||state.job?.id!==id)return;$('#job-status').textContent=statusName[j.status];if(['completed','failed','cancelled'].includes(j.status)){clearInterval(pollTimer);state.job=j;await openProject(state.project.id,state.sample.id);toast(statusName[j.status]);}}catch(e){clearInterval(pollTimer);toast(e.message);}},2500);}
async function download(url,name){try{const r=await fetch(url,{credentials:'same-origin'});if(!r.ok)throw new Error('导出失败，请重新登录或检查权限');const b=await r.blob(),u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}catch(e){toast(e.message);}}

async function rsiView(){
 state.projects=await api('/projects');shell('rsi');const own=state.projects.filter(p=>p.owner_id===state.user.id);
 $('#content').innerHTML=`${evolutionMarkup(true)}<div class="section-title"><h2>每一次反馈，连接下一轮研究。</h2></div><div class="grid2"><section class="panel batch-cycle"><div class="eyebrow">ACCUMULATE → RESEARCH → INHERIT</div><h2>积累经验，再按批次进化。</h2><p class="muted">日常分析与研究迭代各有节奏。下面展示反馈进入研究的组织方式，并非正在运行的任务。</p><div class="batch-stages"><div class="batch-stage"><span>01</span><div><h3>留下有价值的研究线索</h3><p>保存难例、预测分歧和专家修正，关联原图与审核版本。</p></div></div><div class="batch-stage"><span>02</span><div><h3>汇集明确授权的反馈</h3><p>由项目所有者选择贡献的项目，再组织共性问题与研究轨迹。</p></div></div><div class="batch-stage"><span>03</span><div><h3>按研究批次探索新方案</h3><p>提出具体机制、执行程序、固定协议评测，让反馈参与下一轮决策。</p></div></div><div class="batch-stage"><span>04</span><div><h3>把有效改进留给下一轮</h3><p>整体评价能力与采用规则，通过才继承；未通过则保留原状态。</p></div></div></div><div class="batch-caption">一次上传或一次反馈，不会立即升级框架。当前网页保存研究请求，后续研究另行组织。</div></section><section class="panel rsi-feedback"><div class="eyebrow">YOUR OBSERVATION MATTERS</div><h2>贡献下一轮研究线索</h2><p class="muted">发现框架还不擅长的场景？把问题和已保存的修正一起留下。</p><label for="research-project">选择我的项目</label><select id="research-project">${own.map(p=>`<option value="${p.id}">${esc(p.name)}</option>`).join('')||'<option value="">请先创建项目</option>'}</select><label for="reason">希望改善什么？</label><textarea id="reason" placeholder="例如：这些低对比图中候选过少，已保存专家修正，请研究漏检原因。" maxlength="3000"></textarea><label class="muted"><input id="consent" type="checkbox"> 我授权将本项目的图像和审核记录用于后续改进研究；不代表允许公开发布。</label><button class="primary" id="request" ${own.length?'':'disabled'}>登记研究请求</button><div id="requests"></div></section></div>`;
 mountEvolution();
 const list=async()=>{if(!$('#research-project').value)return;const rows=await api('/projects/'+$('#research-project').value+'/research');$('#requests').innerHTML=rows.map(r=>`<div class="request-card"><span class="badge">已登记 · 尚未执行</span><p>${esc(r.reason)}</p><span class="muted">${date(r.created)}</span></div>`).join('');};$('#research-project').onchange=guarded(list);await list();$('#request').onclick=guarded(async()=>{await api('/projects/'+$('#research-project').value+'/research',{method:'POST',body:{reason:$('#reason').value,consent:$('#consent').checked}});$('#reason').value='';$('#consent').checked=false;await list();toast('请求已登记，尚未启动训练或自动升级');});
}
function versionView(){shell('version');$('#content').innerHTML=`<div class="page-head"><div class="eyebrow">TRACEABLE BY DESIGN</div><h1>一个工作台，连接可追溯的分析流程。</h1><p class="muted">网站提供友好的入口，科学分析仍调用当前仓库的真实框架。</p></div><div class="grid2"><section class="panel"><h2>当前分析能力</h2>${Object.entries(state.cap.tasks).map(([k,v])=>`<div class="metric"><span>${esc(v)}</span><span class="badge good">已接入</span></div>`).join('')}<p class="muted" style="margin-top:17px">当前网页接入二维灰度图的物理分支。CIF 定制匹配、多层分析和已训练学生的场景路由，仍通过既有研究脚本执行，未伪装成网页已上线功能。</p></section><section class="panel"><h2>数据与版本</h2><div class="metric"><span>平台</span><span>${esc(state.cap.platform_version)}</span></div><div class="metric"><span>引擎基线提交</span><code>${esc(state.cap.source_commit)}</code></div><div class="metric"><span>科学入口</span><span class="tiny">${esc(state.cap.engine)}</span></div><div class="metric"><span>审核保存</span><span>SQLite · 版本化记录</span></div><div class="notice" style="margin-top:20px">网页改动单独接入，不修改既有科学算法，也不覆盖历史专家标注。原平台保持独立运行。</div></section></div>`;}
async function accountView(){
 shell('accounts');const users=await api('/users');$('#content').innerHTML=`<div class="page-head row between"><div><div class="eyebrow">INVITATION ONLY</div><h1>一个人，一个独立身份。</h1><p class="muted">激活码是登录凭据，请私下交给本人；专家也使用独立激活码。</p></div><button class="primary" id="invite">＋ 创建激活码</button></div><section class="panel">${users.map(u=>`<div class="account-row"><div><strong>${esc(u.name)}</strong> <span class="badge">${{admin:'管理员',expert:'专家',researcher:'研究用户'}[u.role]}</span><div class="muted">账号 ${u.id.slice(0,8)} · ${u.disabled?'已停用':'有效'} · ${date(u.created)}</div></div>${u.id!==state.user.id?`<div class="row"><button data-renew="${u.id}">${u.disabled?'重新启用 / 换码':'重置激活码'}</button>${!u.disabled?`<button data-disable="${u.id}">停用</button>`:''}</div>`:''}</div>`).join('')}</section><p class="muted" style="margin-top:15px">管理员负责邀请和停用账号，不因此获得其他用户私人项目的网页访问权。遗失激活码时请重置原账号，不要另建身份；原项目与标注保持不变。</p>`;
 $('#invite').onclick=()=>{const d=modal(`<h2>邀请一位新用户</h2><form id="invite-form"><label for="invite-name">姓名或称呼</label><input id="invite-name" required maxlength="100"><label for="invite-role">账号用途</label><select id="invite-role"><option value="researcher">研究用户 · 私有项目与分析</option><option value="expert">专家 · 接受明确共享并审核</option></select><div class="actions"><button type="button" data-close>取消</button><button class="primary">生成个人激活码</button></div></form>`);$('#invite-form').onsubmit=guarded(async e=>{e.preventDefault();const u=await api('/users',{method:'POST',body:{name:$('#invite-name').value,role:$('#invite-role').value}});d.innerHTML=`<h2>激活码已生成</h2><p class="muted">请立即复制并私下发送给 ${esc(u.name)}。服务器只保存不可逆摘要，关闭后无法重新查看。</p><div class="codebox" id="new-code">${esc(u.activation_code)}</div><div class="actions"><button id="copy-code">复制激活码</button><button class="primary" id="code-done">我已保存</button></div>`;$('#copy-code').onclick=guarded(async()=>{await navigator.clipboard.writeText(u.activation_code);toast('已复制，请勿公开传播');});$('#code-done').onclick=guarded(async()=>{d.innerHTML='';d.close();await accountView();});});};document.querySelectorAll('[data-disable]').forEach(b=>b.onclick=guarded(async()=>{if(confirm('停用后该用户不能继续登录；已有项目和标注保留。确定吗？')){await api('/users/'+b.dataset.disable+'/disable',{method:'POST'});await accountView();}}));
 document.querySelectorAll('[data-renew]').forEach(b=>b.onclick=guarded(async()=>{if(!confirm('生成新码将立即使旧激活码和旧会话失效。账号、项目和标注保持不变。确定吗？'))return;const u=await api('/users/'+b.dataset.renew+'/renew',{method:'POST'});const d=modal(`<h2>已重置 ${esc(u.name)} 的激活码</h2><p class="muted">请立即复制并私下交给本人。原项目和审核记录仍绑定同一个账号。</p><div class="codebox">${esc(u.activation_code)}</div><div class="actions"><button id="renew-copy">复制激活码</button><button class="primary" id="renew-done">我已保存</button></div>`);$('#renew-copy').onclick=guarded(async()=>{await navigator.clipboard.writeText(u.activation_code);toast('已复制');});$('#renew-done').onclick=guarded(async()=>{d.innerHTML='';d.close();await accountView();});}));
}
window.addEventListener('beforeunload',e=>{if(state.dirty){e.preventDefault();e.returnValue='';}});
window.addEventListener('keydown',e=>{if(/INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName))return;if(e.key.toLowerCase()==='b'&&canvasState){canvasState.raw=true;canvasState.draw();}if(e.key==='Escape'&&canvasState){canvasState.pending=[];canvasState.draw();}if((e.ctrlKey||e.metaKey)&&e.key==='z'&&canvasState){e.preventDefault();$('#undo')?.click();}});
window.addEventListener('keyup',e=>{if(e.key.toLowerCase()==='b'&&canvasState){canvasState.raw=false;canvasState.draw();}});
(async()=>{try{state.user=await api('/me');await boot();}catch{loginView();}})();

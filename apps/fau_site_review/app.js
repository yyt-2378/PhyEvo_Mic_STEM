export default function(component) {
  const {data, parentElement, setTriggerValue} = component;
  // The host may retain DOM nodes while invoking JS with updated server data.
  // Dispose the previous event handlers explicitly before rebinding them.
  parentElement.__fauReviewCleanup?.();
  const q = s => parentElement.querySelector(s), qa = s => [...parentElement.querySelectorAll(s)];
  const shell=q('.fau'), map=q('#fau-map'), detail=q('#fau-detail');
  const colors={occupied:'#ff806b',empty_like:'#4ac9f5',uncertain:'#f1cf65'};
  const names={occupied:'有占位倾向',empty_like:'无占位倾向',uncertain:'不确定'};
  const colorNames={occupied:'红色 · 有 / 额外占位',empty_like:'蓝色 · 无 / 空位形态',uncertain:'黄色 · 不确定'};
  const sites=data.sites.map(s=>({...s})), reviews=data.reviews;
  for(const r of Object.values(reviews)) if(r.manual) sites.push({id:r.site_id,sample:r.sample,x:r.center[0],y:r.center[1],status:'uncertain',manual:true});
  let selected=sites.find(s=>s.id===data.resume)||sites[0], sample=selected.sample;
  let center=[], view=[], radius=40, decision=null, alignment='', note='', dirty=false, pending=false, bare=false, addMode=false, moveMode=false, gesture=null;
  let width=500,height=440,destroyed=false,retryAction=null;
  const listeners=[], images={};
  function on(el,event,fn,options){el.addEventListener(event,fn,options);listeners.push(()=>el.removeEventListener(event,fn,options));}
  function message(s){q('#fau-local-message').textContent=s;}
  function human(s){return reviews[s.id];}
  function point(s){return s.id===selected.id?center:human(s)?.center||[s.x,s.y];}
  function savedState(s){return human(s)?.decision||s.status;}
  function displayState(s){return s.id===selected.id&&dirty?decision:savedState(s);}
  function color(s){return human(s)||(s.id===selected.id&&dirty)||q('#fau-model').checked?colors[displayState(s)]:'#c0cdc3';}
  function changes(){dirty=true;retryAction=null;q('#fau-save-state').textContent='有未保存修改';message('颜色预览尚未保存，请点击“保存改色”。');updateDecision();}
  function canLeave(){return !pending&&(!dirty||window.confirm('当前修改尚未保存。放弃本处修改并切换吗？'));}
  function localList(){return sites.filter(s=>s.sample===sample);}
  function nextId(unreviewed=false){const list=localList(),index=list.findIndex(s=>s.id===selected.id);for(let i=1;i<=list.length;i++){const p=list[(index+i)%list.length];if(!unreviewed||!human(p))return p.id;}return selected.id;}
  function select(id,force=false){
    if(!force&&!canLeave())return false;
    selected=sites.find(s=>s.id===id)||selected;sample=selected.sample;
    const r=human(selected);center=(r?.center||[selected.x,selected.y]).slice();view=center.slice();
    // Prefill the displayed label for correction, but never create a human review until Save.
    decision=r?.decision||selected.status;alignment=r?.alignment||'not_reviewed';note=r?.note||'';
    radius=sample==='RA'?42:37;dirty=false;addMode=false;moveMode=false;pending=false;retryAction=null;
    q('#fau-alignment').value=alignment;q('#fau-note').value=note;q('#fau-add').setAttribute('aria-pressed','false');q('#fau-recenter').setAttribute('aria-pressed','false');
    q('#fau-site-title').textContent='02 · '+selected.id+(selected.manual?' · 手工补点':'');
    q('#fau-coordinate').textContent=`x=${center[0].toFixed(1)} y=${center[1].toFixed(1)}`;
    q('#fau-save-state').textContent=r?'已审核 · 第'+r.revision+'版':'未审核';
    qa('[data-sample]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.sample===sample)));
    q('#fau-reference').src=selected.expected||'';q('#fau-aligned').src=selected.target||'';
    q('.fau-comparison').hidden=!selected.expected;
    q('#fau-evidence').textContent=selected.manual?'手工补点：没有对应模型分数；保存的意见不会自动生成应变数值。':
      `模型：${names[selected.status]}。${selected.reason}。两侧柱：${selected.endpoints_measured?'已定位':'未可靠分辨'}；ΔBIC=${selected.bic==null?'不适用':selected.bic.toFixed(2)}；外围 NCC=${selected.ncc==null?'不适用':selected.ncc.toFixed(3)}；局部 D=${selected.local_D_pct==null?'不适用':selected.local_D_pct.toFixed(3)+'%'}。中心纠正后这些旧指标不会自动重算。`;
    const picker=q('#fau-picker');picker.replaceChildren();
    for(const s of localList()){const opt=document.createElement('option');opt.value=s.id;opt.dataset.modelStatus=s.status;opt.textContent=s.id+' · '+colorNames[savedState(s)]+(human(s)?' · 已审核':' · 待审核')+(s.manual?' · 手工补点':'');picker.appendChild(opt);}picker.value=selected.id;
    q('#fau-progress').textContent=localList().filter(s=>human(s)).length+' / '+localList().length+' 已审核';
    busy(false);message('');updateDecision();updateTips();draw();return true;
  }
  function updateDecision(){
    qa('button[data-decision]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.decision===decision)));
    const current=q('#fau-current'),r=human(selected);
    current.dataset.decision=decision;current.dataset.origin=dirty?'draft':r?'human':'model';
    current.textContent=dirty?`待保存：${colorNames[decision]}`:`当前：${colorNames[decision]}（${r?'你的已保存判断':'模型标记 · 待审核'}）`;
    if(r||dirty)current.textContent+=`\n${selected.manual?'手工补点':(data.edition?.startsWith('feedback-')?'本版学习模型：':'原模型：')+colorNames[selected.status]}`;
    q('#fau-save-stay').textContent=decision!==savedState(selected)?'保存改色 · 留在此处':'确认当前颜色并保存';
  }
  function updateTips(){
    q('#fau-map-tip').textContent=addMode?'补点模式：点击总图上漏掉的位置；选择颜色并保存才会入库。':'总图和放大图都可以点选已有圆圈；选颜色只改分类，不移动原位点。';
    q('#fau-detail-tip').textContent=moveMode?'改中心模式：点击局部原图中的正确中心；不会修改原子坐标和旧分析结果。':'点圆圈选中，拖动平移，滚轮放大。加粗圆圈是当前选中位点；按住 B 可暂时隐藏标记。';
    detail.style.cursor=moveMode?'crosshair':'grab';
  }
  function localToScreen(p){const scale=Math.min(width,height)/(2*radius);return [width/2+(p[0]-view[0])*scale,height/2+(p[1]-view[1])*scale];}
  function screenToLocal(e){const rect=detail.getBoundingClientRect(),scale=Math.min(width,height)/(2*radius);return [view[0]+((e.clientX-rect.left)*width/rect.width-width/2)/scale,view[1]+((e.clientY-rect.top)*height/rect.height-height/2)/scale];}
  function drawMap(){
    const meta=data.samples[sample],ctx=map.getContext('2d');map.width=meta.width;map.height=meta.height;
    ctx.fillStyle='#29302b';ctx.fillRect(0,0,map.width,map.height);if(!images[sample]?.complete)return;
    ctx.filter=`brightness(${q('#fau-brightness').value}%)`;ctx.drawImage(images[sample],0,0);ctx.filter='none';
    if(bare)return;
    const displayScale=Math.max(.1,map.getBoundingClientRect().width/map.width);
    for(const s of localList()){
      const [x,y]=point(s),isSelected=s.id===selected.id;
      ctx.beginPath();ctx.arc(x,y,(isSelected?7:3.5)/displayScale,0,2*Math.PI);ctx.lineWidth=(isSelected?2:1)/displayScale;
      ctx.strokeStyle=color(s);ctx.stroke();
      if(isSelected){ctx.font=`${11/displayScale}px Arial`;ctx.fillStyle='#fff';ctx.fillText(s.id,x+10/displayScale,y-9/displayScale);}
    }
  }
  function drawDetail(){
    const rect=q('.fau-detail-stage').getBoundingClientRect(),ratio=window.devicePixelRatio||1;
    width=Math.max(100,rect.width);height=Math.max(100,rect.height);detail.width=Math.round(width*ratio);detail.height=Math.round(height*ratio);
    const ctx=detail.getContext('2d');ctx.scale(ratio,ratio);ctx.fillStyle='#29302b';ctx.fillRect(0,0,width,height);
    const im=images[sample];if(!im?.complete)return;const scale=Math.min(width,height)/(2*radius);
    ctx.save();ctx.translate(width/2,height/2);ctx.scale(scale,scale);ctx.translate(-view[0],-view[1]);
    ctx.filter=`brightness(${q('#fau-brightness').value}%)`;ctx.imageSmoothingEnabled=false;ctx.drawImage(im,0,0);ctx.filter='none';
    const mode=q('#fau-atoms').value;
    if(!bare&&mode!=='none')for(const p of data.samples[sample][mode==='measured'?'measured':'ideal']){
      if(Math.abs(p[0]-view[0])>radius*width/Math.min(width,height)||Math.abs(p[1]-view[1])>radius*height/Math.min(width,height))continue;
      ctx.beginPath();ctx.arc(p[0],p[1],2.3/scale,0,Math.PI*2);ctx.strokeStyle='#e8f0eb';ctx.lineWidth=1/scale;ctx.stroke();
    }
    ctx.restore();
    if(!bare)for(const s of localList()){
      const [cx,cy]=localToScreen(point(s)),active=s.id===selected.id;
      if(cx < -20 || cy < -20 || cx > width+20 || cy > height+20)continue;
      // One open ring per site. Human/draft color replaces model color, not a second overlay.
      ctx.beginPath();ctx.arc(cx,cy,active?17:11,0,2*Math.PI);ctx.strokeStyle=color(s);ctx.lineWidth=active?2.8:1.5;ctx.stroke();
      if(active){ctx.font='bold 12px Arial';ctx.textAlign='center';const label=s.id,tw=ctx.measureText(label).width;
        ctx.fillStyle='rgba(20,28,23,.8)';ctx.fillRect(cx-tw/2-4,cy+22,tw+8,19);ctx.fillStyle='#fff';ctx.fillText(label,cx,cy+36);}
    }
  }
  function draw(){if(!destroyed){
    drawMap();drawDetail();const counts={occupied:0,empty_like:0,uncertain:0};localList().forEach(s=>counts[displayState(s)]++);
    q('#fau-color-counts').textContent=`${sample} 当前：红 ${counts.occupied} · 蓝 ${counts.empty_like} · 黄 ${counts.uncertain}`+(dirty?'（含未保存预览）':'');
  }}
  function setCenter(xy){const meta=data.samples[sample];center=[Math.max(0,Math.min(meta.width-1,xy[0])),Math.max(0,Math.min(meta.height-1,xy[1]))];changes();q('#fau-coordinate').textContent=`x=${center[0].toFixed(1)} y=${center[1].toFixed(1)}`;draw();}
  function busy(state){pending=state;qa('button,select,textarea').forEach(b=>b.disabled=state);}
  function save(next){
    if(pending)return;if(!decision){message('请先选择：红色、蓝色或黄色。');return;}
    const action=retryAction||{release:data.release,model_release:data.model_release||data.release,request_id:crypto.randomUUID(),site_id:selected.id,sample,
      decision,alignment,note,center:center.slice(),base_revision:human(selected)?.revision||0,next_id:next?nextId(true):selected.id};
    retryAction=action;busy(true);message('正在写入服务器，请等待保存回执……');
    setTriggerValue('action',action);
    const timer=setTimeout(()=>{if(!destroyed){busy(false);message('尚未收到保存回执。可重试同一提交；请不要认为已经保存。');}},15000);
    listeners.push(()=>clearTimeout(timer));
  }
  for(const name of Object.keys(data.samples)){images[name]=new Image();images[name].onload=draw;images[name].src=data.samples[name].image;}
  qa('[data-sample]').forEach(b=>on(b,'click',()=>{const r=sites.find(s=>s.sample===b.dataset.sample&&!human(s))||sites.find(s=>s.sample===b.dataset.sample);select(r.id);}));
  qa('button[data-decision]').forEach(b=>on(b,'click',()=>{decision=b.dataset.decision;changes();updateDecision();draw();}));
  on(q('#fau-alignment'),'change',e=>{alignment=e.target.value;changes();});on(q('#fau-note'),'input',e=>{note=e.target.value;changes();});
  on(q('#fau-picker'),'change',e=>{if(!select(e.target.value))e.target.value=selected.id;});
  on(q('#fau-model'),'change',draw);on(q('#fau-brightness'),'input',draw);on(q('#fau-atoms'),'change',draw);
  on(q('#fau-plus'),'click',()=>{radius=Math.max(9,radius/1.35);draw();});on(q('#fau-minus'),'click',()=>{radius=Math.min(180,radius*1.35);draw();});
  on(q('#fau-recenter'),'click',()=>{moveMode=!moveMode;q('#fau-recenter').setAttribute('aria-pressed',String(moveMode));updateTips();});
  on(q('#fau-reset-center'),'click',()=>{setCenter([selected.x,selected.y]);view=center.slice();draw();});
  on(q('#fau-full'),'click',()=>{if(document.fullscreenElement)document.exitFullscreen();else shell.requestFullscreen?.();});
  on(q('#fau-add'),'click',()=>{if(pending)return;addMode=!addMode;q('#fau-add').setAttribute('aria-pressed',String(addMode));updateTips();});
  on(q('#fau-next'),'click',()=>select(nextId()));on(q('#fau-next-pending'),'click',()=>select(nextId(true)));
  on(q('#fau-next-yellow'),'click',()=>{
    const list=localList(),index=list.findIndex(s=>s.id===selected.id);
    for(let i=1;i<=list.length;i++){const s=list[(index+i)%list.length];if(!human(s)&&s.status==='uncertain'){select(s.id);return;}}
    message('本图的模型黄色位点均已审核，可以查看其他颜色或切换样品。');
  });
  on(q('#fau-prev'),'click',()=>{const list=localList(),i=list.findIndex(s=>s.id===selected.id);select(list[(i-1+list.length)%list.length].id);});
  on(q('#fau-save'),'click',()=>save(true));on(q('#fau-save-stay'),'click',()=>save(false));
  on(map,'click',e=>{
    const r=map.getBoundingClientRect(),xy=[(e.clientX-r.left)*map.width/r.width,(e.clientY-r.top)*map.height/r.height];
    if(addMode){if(!canLeave())return;const id=sample+'-M'+crypto.randomUUID().replaceAll('-','').slice(0,12);sites.push({id,sample,x:xy[0],y:xy[1],status:'uncertain',manual:true});select(id,true);changes();return;}
    const nearest=localList().map(s=>({id:s.id,d:Math.hypot(point(s)[0]-xy[0],point(s)[1]-xy[1])})).sort((a,b)=>a.d-b.d)[0];
    if(nearest&&nearest.d<18*map.width/r.width)select(nearest.id);
  });
  on(detail,'wheel',e=>{e.preventDefault();if(pending)return;radius=Math.max(9,Math.min(180,radius*(e.deltaY>0?1.12:1/1.12)));draw();},{passive:false});
  on(detail,'pointerdown',e=>{if(pending)return;detail.focus();detail.setPointerCapture(e.pointerId);gesture={x:e.clientX,y:e.clientY,view:view.slice(),moved:false};});
  on(detail,'pointermove',e=>{if(!gesture||pending)return;const dx=e.clientX-gesture.x,dy=e.clientY-gesture.y;if(Math.hypot(dx,dy)>3)gesture.moved=true;if(!moveMode){const scale=Math.min(width,height)/(2*radius);view=[gesture.view[0]-dx/scale,gesture.view[1]-dy/scale];draw();}});
  on(detail,'pointerup',e=>{
    if(gesture&&!gesture.moved){
      const xy=screenToLocal(e);
      if(moveMode)setCenter(xy);
      else{const p=localToScreen(xy),nearest=localList().map(s=>{const t=localToScreen(point(s));return {id:s.id,d:Math.hypot(t[0]-p[0],t[1]-p[1])};}).sort((a,b)=>a.d-b.d)[0];
        if(nearest&&nearest.d<=22&&nearest.id!==selected.id)select(nearest.id);}
    }gesture=null;
  });
  on(detail,'pointercancel',()=>{gesture=null;});
  on(window,'keydown',e=>{
    const editing=['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName);if(editing||pending)return;
    if(e.key.toLowerCase()==='b'){bare=true;draw();}if(e.key==='ArrowRight'){e.preventDefault();select(nextId());}if(e.key==='ArrowLeft'){e.preventDefault();q('#fau-prev').click();}
    if(['1','2','3'].includes(e.key)){decision=['occupied','empty_like','uncertain'][Number(e.key)-1];changes();updateDecision();draw();}
    if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();save(true);}
  });
  on(window,'keyup',e=>{if(e.key.toLowerCase()==='b'){bare=false;draw();}});on(window,'blur',()=>{bare=false;draw();});
  on(window,'beforeunload',e=>{if(dirty||pending){e.preventDefault();e.returnValue='';}});
  const resize=new ResizeObserver(draw);resize.observe(q('.fau-detail-stage'));
  q('#fau-receipt').textContent=data.receipt||'';select(selected.id,true);
  const cleanup=()=>{destroyed=true;resize.disconnect();listeners.forEach(remove=>remove());Object.values(images).forEach(im=>im.onload=null);};
  parentElement.__fauReviewCleanup=cleanup;
  return cleanup;
}

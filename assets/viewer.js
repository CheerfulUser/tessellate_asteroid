// Shared by every per-asteroid page. The mesh and folded lightcurve are FETCHED
// rather than embedded: 73 KB per page x 16,000 objects exceeds the 1 GB GitHub
// Pages limit. Everything below the fetch is unchanged from the single-page viewer.
let D;
async function boot(){
  const A = window.AST;
  const [sh, lc] = await Promise.all([
    fetch(A.shape).then(r=>r.json()),
    A.has_lc ? fetch(A.lc).then(r=>r.json()).catch(()=>null) : Promise.resolve(null)
  ]);
  const verts=[]; for(let i=0;i<sh.v.length;i+=3) verts.push([sh.v[i],sh.v[i+1],sh.v[i+2]]);
  // faces are polygons of varying size, so walk the per-face counts rather than step by 3
  const facets=[]; let _o=0;
  for(const k of sh.fn){ facets.push(sh.f.slice(_o,_o+k)); _o+=k; }
  D = Object.assign({}, A.D, {verts, facets,
        lcP: lc? lc.phase:[], lcF: lc? lc.flux:[],
        lcE: lc? (lc.err||null):null, lcM: lc? (lc.model_flux||null):null});
  run();
}
function run(){

const cv = document.getElementById('c'), ctx = cv.getContext('2d');
const lcv = document.getElementById('lc'), lctx = lcv.getContext('2d');
let dpr = Math.min(window.devicePixelRatio||1, 2);
function size(){
  for (const [el,g] of [[cv,ctx],[lcv,lctx]]){
    const r = el.getBoundingClientRect();
    el.width = Math.max(1,r.width*dpr); el.height = Math.max(1,r.height*dpr);
    g.setTransform(dpr,0,0,dpr,0,0);
  }
}
window.addEventListener('resize', size); size();

// Keep the info panel clear of the control panel. Both are absolutely positioned in the same
// corner column, so as the info panel grew it began overlapping the lightcurve; cap it against
// the control panel's MEASURED height rather than a guessed constant, and re-measure on resize.
function fitPanels(){
  const info=document.querySelector('.info'), ctrl=document.querySelector('.ctrl');
  if(!info||!ctrl) return;
  const avail = window.innerHeight - ctrl.getBoundingClientRect().height - 20 - 20 - 16;
  info.style.maxHeight = Math.max(160, avail) + 'px';
}
window.addEventListener('resize', fitPanels);

// ---- colour ramps. Two modes:
//   albedo -- cividis over the RELATIVE albedo map, for reading structure.
//   true   -- what the eye would see: relative albedo x the published geometric albedo gives
//             a real reflectance, sRGB-encoded so the darkness is perceptually right, then a
//             mild taxonomic tint. At p=0.117 this is genuinely dark, like charcoal; a body
//             rendered bright grey would badly misrepresent an asteroid surface.
let colourMode='albedo';
function srgb(x){ x=Math.max(0,Math.min(1,x));
  return x<=0.0031308 ? 12.92*x : 1.055*Math.pow(x,1/2.4)-0.055; }
function trueColour(a){
  const refl = D.geoAlbedo*a;
  const e = srgb(refl);
  return [e*255*D.tint[0], e*255*D.tint[1], e*255*D.tint[2]];
}
function ramp(a){
  if(colourMode==='true') return trueColour(a);
  const t = D.hi>D.lo ? (a-D.lo)/(D.hi-D.lo) : 0.5;
  const i = Math.max(0, Math.min(D.lut.length-1, Math.round(t*(D.lut.length-1))));
  return D.lut[i];
}


// ---- quaternions for free drag
function qMul(a,b){return [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
  a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2], a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
  a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]];}
function qAxis(ax,an){const s=Math.sin(an/2);return [Math.cos(an/2),ax[0]*s,ax[1]*s,ax[2]*s];}
function qNorm(q){const n=Math.hypot(...q);return q.map(v=>v/n);}
function qMat(q){const [w,x,y,z]=q;return [
  [1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
  [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
  [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]];}

let orient = qAxis([1,0,0],-1.2), zoom=1, phase=0, spinning=false;
const scale = (()=>{let m=0;for(const v of D.verts)m=Math.max(m,Math.hypot(v[0],v[1],v[2]));return m||1;})();

function draw(){
  const w=cv.width/dpr, h=cv.height/dpr;
  ctx.clearRect(0,0,w,h);
  const R = qMat(orient);
  const a = 2*Math.PI*phase, ca=Math.cos(a), sa=Math.sin(a);
  const S = Math.min(w,h)*0.33*zoom/scale;
  // Centre the body in the space ABOVE the control panel rather than in the whole canvas,
  // so it never clips the lightcurve. Measured from the panel each frame so it stays correct
  // when the window resizes or the panel reflows.
  const panelH = (document.querySelector('.ctrl')?.getBoundingClientRect().height || 0) + 28;
  const cy = Math.max(S*0.9, (h - panelH)/2);
  // Body spin about z, the spin axis by convexinv's convention. Sign VERIFIED against the
  // data, not reasoned from the transform: rendering the mesh at each phase and correlating
  // the synthetic brightness with the observed folded curve gives r=+0.99 this way and
  // r=-0.69 reversed (validate_viewer_sync.py). Do not "fix" it without re-running that.
  const P = D.verts.map(v=>{
    const x=v[0]*ca - v[1]*sa, y=v[0]*sa + v[1]*ca, z=v[2];
    return [R[0][0]*x+R[0][1]*y+R[0][2]*z, R[1][0]*x+R[1][1]*y+R[1][2]*z,
            R[2][0]*x+R[2][1]*y+R[2][2]*z];
  });
  const faces = D.facets.map((f,i)=>{
    const p=f.map(k=>P[k]);
    // NEWELL'S METHOD, summed over every edge. Faces here are polygons of 4-9 vertices, and
    // taking the normal from just the first three is badly conditioned: 180 of Eurydike's 289
    // faces have a near-collinear leading triplet (face 258: cross product 2.6e-5 against a
    // true area of 0.12). The resulting direction is numerically arbitrary and jitters between
    // frames, flipping both the shading and the backface test -- that was the flickering facet.
    let n=[0,0,0];
    for(let k=0;k<p.length;k++){
      const a=p[k], b=p[(k+1)%p.length];
      n[0]+=(a[1]-b[1])*(a[2]+b[2]);
      n[1]+=(a[2]-b[2])*(a[0]+b[0]);
      n[2]+=(a[0]-b[0])*(a[1]+b[1]);
    }
    const nl=Math.hypot(...n)||1; n=n.map(c=>c/nl);
    const cz=p.reduce((s,q)=>s+q[2],0)/p.length;
    const cen=[p.reduce((s,q)=>s+q[0],0)/p.length,p.reduce((s,q)=>s+q[1],0)/p.length,cz];
    if(n[0]*cen[0]+n[1]*cen[1]+n[2]*cen[2] < 0) n=n.map(c=>-c);
    return {p,n,cz,i};
  }).filter(f=>f.n[2]>0).sort((A,B)=>A.cz-B.cz);
  for(const f of faces){

    // In locked mode light the body from the real Sun direction, so the terminator is where
    // TESS saw it; in free mode there is no meaningful illumination geometry, so shade from
    // the camera instead.
    const lit = (mode==='locked')
      ? 0.18 + 0.82*Math.max(0, f.n[0]*D.sunCam[0]+f.n[1]*D.sunCam[1]+f.n[2]*D.sunCam[2])
      : 0.25 + 0.75*Math.max(0, f.n[2]);
    const c = ramp(D.alb[f.i]);
    ctx.beginPath();
    f.p.forEach((q,k)=>{const X=w/2+q[0]*S, Y=cy-q[1]*S; k?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});
    ctx.closePath();
    ctx.fillStyle = `rgb(${Math.round(c[0]*lit)},${Math.round(c[1]*lit)},${Math.round(c[2]*lit)})`;
    ctx.fill();
    ctx.strokeStyle = D.weak[f.i] ? 'rgba(224,164,88,0.55)' : 'rgba(255,255,255,0.10)';
    ctx.lineWidth = D.weak[f.i] ? 1.1 : 0.5;
    ctx.stroke();
  }
  drawLC();
}

function drawLC(){
  // Axes with real ticks and labels: the panel is a measurement, not a sparkline, and without
  // a flux scale the amplitude is unreadable.
  const w=lcv.width/dpr, h=lcv.height/dpr;
  const L=58, Rp=8, T=8, B=26;    // L leaves room for tick text AND the rotated y label
  lctx.clearRect(0,0,w,h);
  if(!D.lcP.length) return;
  const eArr=D.lcE||D.lcF.map(()=>0);
  let vals=D.lcF.map((f,i)=>f-eArr[i]).concat(D.lcF.map((f,i)=>f+eArr[i]));
  if(D.lcM) vals=vals.concat(D.lcM);
  const fmin=Math.min(...vals), fmax=Math.max(...vals);
  const rng=(fmax-fmin)||1, lo=fmin-0.12*rng, hi=fmax+0.12*rng;
  const X=p=>L+p*(w-L-Rp), Y=f=>h-B-((f-lo)/(hi-lo))*(h-T-B);

  lctx.strokeStyle='#2b3342'; lctx.lineWidth=1;
  lctx.beginPath(); lctx.moveTo(L,T); lctx.lineTo(L,h-B); lctx.lineTo(w-Rp,h-B); lctx.stroke();
  lctx.fillStyle='#8791a3'; lctx.font='9px ui-monospace,monospace';

  lctx.textAlign='center'; lctx.textBaseline='top';
  for(const t of [0,0.25,0.5,0.75,1]){
    const x=X(t);
    lctx.strokeStyle='#2b3342'; lctx.beginPath();
    lctx.moveTo(x,h-B); lctx.lineTo(x,h-B+4); lctx.stroke();
    lctx.fillText(t.toFixed(2),x,h-B+6);
  }
  lctx.textAlign='right'; lctx.textBaseline='middle';
  for(const fv of [lo+(hi-lo)*0.12, (lo+hi)/2, hi-(hi-lo)*0.12]){
    const y=Y(fv);
    lctx.strokeStyle='#2b3342'; lctx.beginPath();
    lctx.moveTo(L-4,y); lctx.lineTo(L,y); lctx.stroke();
    lctx.fillText(fv.toFixed(3),L-6,y);
  }
  lctx.fillStyle='#545e70';
  lctx.textAlign='center'; lctx.textBaseline='bottom';
  lctx.fillText('Rotational phase',(L+w-Rp)/2,h-1);
  lctx.save(); lctx.translate(11,(T+h-B)/2); lctx.rotate(-Math.PI/2);
  lctx.textAlign='center'; lctx.textBaseline='top';
  lctx.fillText('Relative flux',0,0); lctx.restore();

  // model first, so the data sits on top of it
  if(D.lcM){
    lctx.save(); lctx.setLineDash([5,3]);
    lctx.strokeStyle='#e06c75'; lctx.lineWidth=1.5; lctx.beginPath();
    D.lcP.forEach((p,i)=>{i?lctx.lineTo(X(p),Y(D.lcM[i])):lctx.moveTo(X(p),Y(D.lcM[i]));});
    lctx.stroke(); lctx.restore();
  }
  // observed: error bar then marker
  lctx.strokeStyle='#7c6bb0'; lctx.lineWidth=1;
  D.lcP.forEach((p,i)=>{
    const e=eArr[i]; if(!(e>0)) return;
    const x=X(p);
    lctx.beginPath(); lctx.moveTo(x,Y(D.lcF[i]-e)); lctx.lineTo(x,Y(D.lcF[i]+e)); lctx.stroke();
  });
  lctx.fillStyle='#a78bfa';
  D.lcP.forEach((p,i)=>{
    lctx.beginPath(); lctx.arc(X(p),Y(D.lcF[i]),1.9,0,2*Math.PI); lctx.fill();
  });
  if(D.lcM){
    lctx.font='9px ui-monospace,monospace'; lctx.textAlign='right'; lctx.textBaseline='top';
    lctx.fillStyle='#e06c75'; lctx.fillText('- - convex model', w-Rp, T);
  }

  const x=X(phase);
  lctx.strokeStyle='#e0a458'; lctx.lineWidth=1.4;
  lctx.beginPath(); lctx.moveTo(x,T); lctx.lineTo(x,h-B); lctx.stroke();
  let j=0; while(j<D.lcP.length-1 && D.lcP[j+1]<phase) j++;
  const f0=D.lcF[j], f1=D.lcF[Math.min(j+1,D.lcF.length-1)];
  const p0=D.lcP[j], p1=D.lcP[Math.min(j+1,D.lcP.length-1)];
  const t=(p1>p0)?(phase-p0)/(p1-p0):0;
  lctx.fillStyle='#e0a458'; lctx.beginPath();
  lctx.arc(x, Y(f0+(f1-f0)*t), 3.2, 0, 2*Math.PI); lctx.fill();
}

// ---- interaction. Two modes:
//   free   -- drag reorients the VIEW (quaternion); phase is unchanged, so the lightcurve
//             marker stays put. Use it to inspect the body from any direction.
//   locked -- drag rotates the BODY about its spin axis and nothing else, so horizontal drag
//             IS the rotational phase and the lightcurve marker tracks it exactly. Play/pause
//             belongs to this mode only: animating a free-dragged view would not correspond
//             to any physical rotation.
let mode='free';
const slider=document.getElementById('ph'), phv=document.getElementById('phv');
const spinBtn=document.getElementById('spin'), hint=document.getElementById('hint');
const mFree=document.getElementById('mFree'), mLock=document.getElementById('mLock');
function setPhase(p){ phase=((p%1)+1)%1; slider.value=Math.round(phase*1000); phv.textContent=phase.toFixed(3); }
function setMode(m){
  mode=m;
  const lock = m==='locked';
  mFree.setAttribute('aria-pressed', lock?'false':'true');
  mLock.setAttribute('aria-pressed', lock?'true':'false');
  spinBtn.disabled = !lock;
  if(!lock && spinning){ spinning=false; spinBtn.setAttribute('aria-pressed','false');
    spinBtn.innerHTML='&#9654; Play'; }
  hint.innerHTML = lock
    ? 'Viewed equator-on, with the assumed spin axis vertical. Drag horizontally to turn the body about that axis &mdash; the lightcurve marker follows. Press Play to rotate at a steady rate.'
    : 'Drag to rotate the view freely. Switch to <em>Locked to axis</em> to drag the body around its spin axis, synced to the lightcurve.';
  cv.style.cursor = lock ? 'ew-resize' : 'grab';
  if(lock) orient = D.camQ.slice();   // snap to the TESS viewing geometry
  draw();
}
// Surface-colouring toggle removed: it gave the mesh and the info table more room, and
// with the per-facet albedo map disabled the two modes differed only by a flat tint.
// setColour is kept as a no-op so the existing init call needs no special-casing.
function setColour(){ colourMode='albedo'; draw(); }
mFree.addEventListener('click', ()=>setMode('free'));
mLock.addEventListener('click', ()=>setMode('locked'));
slider.addEventListener('input', ()=>{ setPhase(slider.value/1000); if(!spinning) draw(); });
spinBtn.addEventListener('click', ()=>{
  if(mode!=='locked') return;
  spinning=!spinning;
  spinBtn.setAttribute('aria-pressed', spinning?'true':'false');
  spinBtn.innerHTML = spinning ? '&#10073;&#10073; Pause' : '&#9654; Play';
});
document.getElementById('reset').addEventListener('click', ()=>{
  orient=qAxis([1,0,0],-1.2); zoom=1; setPhase(0); draw();
});
let drag=false,lx=0,ly=0;
cv.addEventListener('pointerdown',e=>{drag=true;lx=e.clientX;ly=e.clientY;cv.setPointerCapture(e.pointerId);});
window.addEventListener('pointermove',e=>{
  if(!drag) return;
  const dx=e.clientX-lx, dy=e.clientY-ly; lx=e.clientX; ly=e.clientY;
  if(mode==='locked'){
    setPhase(phase + dx/360);          // a full drag across ~360 px is one rotation
    draw();
  } else {
    orient=qNorm(qMul(qAxis([1,0,0],dy*0.008), qMul(qAxis([0,1,0],dx*0.008), orient)));
    if(!spinning) draw();
  }
});
window.addEventListener('pointerup',()=>{drag=false;});
cv.addEventListener('wheel',e=>{e.preventDefault();zoom*=e.deltaY<0?1.08:0.93;
  zoom=Math.max(0.4,Math.min(zoom,3.5)); if(!spinning) draw();},{passive:false});

const PERIOD_MS = 8000;
let last=performance.now();
function tick(now){
  const dt=now-last; last=now;
  if(spinning){ setPhase(phase + dt/PERIOD_MS); draw(); }
  requestAnimationFrame(tick);
}
fitPanels(); setPhase(0); setColour('albedo'); setMode('locked');
// autoplay: the body is rotating as soon as the page opens, which is what makes the shape and
// its lightcurve legible at a glance. setMode('locked') must run FIRST -- it clears `spinning`
// when leaving locked mode, so setting the flag before it would be undone immediately.
spinning = true;
spinBtn.setAttribute('aria-pressed', 'true');
spinBtn.innerHTML = '&#10073;&#10073; Pause';
requestAnimationFrame(tick);
window.addEventListener('resize', draw);

}
boot();

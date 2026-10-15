// Mobile: the info panel collapses to its name, tapping expands it. The .open class has no
// styling outside the max-width query, so binding this on every viewport is harmless -- desktop
// toggles a class that does nothing. Kept inline in both shared scripts so no page needs an
// extra tag, and guarded so it can never take a page down.
(function(){
  function initInfoToggle(){
    try{
      var info=document.querySelector('.panel.info');
      if(!info) return;
      var h=info.querySelector('h1');
      if(!h||h.dataset.tapInit) return;
      h.dataset.tapInit='1';
      h.setAttribute('role','button');
      h.setAttribute('tabindex','0');
      var sync=function(){ h.setAttribute('aria-expanded', info.classList.contains('open')?'true':'false'); };
      var toggle=function(e){ e.preventDefault(); info.classList.toggle('open'); sync(); };
      h.addEventListener('click', toggle);
      h.addEventListener('keydown', function(e){
        if(e.key==='Enter'||e.key===' ') toggle(e);
      });
      sync();
    }catch(e){ /* never break the page over a UI affordance */ }
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',initInfoToggle);
  else initInfoToggle();
})();

// Send the back link where the visitor actually came from. Asteroid pages hardcode
// "<- Catalog" pointing at index.html, but most visitors arrive from search.html, and dropping
// them on the home page loses the query, filters and sort they had set up. Kept inline in both
// shared scripts rather than a third file, so no page needs an extra tag or request.
(function(){
  function fixBackLink(){
    try{
      var a=document.querySelector('a.home');
      if(!a||a.dataset.backFixed) return;
      var ref=document.referrer; if(!ref) return;
      var u=new URL(ref, window.location.href);
      if(u.origin!==window.location.origin) return;          // same-origin navigation only
      if(!/\/search\.html$/.test(u.pathname)) return;
      a.href='../search.html'; a.innerHTML='&larr; Back to browse';
      a.dataset.backFixed='1';
    }catch(e){ /* a broken referrer must never take the page down */ }
  }
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',fixBackLink);
  else fixBackLink();
})();

// MOCK-UP (local only, not published): viewer.js extended for binary systems. A shared clock
// drives the primary's rotation AND the orbit, the secondary is drawn orbiting the primary, and the
// lightcurve panel toggles between the primary fold and the mutual-event fold (residuals folded at
// the orbit period). Values the data do not yet determine are labelled as assumptions in the panel.
// Shared by every per-asteroid page. The mesh and folded lightcurve are FETCHED
// rather than embedded: 73 KB per page x 16,000 objects exceeds the 1 GB GitHub
// Pages limit. Everything below the fetch is unchanged from the single-page viewer.
let D;
// Model choice for objects that also have a joint TESS + catalog fit (AST.alt): that model is the
// default, and ?model=tess shows the TESS-only catalogue model. Switching reloads the page with the
// other parameter rather than rebuilding in place, so no event handler is ever bound twice.
function modelChoice(A0){
  const useAlt = !!A0.alt && new URLSearchParams(location.search).get('model') !== 'tess';
  try{
    document.querySelectorAll('[data-model]').forEach(el=>{
      el.hidden = (el.dataset.model === 'atlas') !== useAlt;
    });
    if(A0.alt){
      // its own group, at the end of the rotation-mode row and offset from it
      const modeRow = document.querySelector('.panel.ctrl .row');
      if(modeRow && !document.getElementById('modelrow')){
        const row = document.createElement('span');
        row.id = 'modelrow';
        row.style.cssText = 'display:inline-flex;gap:8px;align-items:center;margin-left:18px';
        row.setAttribute('role', 'group'); row.setAttribute('aria-label', 'Shape model');
        for(const [label, alt] of [[A0.alt.label || 'TESS + catalog', true], ['TESS only', false]]){
          const btn = document.createElement('button');
          btn.className = 'btn'; btn.textContent = label;
          btn.setAttribute('aria-pressed', alt === useAlt ? 'true' : 'false');
          btn.addEventListener('click', ()=>{
            if(alt === useAlt) return;
            const u = new URL(location.href);
            if(alt) u.searchParams.delete('model'); else u.searchParams.set('model', 'tess');
            location.href = u.toString();
          });
          row.appendChild(btn);
        }
        modeRow.appendChild(row);
      }
    }
  }catch(e){ /* never break the page over the toggle */ }
  if(!useAlt) return A0;
  return Object.assign({}, A0, {shape: A0.alt.shape, lc: A0.alt.lc, D: Object.assign({}, A0.D, A0.alt.D)});
}
async function boot(){
  const A = modelChoice(window.AST);
  const [sh, lc, bin] = await Promise.all([
    fetch(A.shape).then(r=>r.json()),
    A.has_lc ? fetch(A.lc).then(r=>r.json()).catch(()=>null) : Promise.resolve(null),
    A.binary ? fetch(A.binary).then(r=>r.json()).catch(()=>null) : Promise.resolve(null)
  ]);
  const verts=[]; for(let i=0;i<sh.v.length;i+=3) verts.push([sh.v[i],sh.v[i+1],sh.v[i+2]]);
  // faces are polygons of varying size, so walk the per-face counts rather than step by 3
  const facets=[]; let _o=0;
  for(const k of sh.fn){ facets.push(sh.f.slice(_o,_o+k)); _o+=k; }
  D = Object.assign({}, A.D, {verts, facets,
        lcP: lc? lc.phase:[], lcF: lc? lc.flux:[],
        lcE: lc? (lc.err||null):null, lcM: lc? (lc.model_flux||null):null, bin});
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
// ---- binary state: one clock (hours) drives both the primary's rotation and the orbit
const B = D.bin;
// any number of companions (each its own orbit and event fold) and further periodic signals (folds only)
const COMPS = B ? (B.companions || [B]) : [], SIGS = B ? (B.signals || []) : [];
const P1 = B ? B.P1_hr : 1, PORB = COMPS.length ? COMPS[0].P_orb_hr : 1;
const T0 = (COMPS.length && COMPS[0].t0_hr != null) ? COMPS[0].t0_hr : 0;   // MJD hours at clock zero
let tHr = 0, ophase = 0, lcView = 'primary';
function compPhase(c){ const t0 = (c.t0_hr != null) ? c.t0_hr : T0;
  return ((((tHr + T0 - t0) / c.P_orb_hr) % 1) + 1) % 1; }
function sigPhase(s){ return ((((tHr + T0) / s.period_hr) % 1) + 1) % 1; }
const scale = (()=>{let m=0;for(const v of D.verts)m=Math.max(m,Math.hypot(v[0],v[1],v[2]));return m||1;})();

// ---- km scale. convexinv shapes are in arbitrary units, so km per unit comes from the catalogue
// diameter, taken as the diameter of the sphere with the mesh's volume (the scaling used against
// occultation chords). Read from AST.diam_km when the page carries it, else from the info panel's
// Diameter row; with no diameter there is no bar.
const kmPerUnit = (()=>{
  let dk = (window.AST && +window.AST.diam_km) || 0;
  if(!(dk > 0)){
    for(const row of document.querySelectorAll('.stat-row')){
      const l = row.querySelector('.stat-label'), v = row.querySelector('.stat-value');
      if(l && v && l.textContent.trim().toLowerCase() === 'diameter'){
        const m = v.textContent.match(/([0-9.]+)\s*km/); if(m) dk = parseFloat(m[1]);
      }
    }
  }
  if(!(dk > 0)) return null;
  // volume as a sum of tetrahedra from the vertex centroid, which lies inside a convex hull, so
  // every term is positive whatever the facet winding
  const c = [0,0,0]; for(const v of D.verts){ c[0]+=v[0]; c[1]+=v[1]; c[2]+=v[2]; }
  c[0]/=D.verts.length; c[1]/=D.verts.length; c[2]/=D.verts.length;
  let V = 0;
  for(const f of D.facets){
    const a = D.verts[f[0]].map((x,i)=>x-c[i]);
    for(let k=1;k<f.length-1;k++){
      const b = D.verts[f[k]].map((x,i)=>x-c[i]), d = D.verts[f[k+1]].map((x,i)=>x-c[i]);
      V += Math.abs(a[0]*(b[1]*d[2]-b[2]*d[1]) - a[1]*(b[0]*d[2]-b[2]*d[0]) + a[2]*(b[0]*d[1]-b[1]*d[0])) / 6;
    }
  }
  return V > 0 ? dk / (2*Math.cbrt(3*V/(4*Math.PI))) : null;
})();

// Extent bars, fixed: horizontal = the body's maximum width across the spin axis (largest
// vertex-to-vertex distance projected on the equatorial plane, the widest it appears while it
// spins), vertical = its height along the spin axis (body z, unchanged by spin). Centred on the
// body, horizontal beneath and vertical to the right (the panels sit on the left), just outside
// the circle the body sweeps in any orientation.
const extW = (()=>{
  let m = 0;
  for(let i=0;i<D.verts.length;i++) for(let j=i+1;j<D.verts.length;j++){
    const a=D.verts[i], b=D.verts[j];
    m = Math.max(m, Math.hypot(a[0]-b[0], a[1]-b[1]));
  }
  return m;
})();
const extH = (()=>{
  let lo=Infinity, hi=-Infinity;
  for(const v of D.verts){ lo=Math.min(lo,v[2]); hi=Math.max(hi,v[2]); }
  return hi-lo;
})();
// equivalent-sphere radius of the primary in mesh units (same volume integral as the km scale)
const Req = (()=>{
  const c=[0,0,0]; for(const v of D.verts){c[0]+=v[0];c[1]+=v[1];c[2]+=v[2];}
  c[0]/=D.verts.length; c[1]/=D.verts.length; c[2]/=D.verts.length;
  let V=0;
  for(const f of D.facets){ const a=D.verts[f[0]].map((x,i)=>x-c[i]);
    for(let k=1;k<f.length-1;k++){ const b=D.verts[f[k]].map((x,i)=>x-c[i]), d=D.verts[f[k+1]].map((x,i)=>x-c[i]);
      V+=Math.abs(a[0]*(b[1]*d[2]-b[2]*d[1])-a[1]*(b[0]*d[2]-b[2]*d[0])+a[2]*(b[0]*d[1]-b[1]*d[0]))/6; } }
  return Math.cbrt(3*V/(4*Math.PI));
})();
// the secondary: a triaxial ellipsoid (axis ratios from the binary file), radius q * Req, on a
// circular orbit of radius (a/Dp) * 2 Req in the primary's equatorial plane, its long axis kept
// pointing at the primary (synchronous rotation)
const SECS = COMPS.map(C=>{
  const rs = C.q * Req, ax = C.sec_axes || [1,1,1], NL=10, NM=20, V=[], F=[];
  for(let i=0;i<=NL;i++){ const th=Math.PI*i/NL;
    for(let j=0;j<NM;j++){ const ph=2*Math.PI*j/NM;
      V.push([rs*ax[0]*Math.sin(th)*Math.cos(ph), rs*ax[1]*Math.sin(th)*Math.sin(ph), rs*ax[2]*Math.cos(th)]); } }
  for(let i=0;i<NL;i++) for(let j=0;j<NM;j++)
    F.push([i*NM+j, i*NM+(j+1)%NM, (i+1)*NM+(j+1)%NM, (i+1)*NM+j]);
  return {V, F, a: C.a_Dp * 2 * Req, rs};
});
// The orbit is the FITTED two-disc geometry (build_binary_mock.py), set relative to the TESS line of sight: o = towards
// TESS, u = across the sky in the primary's equatorial plane, w = o x u. At conjunction (theta' = 0) the secondary is
// between the primary and TESS, offset b on the sky (impact parameter); the sky separation is then
// sqrt((a sin theta')^2 + (b cos theta')^2), exactly what the fit used, so the drawn overlaps reproduce the dips.
const ORBS = SECS.map((SEC, k)=>{
  const C = COMPS[k];
  const R = qMat(D.camQ);
  const o = [R[2][0], R[2][1], R[2][2]];
  let u = [-o[1], o[0], 0]; const nu = Math.hypot(...u) || 1; u = u.map(x=>x/nu);
  const w = [o[1]*u[2]-o[2]*u[1], o[2]*u[0]-o[0]*u[2], o[0]*u[1]-o[1]*u[0]];
  const beta = Math.asin(Math.min(1, C.b_Rp*Req/SEC.a));
  return {o, u, w, cb: Math.cos(beta), sb: Math.sin(beta)};
});
// bodies known only from a further signal (their spin): drawn translucent, spinning at the measured period, at a fixed
// placeholder position outside every fitted orbit and offset on the sky so they never cross the primary
const UNC = (B && B.unconstrained ? B.unconstrained : []).map((U, j)=>{
  const rs = Math.max(U.size_ratio_min, 0.05) * Req, ax=[1.4,1,0.9], NL=10, NM=20, V=[], F=[];
  for(let i=0;i<=NL;i++){ const th=Math.PI*i/NL;
    for(let jj=0;jj<NM;jj++){ const ph=2*Math.PI*jj/NM;
      V.push([rs*ax[0]*Math.sin(th)*Math.cos(ph), rs*ax[1]*Math.sin(th)*Math.sin(ph), rs*ax[2]*Math.cos(th)]); } }
  for(let i=0;i<NL;i++) for(let jj=0;jj<NM;jj++) F.push([i*NM+jj, i*NM+(jj+1)%NM, (i+1)*NM+(jj+1)%NM, (i+1)*NM+jj]);
  const R = qMat(D.camQ), o=[R[2][0],R[2][1],R[2][2]];
  let u=[-o[1],o[0],0]; const nu=Math.hypot(...u)||1; u=u.map(x=>x/nu);
  const w=[o[1]*u[2]-o[2]*u[1], o[2]*u[0]-o[0]*u[2], o[0]*u[1]-o[1]*u[0]];
  const amax = Math.max(Req, ...SECS.map(x=>x.a+x.rs)), dist = 1.35*amax + 2*rs;
  const pos = [0,1,2].map(i=> dist*(0.95*u[i] + 0.3*w[i]) * (j%2 ? -1 : 1));
  return {V, F, rs, pos, P: U.P_spin_hr, label: U.label};
});
function secPos(op, k){
  const C = COMPS[k], ORB = ORBS[k];
  const th = 2*Math.PI*(op - (C.dip_phase_front || 0)), s1=Math.sin(th), c1=Math.cos(th), a=SECS[k].a;
  return [0,1,2].map(i=> a*(s1*ORB.u[i] + c1*(ORB.cb*ORB.o[i] + ORB.sb*ORB.w[i])));
}
function fmtKm(km){
  return km >= 1 ? `${+km.toPrecision(3)} km` : `${+(km*1000).toPrecision(3)} m`;
}
function drawExtentBars(S, cy0, w, h, panelH){
  if(!kmPerUnit || !(extW > 0) || !(extH > 0)) return;
  // Each bar tracks the body until it reaches its length at the default zoom, then holds that
  // length while its label falls to the distance it now spans: zooming out it shrinks with the body
  // and reads the full width / height, zooming in it stays readable instead of running off screen.
  const S1 = S / zoom;
  // Positions are anchored where the bars sit at the default zoom (cy0 = the body centre at zoom 1),
  // so they stay put on the page as the view zooms; the body centre itself drifts with zoom.
  const hw = Math.min(extW*S, extW*S1)/2, hh = Math.min(extH*S, extH*S1)/2, tick = 5, gap = 16;
  const col = (getComputedStyle(document.body).getPropertyValue('--text-dim') || '').trim() || '#9aa3b2';
  ctx.save();
  ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1.2;
  ctx.font = '12px "IBM Plex Mono", ui-monospace, monospace';
  // horizontal: maximum width, beneath the body, kept above the control panel
  const yb = Math.min(cy0 + scale*S1 + gap, h - panelH - 24);
  ctx.beginPath();
  ctx.moveTo(w/2-hw, yb); ctx.lineTo(w/2+hw, yb);
  ctx.moveTo(w/2-hw, yb-tick); ctx.lineTo(w/2-hw, yb+tick);
  ctx.moveTo(w/2+hw, yb-tick); ctx.lineTo(w/2+hw, yb+tick);
  ctx.stroke();
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  ctx.fillText(fmtKm(2*hw/S*kmPerUnit), w/2, yb + 7);
  // vertical: height along the spin axis, to the right of the body, kept inside the canvas
  const xb = Math.min(w/2 + scale*S1 + gap, w - 70);
  ctx.beginPath();
  ctx.moveTo(xb, cy0-hh); ctx.lineTo(xb, cy0+hh);
  ctx.moveTo(xb-tick, cy0-hh); ctx.lineTo(xb+tick, cy0-hh);
  ctx.moveTo(xb-tick, cy0+hh); ctx.lineTo(xb+tick, cy0+hh);
  ctx.stroke();
  ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
  ctx.fillText(fmtKm(2*hh/S*kmPerUnit), xb + 8, cy0);
  ctx.restore();
}

function draw(){
  const w=cv.width/dpr, h=cv.height/dpr;
  ctx.clearRect(0,0,w,h);
  const R = qMat(orient);
  const a = 2*Math.PI*phase, ca=Math.cos(a), sa=Math.sin(a);
  let S = Math.min(w,h)*0.33*zoom/Math.max(scale, ...SECS.map(x=>x.a + x.rs), ...UNC.map(x=>Math.hypot(...x.pos) + x.rs));
  // Centre the body in the space ABOVE the control panel rather than in the whole canvas,
  // so it never clips the lightcurve. Measured from the panel each frame so it stays correct
  // when the window resizes or the panel reflows.
  const panelH = (document.querySelector('.ctrl')?.getBoundingClientRect().height || 0) + 28;
  // On mobile the info panel is a bar ACROSS THE TOP, so the free space starts below it and
  // centring on the full canvas puts the body too low. Detect that layout by the panel spanning
  // the width -- the desktop panel is a narrow column and never does. Capped so an EXPANDED
  // panel cannot shove the body off-screen.
  let topOff = 0;
  const infoEl = document.querySelector('.panel.info');
  if(infoEl){
    const ri = infoEl.getBoundingClientRect();
    if(ri.width > w*0.8) topOff = Math.min(ri.bottom + 12, h*0.35);
  }
  const band = h - panelH - topOff;
  // and shrink to fit that band, so the shorter space does not clip the body
  if(topOff > 0 && band > 0) S = Math.min(S, band/(2.2*scale));
  const cy = topOff > 0 ? topOff + band/2 : Math.max(S*0.9, (h - panelH)/2);
  // Body spin about z, the spin axis by convexinv's convention. Sign VERIFIED against the
  // data, not reasoned from the transform: rendering the mesh at each phase and correlating
  // the synthetic brightness with the observed folded curve gives r=+0.99 this way and
  // r=-0.69 reversed (validate_viewer_sync.py). Do not "fix" it without re-running that.
  const P = D.verts.map(v=>{
    const x=v[0]*ca - v[1]*sa, y=v[0]*sa + v[1]*ca, z=v[2];
    return [R[0][0]*x+R[0][1]*y+R[0][2]*z, R[1][0]*x+R[1][1]*y+R[1][2]*z,
            R[2][0]*x+R[2][1]*y+R[2][2]*z];
  });
  let faces = D.facets.map((f,i)=>{
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
  }).filter(f=>f.n[2]>0);
  SECS.forEach((SEC, kc)=>{
    const sp = secPos(compPhase(COMPS[kc]), kc), th = Math.atan2(sp[1], sp[0]), ct=Math.cos(th), st=Math.sin(th);
    const cx=sp[0], cyo=sp[1], cz0=sp[2];
    const PS = SEC.V.map(v=>{
      const x=v[0]*ct - v[1]*st + cx, y=v[0]*st + v[1]*ct + cyo, z=v[2] + cz0;
      return [R[0][0]*x+R[0][1]*y+R[0][2]*z, R[1][0]*x+R[1][1]*y+R[1][2]*z, R[2][0]*x+R[2][1]*y+R[2][2]*z];
    });
    const cc = [0,1,2].map(i=>R[i][0]*cx+R[i][1]*cyo+R[i][2]*cz0);
    for(const f of SEC.F){
      const p=f.map(k=>PS[k]);
      let n=[0,0,0];
      for(let k=0;k<p.length;k++){ const a=p[k], b=p[(k+1)%p.length];
        n[0]+=(a[1]-b[1])*(a[2]+b[2]); n[1]+=(a[2]-b[2])*(a[0]+b[0]); n[2]+=(a[0]-b[0])*(a[1]+b[1]); }
      const nl=Math.hypot(...n)||1; n=n.map(c=>c/nl);
      const cen=[0,1,2].map(i=>p.reduce((s,q)=>s+q[i],0)/p.length);
      if(n[0]*(cen[0]-cc[0])+n[1]*(cen[1]-cc[1])+n[2]*(cen[2]-cc[2]) < 0) n=n.map(c=>-c);
      if(n[2]>0) faces.push({p, n, cz: cen[2], i: -1, sec: true});
    }
  });
  const uncLabels = [];
  UNC.forEach(Ub=>{
    const ang = 2*Math.PI*((tHr + T0) / Ub.P), ct=Math.cos(ang), st=Math.sin(ang), [px,py,pz]=Ub.pos;
    const PS = Ub.V.map(v=>{ const x=v[0]*ct - v[1]*st + px, y=v[0]*st + v[1]*ct + py, z=v[2] + pz;
      return [R[0][0]*x+R[0][1]*y+R[0][2]*z, R[1][0]*x+R[1][1]*y+R[1][2]*z, R[2][0]*x+R[2][1]*y+R[2][2]*z]; });
    const cc = [0,1,2].map(i=>R[i][0]*px+R[i][1]*py+R[i][2]*pz);
    for(const f of Ub.F){
      const p=f.map(k=>PS[k]); let n=[0,0,0];
      for(let k=0;k<p.length;k++){ const a=p[k], b=p[(k+1)%p.length];
        n[0]+=(a[1]-b[1])*(a[2]+b[2]); n[1]+=(a[2]-b[2])*(a[0]+b[0]); n[2]+=(a[0]-b[0])*(a[1]+b[1]); }
      const nl=Math.hypot(...n)||1; n=n.map(c=>c/nl);
      const cen=[0,1,2].map(i=>p.reduce((s,q)=>s+q[i],0)/p.length);
      if(n[0]*(cen[0]-cc[0])+n[1]*(cen[1]-cc[1])+n[2]*(cen[2]-cc[2]) < 0) n=n.map(c=>-c);
      if(n[2]>0) faces.push({p, n, cz: cen[2], i: -1, sec: true, ghost: true});
    }
    uncLabels.push([cc, Ub.rs]);
  });
  faces.sort((A,B)=>A.cz-B.cz);
  for(const f of faces){

    // In locked mode light the body from the real Sun direction, so the terminator is where
    // TESS saw it; in free mode there is no meaningful illumination geometry, so shade from
    // the camera instead.
    // Ambient floor raised from 0.18. At 0.18 an unlit facet rendered at about RGB 26 against a
    // background of 17 -- indistinguishable, so facets crossing the terminator read as HOLES in
    // the body rather than as its night side. 0.34 keeps the terminator obvious while leaving
    // the dark side clearly part of the object.
    const lit = (mode==='locked')
      ? 0.34 + 0.66*Math.max(0, f.n[0]*D.sunCam[0]+f.n[1]*D.sunCam[1]+f.n[2]*D.sunCam[2])
      : 0.34 + 0.66*Math.max(0, f.n[2]);
    const c = f.sec ? [150,160,180] : ramp(D.alb[f.i]);
    ctx.save();
    ctx.globalAlpha = Math.min(1, f.n[2] / 0.10) * (f.ghost ? 0.45 : 1);
    ctx.beginPath();
    f.p.forEach((q,k)=>{const X=w/2+q[0]*S, Y=cy-q[1]*S; k?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});
    ctx.closePath();
    ctx.fillStyle = `rgb(${Math.round(c[0]*lit)},${Math.round(c[1]*lit)},${Math.round(c[2]*lit)})`;
    ctx.fill();
    // Fade the WHOLE facet as it turns edge-on, fill included.
    //
    // Fading only the stroke was not enough. The fill's geometric area does fall to zero at the
    // limb, but canvas renders a sub-pixel sliver as an ANTIALIASED line with partial coverage,
    // so it stays visible far below one pixel of area and then vanishes outright when the facet
    // is culled at n_z = 0. That step is the flicker. Applying the same factor through
    // globalAlpha makes both fill and stroke reach zero exactly when the facet is dropped.
    ctx.strokeStyle = (!f.sec && D.weak[f.i]) ? 'rgba(224,164,88,0.55)' : 'rgba(255,255,255,0.16)';
    ctx.lineWidth = (!f.sec && D.weak[f.i]) ? 1.1 : 0.5;
    ctx.stroke();
    ctx.restore();
  }
  // schematic bodies: say so on the canvas
  ctx.save(); ctx.font='11px "IBM Plex Mono", ui-monospace, monospace'; ctx.fillStyle='rgba(200,205,220,0.8)'; ctx.textAlign='center';
  for(const [c0, rs] of uncLabels){ ctx.fillText('orbit unknown (schematic)', w/2 + c0[0]*S, cy - c0[1]*S - rs*S*1.6 - 6); }
  ctx.restore();
  // the body centre at the default zoom, where the scale bars are anchored
  const cy0 = topOff > 0 ? cy : Math.max(S/zoom*0.9, (h - panelH)/2);
  drawExtentBars(S, cy0, w, h, panelH);
  drawLC();
}

function interp(arr, p){                // a model sampled evenly over phase 0..1, interpolated
  const x = ((p%1)+1)%1 * (arr.length-1), i = Math.floor(x), f = x - i;
  return arr[i]*(1-f) + arr[Math.min(i+1, arr.length-1)]*f;
}
function dipModel(p, C){ return interp((C || COMPS[0]).model, p); }
function drawLC(){
  if(D.bin && lcView.startsWith('ev')) return drawEvents(+lcView.slice(2));   // drawLC has its own local B (bottom margin)
  if(D.bin && lcView.startsWith('sig')) return drawSignal(+lcView.slice(3));
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

function drawFold(P, F, E, model, mphase, xlab, legL, legR){
  const w=lcv.width/dpr, h=lcv.height/dpr, L=58, Rp=8, T=22, Bm=26;   // T leaves a strip for the legend
  lctx.clearRect(0,0,w,h);
  const mod=[]; for(let k=0;k<=200;k++) mod.push(interp(model, k/200));
  const ok=E.map(e=>e>0);
  const vals=F.filter((f,i)=>ok[i]).map((f,i)=>f).concat(F.map((f,i)=>ok[i]?f-E[i]:mod[0])).concat(F.map((f,i)=>ok[i]?f+E[i]:mod[0])).concat(mod);
  const fmin=Math.min(...vals), fmax=Math.max(...vals), rng=(fmax-fmin)||1, lo=fmin-0.12*rng, hi=fmax+0.12*rng;
  const X=p=>L+p*(w-L-Rp), Y=f=>h-Bm-((f-lo)/(hi-lo))*(h-T-Bm);
  lctx.strokeStyle='#2b3342'; lctx.lineWidth=1;
  lctx.beginPath(); lctx.moveTo(L,T); lctx.lineTo(L,h-Bm); lctx.lineTo(w-Rp,h-Bm); lctx.stroke();
  lctx.fillStyle='#8791a3'; lctx.font='9px ui-monospace,monospace';
  lctx.textAlign='center'; lctx.textBaseline='top';
  for(const t of [0,0.25,0.5,0.75,1]){ const x=X(t); lctx.beginPath(); lctx.moveTo(x,h-Bm); lctx.lineTo(x,h-Bm+4); lctx.stroke(); lctx.fillText(t.toFixed(2),x,h-Bm+6); }
  lctx.textAlign='right'; lctx.textBaseline='middle';
  for(const fv of [lo+(hi-lo)*0.12,(lo+hi)/2,hi-(hi-lo)*0.12]){ const y=Y(fv); lctx.beginPath(); lctx.moveTo(L-4,y); lctx.lineTo(L,y); lctx.stroke(); lctx.fillText(fv.toFixed(3),L-6,y); }
  lctx.fillStyle='#545e70'; lctx.textAlign='center'; lctx.textBaseline='bottom';
  lctx.fillText(xlab,(L+w-Rp)/2,h-1);
  lctx.save(); lctx.translate(11,(T+h-Bm)/2); lctx.rotate(-Math.PI/2); lctx.textAlign='center'; lctx.textBaseline='top';
  lctx.fillText('Residual flux',0,0); lctx.restore();
  lctx.save(); lctx.setLineDash([5,3]); lctx.strokeStyle='#e06c75'; lctx.lineWidth=1.5; lctx.beginPath();
  mod.forEach((f,k)=>{k?lctx.lineTo(X(k/200),Y(f)):lctx.moveTo(X(0),Y(f));}); lctx.stroke(); lctx.restore();
  lctx.strokeStyle='#7c6bb0';
  P.forEach((p,i)=>{ if(!ok[i]) return; lctx.beginPath(); lctx.moveTo(X(p),Y(F[i]-E[i])); lctx.lineTo(X(p),Y(F[i]+E[i])); lctx.stroke(); });
  lctx.fillStyle='#a78bfa';
  P.forEach((p,i)=>{ if(!ok[i]) return; lctx.beginPath(); lctx.arc(X(p),Y(F[i]),2.2,0,2*Math.PI); lctx.fill(); });
  lctx.font='9px ui-monospace,monospace'; lctx.textBaseline='top';
  lctx.textAlign='right'; lctx.fillStyle='#e06c75'; lctx.fillText(legR, w-Rp, 3);
  lctx.textAlign='left'; lctx.fillStyle='#a78bfa'; lctx.fillText(legL, L+4, 3);
  const x=X(mphase);
  lctx.strokeStyle='#e0a458'; lctx.lineWidth=1.4; lctx.beginPath(); lctx.moveTo(x,T); lctx.lineTo(x,h-Bm); lctx.stroke();
  lctx.fillStyle='#e0a458'; lctx.beginPath(); lctx.arc(x, Y(interp(model, mphase)), 3.2, 0, 2*Math.PI); lctx.fill();
}
function drawEvents(k){
  const C = COMPS[k];
  drawFold(C.ev_phase, C.ev_mean, C.ev_err, C.model, compPhase(C), 'Orbital phase (P = '+C.P_orb_hr.toFixed(3)+' h)',
           '\u25CF residuals after the primary, all orbits stacked', '- - two-disc event fit');
}
function drawSignal(j){
  const G = SIGS[j];
  drawFold(G.phase, G.mean, G.err, G.model, sigPhase(G), 'Phase (P = '+G.period_hr.toFixed(4)+' h)',
           '\u25CF residuals after the primary and the events', '- - harmonic fit');
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
function setTime(t){
  tHr = ((t % PORB) + PORB) % PORB;
  phase = ((tHr/P1) % 1 + 1) % 1;
  ophase = COMPS.length ? compPhase(COMPS[0]) : 0;
  slider.value = Math.round(ophase*1000);
  phv.textContent = `t ${tHr.toFixed(2)} h · rotation ${phase.toFixed(2)} · orbit ${ophase.toFixed(2)}`;
}
function setPhase(p){ setTime(tHr + (((p - phase) % 1 + 1.5) % 1 - 0.5) * P1); }   // a rotation step moves the clock
function setMode(m){
  mode=m;
  const lock = m==='locked';
  mFree.setAttribute('aria-pressed', lock?'false':'true');
  mLock.setAttribute('aria-pressed', lock?'true':'false');
  spinBtn.disabled = !lock;
  if(!lock && spinning){ spinning=false; spinBtn.setAttribute('aria-pressed','false');
    spinBtn.innerHTML='&#9654; Play'; }
  hint.innerHTML = lock
    ? 'Snapped to the TESS viewing geometry. Drag horizontally to turn the body about its spin axis &mdash; the lightcurve marker follows. Press Play to rotate at a steady rate.'
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
slider.addEventListener('input', ()=>{ setTime(slider.value/1000*PORB); if(!spinning) draw(); });
// lightcurve toggle: the primary's fold or the mutual events (residuals folded at the orbit)
(function(){
  const lcc = document.getElementById('lc'); if(!B || !lcc) return;
  const row = document.createElement('div'); row.className='row'; row.setAttribute('role','group');
  row.setAttribute('aria-label','Lightcurve view');
  const views = [['Primary rotation','primary']]
    .concat(COMPS.map((C,k)=>[C.label || (['Secondary','Tertiary','Quaternary','Quinary'][k] || ('Body '+(k+2)))+' events', 'ev'+k]))
    .concat(SIGS.map((G,j)=>[G.label || ('Signal '+(j+1)), 'sig'+j]));
  for(const [lab, v] of views){
    const b=document.createElement('button'); b.className='btn'; b.textContent=lab; b.dataset.view=v;
    b.setAttribute('aria-pressed', v===lcView?'true':'false');
    b.addEventListener('click', ()=>{ lcView=v; row.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed', x.dataset.view===v?'true':'false')); draw(); });
    row.appendChild(b);
  }
  lcc.parentNode.insertBefore(row, lcc);
  const lab = document.querySelector('label[for="ph"]'); if(lab) lab.textContent='Time';
})();
spinBtn.addEventListener('click', ()=>{
  if(mode!=='locked') return;
  spinning=!spinning;
  spinBtn.setAttribute('aria-pressed', spinning?'true':'false');
  spinBtn.innerHTML = spinning ? '&#10073;&#10073; Pause' : '&#9654; Play';
});
document.getElementById('reset').addEventListener('click', ()=>{
  orient=qAxis([1,0,0],-1.2); zoom=1; setTime(0); draw();
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
  if(spinning){ setTime(tHr + dt/PERIOD_MS*P1); draw(); }
  requestAnimationFrame(tick);
}
fitPanels(); setTime(0); setColour('albedo'); setMode('locked');
// ---- checks for the mock-up (URL ?t=<hours>&view=events&still=1&debug=1): fixed states for screenshots, and a sweep of
// the orbit reporting where the secondary passes in front of / behind the primary, to compare with the event fold
const QS = new URLSearchParams(location.search);
if(QS.has('t')) setTime(parseFloat(QS.get('t')));
if(QS.has('view') && B){ const v = QS.get('view')==='events' ? 'ev0' : QS.get('view'); lcView=v;
  document.querySelectorAll('[data-view]').forEach(x=>x.setAttribute('aria-pressed', x.dataset.view===v?'true':'false')); }
if(QS.get('debug')==='1' && COMPS.length){
  const R = qMat(D.camQ), rows=[];
  for(let k=0;k<1000;k++){
    const op=k/1000, s=secPos(op, 0);
    const cx=R[0][0]*s[0]+R[0][1]*s[1]+R[0][2]*s[2], cy=R[1][0]*s[0]+R[1][1]*s[1]+R[1][2]*s[2], cz=R[2][0]*s[0]+R[2][1]*s[1]+R[2][2]*s[2];
    rows.push([op, Math.hypot(cx,cy)/Req, cz, dipModel(op, COMPS[0])]);
  }
  // phases where the drawn bodies overlap on the sky, and where the fitted model has any loss
  const B0 = COMPS[0];
  const touch = rows.filter(r=>r[1] < 1+B0.q).map(r=>r[0]);
  const base = Math.max(...B0.model), loss = rows.filter(r=>r[3] < base - 1e-6).map(r=>r[0]);
  const info={drawn_overlap_frac:touch.length/1000, model_loss_frac:loss.length/1000,
    both:rows.filter(r=>r[1] < 1+B0.q && r[3] < base-1e-6).length/1000,
    min_sep_front_Req:Math.min(...rows.filter(r=>r[2]>0).map(r=>r[1])), b_Rp:B0.b_Rp,
    a_Req:SECS[0].a/Req, rs_Req:SECS[0].rs/Req, P1:P1, PORB:PORB, n_companions:COMPS.length, n_signals:SIGS.length};
  document.body.dataset.debug=JSON.stringify(info); console.log('DEBUG '+JSON.stringify(info));
}
// autoplay: the body is rotating as soon as the page opens, which is what makes the shape and
// its lightcurve legible at a glance. setMode('locked') must run FIRST -- it clears `spinning`
// when leaving locked mode, so setting the flag before it would be undone immediately.
spinning = !QS.has('still');
spinBtn.setAttribute('aria-pressed', spinning ? 'true' : 'false');
spinBtn.innerHTML = spinning ? '&#10073;&#10073; Pause' : '&#9654; Play';
draw();
requestAnimationFrame(tick);
window.addEventListener('resize', draw);

}
boot();

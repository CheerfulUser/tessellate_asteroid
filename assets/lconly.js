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

// Lightcurve-only pages: objects with a reliable period but no convex shape model.
// The axis treatment is copied from viewer.js drawLC so these pages read identically to the
// full ones; there is no mesh, no camera and no phase marker, so none of that is loaded.
(async () => {
  const A = window.LCONLY;
  const lc = await fetch(A.lc).then(r => r.json()).catch(() => null);
  const cv = document.getElementById('lc');
  if (!cv || !lc || !lc.phase || !lc.phase.length) return;
  const ctx = cv.getContext('2d');
  let dpr = Math.min(window.devicePixelRatio || 1, 2);
  const P = lc.phase, F = lc.flux, E = lc.err || F.map(() => 0);

  function size() {
    const r = cv.getBoundingClientRect();
    cv.width = Math.max(1, r.width * dpr);
    cv.height = Math.max(1, r.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function draw() {
    const w = cv.width / dpr, h = cv.height / dpr;
    const L = 58, Rp = 8, T = 8, B = 26;
    ctx.clearRect(0, 0, w, h);
    const vals = F.map((f, i) => f - E[i]).concat(F.map((f, i) => f + E[i]));
    const fmin = Math.min(...vals), fmax = Math.max(...vals);
    const rng = (fmax - fmin) || 1, lo = fmin - 0.12 * rng, hi = fmax + 0.12 * rng;
    const X = p => L + p * (w - L - Rp), Y = f => h - B - ((f - lo) / (hi - lo)) * (h - T - B);

    ctx.strokeStyle = '#2b3342'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, T); ctx.lineTo(L, h - B); ctx.lineTo(w - Rp, h - B); ctx.stroke();
    ctx.fillStyle = '#8791a3'; ctx.font = '9px ui-monospace,monospace';

    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      const x = X(t);
      ctx.strokeStyle = '#2b3342'; ctx.beginPath();
      ctx.moveTo(x, h - B); ctx.lineTo(x, h - B + 4); ctx.stroke();
      ctx.fillText(t.toFixed(2), x, h - B + 6);
    }
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const fv of [lo + (hi - lo) * 0.12, (lo + hi) / 2, hi - (hi - lo) * 0.12]) {
      const y = Y(fv);
      ctx.strokeStyle = '#2b3342'; ctx.beginPath();
      ctx.moveTo(L - 4, y); ctx.lineTo(L, y); ctx.stroke();
      ctx.fillText(fv.toFixed(3), L - 6, y);
    }
    ctx.fillStyle = '#545e70';
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText('Rotational phase', (L + w - Rp) / 2, h - 1);
    ctx.save(); ctx.translate(11, (T + h - B) / 2); ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText('Relative flux', 0, 0); ctx.restore();

    ctx.strokeStyle = '#7c6bb0'; ctx.lineWidth = 1;
    P.forEach((p, i) => {
      const e = E[i]; if (!(e > 0)) return;
      const x = X(p);
      ctx.beginPath(); ctx.moveTo(x, Y(F[i] - e)); ctx.lineTo(x, Y(F[i] + e)); ctx.stroke();
    });
    ctx.fillStyle = '#a78bfa';
    P.forEach((p, i) => { ctx.beginPath(); ctx.arc(X(p), Y(F[i]), 1.9, 0, 2 * Math.PI); ctx.fill(); });
  }

  window.addEventListener('resize', () => { size(); draw(); });
  size(); draw();
})();

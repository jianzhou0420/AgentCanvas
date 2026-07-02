// docs-site — small client-side behaviors.
// Pure-vanilla, no deps. Loaded at end of <body> by every page.

(function () {
  // ----- theme toggle -----
  var html = document.documentElement;
  var toggle = document.querySelector('.theme-toggle');
  function applyIcon() {
    if (!toggle) return;
    toggle.textContent = html.getAttribute('data-theme') === 'dark' ? '☀️' : '🌙';
  }
  applyIcon();
  if (toggle) {
    toggle.addEventListener('click', function () {
      var cur = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', cur);
      localStorage.setItem('site-theme', cur);
      applyIcon();
    });
  }

  // ----- mobile nav drawer toggle (button is static; sidebar links wired post-render) -----
  var navToggle = document.querySelector('.mobile-nav-toggle');
  if (navToggle) {
    navToggle.addEventListener('click', function () {
      document.body.classList.toggle('nav-open');
    });
  }

  // ----- client-rendered navigation (top tabs + left sidebar) -----
  // The nav tree is fetched from assets/nav.json (built by _wrap_handwritten.py
  // from _nav.build_nav_data) and rendered here, so adding/removing/renaming a
  // page touches only that page + the JSON — never every sibling's baked chrome.
  // <body> carries data-asset-prefix / data-tab / data-page-rel for resolution.
  (function () {
    var body = document.body;
    var tabsHost = document.getElementById('site-tabs');
    var sideHost = document.getElementById('sidebar-left');
    if (!tabsHost && !sideHost) return; // external-layout page (e.g. AAS) — skip
    var PREFIX = body.getAttribute('data-asset-prefix') || '';
    var activeTab = body.getAttribute('data-tab') || '';
    var pageRel = body.getAttribute('data-page-rel') || '';

    function el(tag, cls, txt) {
      var e = document.createElement(tag);
      if (cls) e.className = cls;
      if (txt != null) e.textContent = txt;
      return e;
    }
    function containsActive(node) {
      if (node.kind === 'page') return node.href === pageRel;
      if (node.href && node.href === pageRel) return true;
      return (node.children || []).some(containsActive);
    }
    function renderNode(node) {
      if (node.kind === 'page') {
        var a = el('a', 'sb-page' + (node.href === pageRel ? ' active' : ''), node.label);
        a.href = PREFIX + node.href;
        return a;
      }
      // section = top-level curated divider (sb-section); group = nested (sb-subsection)
      var isSection = node.kind === 'section';
      var d = el('details', isSection ? 'sb-section' : 'sb-subsection');
      if (node.key) d.setAttribute('data-key', node.key);
      if (containsActive(node)) d.open = true;
      var summary = el('summary', isSection ? 'sb-section-label' : 'sb-subsection-label');
      if (node.href) {
        var link = el('a', 'sb-group-link' + (node.href === pageRel ? ' active' : ''), node.label);
        link.href = PREFIX + node.href;
        summary.appendChild(link);
      } else if (isSection) {
        summary.textContent = node.label;
      } else {
        summary.appendChild(el('span', 'sb-group-label-text', node.label));
      }
      d.appendChild(summary);
      var pages = el('div', 'sb-pages');
      (node.children || []).forEach(function (c) { pages.appendChild(renderNode(c)); });
      d.appendChild(pages);
      return d;
    }
    function renderTabs(data) {
      if (!tabsHost) return;
      tabsHost.innerHTML = '';
      data.tabs.forEach(function (t) {
        var a = el('a', 'tab' + (t.tab === activeTab ? ' active' : ''), t.tab);
        a.href = PREFIX + t.landing;
        tabsHost.appendChild(a);
      });
    }
    function renderSidebar(data) {
      if (!sideHost) return;
      var inner = sideHost.querySelector('.sidebar-inner') || sideHost;
      inner.innerHTML = '';
      var tab = data.tabs.filter(function (t) { return t.tab === activeTab; })[0];
      if (!tab) return;
      tab.nodes.forEach(function (n) { inner.appendChild(renderNode(n)); });
    }
    // Behaviors that depend on the now-rendered sidebar DOM. Mirrors what used to
    // run against baked markup at load time.
    function wireSidebar() {
      // close mobile drawer when a sidebar link is tapped
      document.querySelectorAll('.sidebar-left a').forEach(function (a) {
        a.addEventListener('click', function () { body.classList.remove('nav-open'); });
      });
      // a link inside a <summary> should navigate, not toggle the <details>
      document.querySelectorAll('.sb-subsection > summary .sb-group-link').forEach(function (a) {
        a.addEventListener('click', function (e) { e.stopPropagation(); });
      });
      // persist divider open/closed per session; never fold the active branch
      var STORE = 'sb-open';
      var state = {};
      try { state = JSON.parse(sessionStorage.getItem(STORE) || '{}'); } catch (_) {}
      function persist() { try { sessionStorage.setItem(STORE, JSON.stringify(state)); } catch (_) {} }
      document.querySelectorAll('.sidebar-left details[data-key]').forEach(function (d) {
        var key = d.getAttribute('data-key');
        var hasActive = !!d.querySelector('.active');
        if (key in state) {
          if (state[key]) d.open = true;
          else if (!hasActive) d.open = false;
        }
        d.addEventListener('toggle', function () { state[key] = d.open; persist(); });
      });
    }

    // ----- preserve sidebar scroll across navigation -----
    // The sidebar is re-rendered on every page load, so without this it snaps
    // back to the top each time. We persist .sidebar-left's scrollTop per session
    // and restore it right after the tree is injected. Scroll container is
    // #sidebar-left itself (overflow-y:auto), not .sidebar-inner.
    var SCROLL_KEY = 'sb-scroll';
    function saveScroll() {
      if (!sideHost) return;
      try { sessionStorage.setItem(SCROLL_KEY, String(sideHost.scrollTop)); } catch (_) {}
    }
    function restoreScroll() {
      if (!sideHost) return;
      var v = null;
      try { v = sessionStorage.getItem(SCROLL_KEY); } catch (_) {}
      if (v !== null) sideHost.scrollTop = parseInt(v, 10) || 0;
    }
    if (sideHost) {
      var scrollPending = false;
      sideHost.addEventListener('scroll', function () {
        if (scrollPending) return;
        scrollPending = true;
        requestAnimationFrame(function () { scrollPending = false; saveScroll(); });
      }, { passive: true });
      // Final-position safety net: capture before the document is torn down.
      window.addEventListener('pagehide', saveScroll);
    }

    fetch(PREFIX + 'assets/nav.json')
      .then(function (r) { return r.json(); })
      .then(function (data) {
        renderTabs(data);
        renderSidebar(data);
        wireSidebar();
        // Restore after layout settles so scrollTop actually takes effect.
        requestAnimationFrame(restoreScroll);
      })
      .catch(function () { /* leave shells empty if nav.json is unreachable */ });
  })();

  // ----- right-TOC active-section highlighting via IntersectionObserver -----
  var tocLinks = document.querySelectorAll('.sidebar-right a[href^="#"]');
  if (tocLinks.length && 'IntersectionObserver' in window) {
    var linkMap = {};
    tocLinks.forEach(function (a) { linkMap[a.getAttribute('href').slice(1)] = a; });

    var lastActive = null;
    function activate(id) {
      if (lastActive) lastActive.classList.remove('toc-active');
      var a = linkMap[id];
      if (a) { a.classList.add('toc-active'); lastActive = a; }
    }

    var headings = Object.keys(linkMap)
      .map(function (id) { return document.getElementById(id); })
      .filter(Boolean);

    var observer = new IntersectionObserver(function (entries) {
      // Pick the topmost intersecting heading
      var visible = entries.filter(function (e) { return e.isIntersecting; });
      if (visible.length) {
        visible.sort(function (a, b) { return a.boundingClientRect.top - b.boundingClientRect.top; });
        activate(visible[0].target.id);
      }
    }, { rootMargin: '-80px 0px -70% 0px', threshold: 0 });

    headings.forEach(function (h) { observer.observe(h); });
  }

  // ----- client-side search -----
  // Zero-dependency: a flat index (assets/search-index.json) is baked by
  // _wrap_handwritten.py from every page's title + body text. We fetch it
  // lazily on first open and filter in the browser. Links are resolved
  // against data-asset-prefix on <body> so they work at any folder depth and
  // under any GitHub Pages base path.
  (function () {
    var btn = document.querySelector('.search-btn');
    if (!btn) return;
    var PREFIX = document.body.getAttribute('data-asset-prefix') || '';
    var index = null;            // lazily fetched array of {title,url,text}
    var overlay, input, list;    // built on first open
    var results = [];
    var sel = -1;

    function build() {
      overlay = document.createElement('div');
      overlay.className = 'search-overlay';
      overlay.innerHTML =
        '<div class="search-modal" role="dialog" aria-label="Search">' +
        '<input class="search-input" type="search" placeholder="Search…" aria-label="Search query" autocomplete="off" spellcheck="false">' +
        '<div class="search-results"></div>' +
        '<div class="search-hint">↑↓ navigate · ↵ open · esc close</div>' +
        '</div>';
      document.body.appendChild(overlay);
      input = overlay.querySelector('.search-input');
      list = overlay.querySelector('.search-results');
      overlay.addEventListener('click', function (e) { if (e.target === overlay) close(); });
      input.addEventListener('input', function () { render(input.value); });
      input.addEventListener('keydown', onKey);
    }

    function open() {
      if (!overlay) build();
      overlay.classList.add('open');
      document.body.classList.add('search-open');
      input.value = '';
      render('');
      input.focus();
      if (index === null) {
        fetch(PREFIX + 'assets/search-index.json')
          .then(function (r) { return r.json(); })
          .then(function (data) { index = data; render(input.value); })
          .catch(function () { index = []; render(input.value); });
      }
    }

    function close() {
      if (overlay) overlay.classList.remove('open');
      document.body.classList.remove('search-open');
    }

    function hint(msg) {
      var d = document.createElement('div');
      d.className = 'search-empty';
      d.textContent = msg;
      return d;
    }

    function render(q) {
      q = q.trim().toLowerCase();
      list.innerHTML = '';
      sel = -1;
      results = [];
      if (!q) {
        list.appendChild(hint(index === null ? 'Loading…'
          : 'Type to search ' + index.length + ' page' + (index.length === 1 ? '' : 's') + '.'));
        return;
      }
      if (!index) { list.appendChild(hint('Loading…')); return; }
      results = index.filter(function (e) {
        return e.title.toLowerCase().indexOf(q) !== -1 || e.text.toLowerCase().indexOf(q) !== -1;
      }).slice(0, 20);
      if (!results.length) { list.appendChild(hint('No matches.')); return; }
      results.forEach(function (e, i) { list.appendChild(row(e, q, i)); });
      select(0);
    }

    function row(e, q, i) {
      var a = document.createElement('a');
      a.className = 'search-result';
      a.href = PREFIX + e.url;
      var t = document.createElement('div');
      t.className = 'search-result-title';
      t.textContent = e.title;
      var s = document.createElement('div');
      s.className = 'search-result-snippet';
      s.appendChild(snippet(e.text, q));
      a.appendChild(t);
      a.appendChild(s);
      a.addEventListener('mouseenter', function () { select(i); });
      return a;
    }

    function snippet(text, q) {
      var frag = document.createDocumentFragment();
      var lo = text.toLowerCase().indexOf(q);
      if (lo === -1) {
        frag.appendChild(document.createTextNode(text.slice(0, 120) + (text.length > 120 ? '…' : '')));
        return frag;
      }
      var start = Math.max(0, lo - 40);
      frag.appendChild(document.createTextNode((start > 0 ? '…' : '') + text.slice(start, lo)));
      var mk = document.createElement('mark');
      mk.textContent = text.slice(lo, lo + q.length);
      frag.appendChild(mk);
      var end = lo + q.length;
      frag.appendChild(document.createTextNode(text.slice(end, end + 80) + (end + 80 < text.length ? '…' : '')));
      return frag;
    }

    function select(i) {
      var rows = list.querySelectorAll('.search-result');
      if (!rows.length) { sel = -1; return; }
      sel = (i + rows.length) % rows.length;
      rows.forEach(function (r, j) { r.classList.toggle('sel', j === sel); });
      rows[sel].scrollIntoView({ block: 'nearest' });
    }

    function onKey(e) {
      if (e.key === 'ArrowDown') { e.preventDefault(); select(sel + 1); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); select(sel - 1); }
      else if (e.key === 'Enter') {
        var rows = list.querySelectorAll('.search-result');
        if (rows[sel]) { e.preventDefault(); window.location.href = rows[sel].href; }
      } else if (e.key === 'Escape') { close(); }
    }

    btn.addEventListener('click', open);
    document.addEventListener('keydown', function (e) {
      var open_ = document.body.classList.contains('search-open');
      var typing = /^(INPUT|TEXTAREA|SELECT)$/.test((document.activeElement || {}).tagName || '');
      if (e.key === '/' && !typing && !open_) { e.preventDefault(); open(); }
      else if (e.key === 'Escape' && open_) { close(); }
    });
  })();
})();

/* ---- Mermaid diagrams ------------------------------------------------------
 * Any doc page can drop a `<pre class="mermaid">…</pre>` block and it renders as
 * a diagram. Loaded here (not per-page) because the wrap step extracts inline
 * <main> scripts to body-end on the first pass and then DROPS them on the next
 * re-wrap — so page-local mermaid init is not durable. Gated on the block being
 * present, so pages without a diagram pay zero (no CDN fetch). Theme follows the
 * site's <html data-theme> and re-renders on toggle. */
(function () {
  var blocks = Array.prototype.slice.call(document.querySelectorAll('pre.mermaid'));
  if (!blocks.length) return;
  // Cache each graph's source — mermaid replaces the element's text with SVG on
  // first run, so we need the original to re-render when the theme flips.
  blocks.forEach(function (b) { b.dataset.src = b.textContent; });
  import('https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs')
    .then(function (mod) {
      var mermaid = mod.default;
      function siteTheme() {
        return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'neutral';
      }
      function render() {
        blocks.forEach(function (b) {
          b.removeAttribute('data-processed');
          b.innerHTML = b.dataset.src;
        });
        mermaid.initialize({ startOnLoad: false, theme: siteTheme(), flowchart: { curve: 'basis', useMaxWidth: true } });
        mermaid.run({ nodes: blocks });
      }
      render();
      new MutationObserver(function (muts) {
        muts.forEach(function (m) { if (m.attributeName === 'data-theme') render(); });
      }).observe(document.documentElement, { attributes: true });
    })
    .catch(function (e) { console.error('mermaid load failed', e); });
})();

/* ---- Local changelog injection ---------------------------------------------
 * Per-page Changelog sections are personal dev history and live outside the
 * tracked tree, as gitignored fragments under docs/_changelogs/<page-rel>
 * (a parallel tree mirroring docs/pages/**). On the local dev site we fetch
 * the page's fragment and re-attach it at the tail of <main> plus a right-TOC
 * entry. Localhost-gated: the published site never even issues the fetch —
 * fragments aren't tracked, so there is nothing to find there anyway. */
(function () {
  if (!/^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname)) return;
  var body = document.body;
  var rel = body.getAttribute('data-page-rel') || '';
  var PREFIX = body.getAttribute('data-asset-prefix') || '';
  if (rel.indexOf('pages/') !== 0) return;
  var main = document.querySelector('main.doc-body');
  if (!main) return;
  fetch(PREFIX + '_changelogs/' + rel)
    .then(function (r) { return r.ok ? r.text() : null; })
    .then(function (html) {
      if (!html) return;
      var wrap = document.createElement('div');
      wrap.className = 'local-changelog';
      wrap.innerHTML = '<hr>\n' + html;
      // Same tail slot the section occupied before extraction: ahead of the
      // page-nav if the page has one, else ahead of the in-main footer.
      main.insertBefore(wrap, main.querySelector('nav.page-nav, footer'));
      var h2 = wrap.querySelector('h2[id]');
      var toc = document.querySelector('.sidebar-right .toc-list');
      if (h2 && toc && !toc.querySelector('a[href="#' + h2.id + '"]')) {
        var a = document.createElement('a');
        a.className = 'toc-h2';
        a.href = '#' + h2.id;
        a.textContent = h2.textContent.trim();
        toc.appendChild(a);
      }
    })
    .catch(function () { /* fragment tree absent — nothing to attach */ });
})();

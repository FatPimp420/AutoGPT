/* Offline shell + cached last training/game data.
 * v3: the app SHELL (HTML) is now network-first so a redeploy shows up
 * immediately when online, instead of being pinned to a cached old page.
 * Static assets stay cache-first; data/ stays network-first. */
const SHELL = "tribes-shell-v3";
const DATA = "tribes-data-v3";
const SHELL_FILES = ["./", "index.html", "manifest.webmanifest",
  "icon-192.png", "icon-512.png", "apple-touch-icon.png"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(SHELL).then(c => c.addAll(SHELL_FILES)));
  self.skipWaiting();
});

self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(
    keys.filter(k => k !== SHELL && k !== DATA).map(k => caches.delete(k))
  )).then(() => self.clients.claim()));
});

function networkFirst(e, cacheName) {
  e.respondWith(
    fetch(e.request).then(r => {
      const copy = r.clone();
      caches.open(cacheName).then(c => c.put(e.request, copy));
      return r;
    }).catch(() => caches.match(e.request))
  );
}

self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  const isHTML = e.request.mode === "navigate" ||
    url.pathname.endsWith("/") || url.pathname.endsWith("index.html");
  // Fresh UI and fresh data when online; cached copies when offline.
  if (isHTML) return networkFirst(e, SHELL);
  if (url.pathname.includes("/data/")) return networkFirst(e, DATA);
  // Static assets (icons, manifest): cache-first with background refresh.
  e.respondWith(
    caches.match(e.request).then(hit => {
      const fresh = fetch(e.request).then(r => {
        caches.open(SHELL).then(c => c.put(e.request, r.clone()));
        return r;
      }).catch(() => hit);
      return hit || fresh;
    })
  );
});

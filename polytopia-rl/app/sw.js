/* Offline shell + cached last training data. */
const SHELL = "tribes-shell-v1";
const DATA = "tribes-data-v1";
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

self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (url.pathname.endsWith("/data/data.json")) {
    // network-first: fresh data when online, last synced state when not
    e.respondWith(
      fetch(e.request).then(r => {
        const copy = r.clone();
        caches.open(DATA).then(c => c.put(e.request, copy));
        return r;
      }).catch(() => caches.match(e.request))
    );
  } else {
    // cache-first shell with background refresh
    e.respondWith(
      caches.match(e.request).then(hit => {
        const fresh = fetch(e.request).then(r => {
          caches.open(SHELL).then(c => c.put(e.request, r.clone()));
          return r;
        }).catch(() => hit);
        return hit || fresh;
      })
    );
  }
});

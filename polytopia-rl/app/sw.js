/* Offline shell + cached last training/game data. */
const SHELL = "tribes-shell-v2";
const DATA = "tribes-data-v2";
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
  // Everything under data/ (training payload, game replays, index) is
  // network-first: fresh when online, last-synced copy when offline.
  if (url.pathname.includes("/data/")) {
    e.respondWith(
      fetch(e.request).then(r => {
        const copy = r.clone();
        caches.open(DATA).then(c => c.put(e.request, copy));
        return r;
      }).catch(() => caches.match(e.request))
    );
  } else {
    // cache-first app shell with background refresh
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

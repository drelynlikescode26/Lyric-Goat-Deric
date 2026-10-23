const CACHE = 'lyric-goat-v12-writing-workshop';
const STATIC = [
  '/',
  '/static/style.css',
  '/static/script.js',
  '/static/studio.css',
  '/static/studio-mac.css',
  '/static/manifest.json',
  '/static/icon-192.png',
  '/static/icon-512.png',
];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(STATIC)));
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys =>
    Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
  ));
  self.clients.claim();
});

self.addEventListener('fetch', e => {
  // Always network-first for API calls
  if (e.request.url.includes('/lyrics') || e.request.url.includes('/settings') || e.request.url.includes('/storage') || e.request.url.includes('/compare') || e.request.url.includes('/process') || e.request.url.includes('/regenerate') || e.request.url.includes('/api/')) {
    return;
  }
  e.respondWith(
    fetch(e.request).catch(() => caches.match(e.request))
  );
});

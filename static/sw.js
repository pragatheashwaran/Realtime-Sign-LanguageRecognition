const CACHE_NAME = 'sensisign-v1';
const ASSETS = [
    '/',
    '/static/manifest.json',
    'https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS))
    );
});

self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request).then((response) => response || fetch(event.request))
    );
});

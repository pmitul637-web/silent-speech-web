// Service worker: makes the app open instantly and keep working with no
// connection. Only Claude's replies need the network.

const VERSION = "silent-speech-v5";
const SHELL = [
  "/",
  "/index.html",
  "/style.css",
  "/app.js",
  "/matcher.js",
  "/predict.js",
  "/manifest.webmanifest",
  "/icons/icon-180.png",
  "/icons/icon-192.png",
  "/icons/icon-512.png",
  "/models/face_landmarker.task",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(VERSION).then((cache) =>
      // addAll fails the whole install if any single file 404s, so add them
      // one at a time and tolerate misses.
      Promise.all(SHELL.map((url) =>
        cache.add(url).catch(() => null)
      ))
    ).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;             // /ask is a POST — always live

  const url = new URL(req.url);

  // The MediaPipe bundle and its wasm come from a CDN. Cache them the first
  // time they load, so later starts work with no connection.
  const isCdn = url.hostname === "cdn.jsdelivr.net";

  event.respondWith(
    caches.match(req).then((hit) => {
      if (hit) return hit;
      return fetch(req).then((res) => {
        if (res && (res.ok || res.type === "opaque") && (isCdn || url.origin === self.location.origin)) {
          const copy = res.clone();
          caches.open(VERSION).then((c) => c.put(req, copy)).catch(() => {});
        }
        return res;
      }).catch(() => {
        if (req.mode === "navigate") return caches.match("/index.html");
        throw new Error("offline and not cached");
      });
    })
  );
});

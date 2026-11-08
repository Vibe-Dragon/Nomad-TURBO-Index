// Read a ZIM from a resume point and print article sections as JSON lines.
// Uses the NOMAD image only for libzim and htmlparser2. It does not call NOMAD code.
//
// The meat of a page is indexed no matter how long the page is. Navigation,
// scripts, author boxes, and similar chrome are left out. A long page is
// parsed in slices so one book cannot force a giant DOM into memory. The
// text is still stored in full and chunked later. Do not bring back a size cap.
import { createRequire } from "node:module";
import { availableParallelism } from "node:os";
import { pathToFileURL } from "node:url";
import { setFlagsFromString } from "node:v8";
import { runInNewContext } from "node:vm";
import { isMainThread, parentPort, Worker } from "node:worker_threads";
import {
  appendPath,
  boilerplate,
  classifyEntry,
  skipByPath,
  clearAhead,
  clearGo,
  clearOcrQueue,
  writeOcrExport,
  loadPrints,
  matchesPrint,
  readAhead,
  readList,
  removeAhead,
  resetList,
  svgText,
  waitForGo,
  writeAhead,
  zipFiles,
} from "./zim-extra.mjs";

const require = createRequire("/app/package.json");
const libzim = require("@openzim/libzim");
const { Archive } = libzim;
const { Parser } = require("htmlparser2");

const SKIP_TAGS = new Set([
  "script", "style", "nav", "header", "footer", "noscript", "iframe", "svg",
]);

const SKIP_CLASS = new Set([
  "navbox", "sidebar", "infobox", "mw-editsection", "reference", "reflist",
  "toc", "noprint", "mw-jump-link", "mw-headline-anchor", "navbar", "hatnote",
  "ambox", "sistersitebox", "portal", "geo-nondefault", "authority-control",
]);

const NON_CONTENT = [
  /^see also$/i,
  /^references$/i,
  /^external links$/i,
  /^further reading$/i,
  /^notes$/i,
  /^bibliography$/i,
  /^navigation$/i,
];

// How much HTML to pull from libzim at once. This bounds memory. It is not a
// limit on how much of the page is indexed.
const SLICE = 256 * 1024;
const WORKERS = Math.max(1, Math.min(4, availableParallelism()));

function arg(name, fallback = "") {
  const index = process.argv.indexOf(name);
  if (index === -1 || index + 1 >= process.argv.length) return fallback;
  return process.argv[index + 1];
}

function meta(archive, key) {
  try {
    return archive.getMetadata(key) || "";
  } catch {
    return "";
  }
}

function isArticle(entry) {
  try {
    if (entry.isRedirect) return false;
    const mime = entry.item.mimetype || "";
    return mime === "text/html" || mime === "application/xhtml+xml";
  } catch {
    return false;
  }
}

function blobBuffer(blob) {
  if (!blob) return Buffer.alloc(0);
  if (Buffer.isBuffer(blob)) return blob;
  if (Buffer.isBuffer(blob.data)) return blob.data;
  if (blob.data) return Buffer.from(blob.data);
  return Buffer.from(String(blob));
}

function bytesOf(item) {
  return blobBuffer(item && item.data);
}

function itemSize(item) {
  try {
    const size = Number(item && item.size ? item.size : 0);
    return Number.isFinite(size) && size > 0 ? size : 0;
  } catch {
    return 0;
  }
}

// Half of whatever cap libzim already chose. The lookahead keeps about that
// much decompressed data loaded ahead of the article thread. It does not
// raise the cap: a larger cache on a file that does not reuse clusters fills
// swap without skipping the first read.
export function lookaheadBudget(cacheMax) {
  const max = Number(cacheMax);
  if (!(max > 0) || !Number.isFinite(max)) return 0;
  return Math.floor(max / 2);
}

export function lookaheadShouldWait(aheadBytes, budget) {
  const cap = Number(budget);
  if (!(cap > 0) || !Number.isFinite(cap)) return true;
  return (Number(aheadBytes) || 0) >= cap;
}

export function notePrefetch(state, seq, bytes) {
  const size = Math.max(0, Number(bytes) || 0);
  state.inflight.push({ seq, bytes: size });
  state.ahead += size;
}

export function noteConsumed(state, mainSeq) {
  const done = Number(mainSeq) || 0;
  while (state.inflight.length && state.inflight[0].seq <= done) {
    state.ahead -= state.inflight.shift().bytes;
  }
  if (state.ahead < 0) state.ahead = 0;
  return state.ahead;
}

function clusterCacheBudget() {
  try {
    if (typeof libzim.getClusterCacheMaxSize !== "function") return 0;
    return lookaheadBudget(libzim.getClusterCacheMaxSize());
  } catch {
    return 0;
  }
}

// One byte is enough to make libzim decompress the cluster into its cache.
// The copy is not kept. The article thread reads the page itself.
function touchArticle(item) {
  const size = itemSize(item);
  try {
    if (item && typeof item.getData === "function") item.getData(0, 1);
    else if (item && item.data != null) void item.data;
  } catch {
    // The article thread still reads this page.
  }
  return size;
}

function cutUtf8(buf) {
  if (!buf.length) return [buf, Buffer.alloc(0)];
  let index = buf.length - 1;
  let continued = 0;
  while (index >= 0 && continued < 3 && (buf[index] & 0xc0) === 0x80) {
    index -= 1;
    continued += 1;
  }
  if (index < 0) return [Buffer.alloc(0), buf];
  const lead = buf[index];
  let need = 1;
  if ((lead & 0x80) === 0) need = 1;
  else if ((lead & 0xe0) === 0xc0) need = 2;
  else if ((lead & 0xf0) === 0xe0) need = 3;
  else if ((lead & 0xf8) === 0xf0) need = 4;
  else return [buf, Buffer.alloc(0)];
  if (buf.length - index < need) return [buf.subarray(0, index), buf.subarray(index)];
  return [buf, Buffer.alloc(0)];
}

export function* textSlices(item, slice = SLICE) {
  const size = itemSize(item);
  if (!(size > slice) || typeof item.getData !== "function") {
    yield bytesOf(item).toString("utf8");
    return;
  }
  let offset = 0;
  let pending = Buffer.alloc(0);
  while (offset < size) {
    const count = Math.min(slice, size - offset);
    const chunk = blobBuffer(item.getData(offset, count));
    const step = chunk.length > 0 ? chunk.length : count;
    if (step <= 0) break;
    offset += step;
    if (!chunk.length) continue;
    let buf = pending.length ? Buffer.concat([pending, chunk]) : chunk;
    pending = Buffer.alloc(0);
    if (offset < size) {
      const [head, tail] = cutUtf8(buf);
      buf = head;
      pending = tail;
    }
    if (buf.length) yield buf.toString("utf8");
  }
  if (pending.length) yield pending.toString("utf8");
}

function nonContent(heading) {
  return NON_CONTENT.some((pattern) => pattern.test(heading));
}

function isChrome(name, attribs) {
  if (SKIP_TAGS.has(name)) return true;
  if (String(attribs.role || "").toLowerCase() === "navigation") return true;
  if (String(attribs.id || "") === "coordinates") return true;
  const classes = String(attribs.class || "").split(/\s+/);
  return classes.some((token) => SKIP_CLASS.has(token));
}

function squash(parts) {
  return parts.join(" ").replace(/\s+/g, " ").trim();
}

function createCollector() {
  const sections = [];
  const all = [];
  let current = { heading: "Introduction", level: 2, parts: [], skip: false };
  let skipDepth = 0;
  let heading = null;
  let tableDepth = 0;
  let row = null;
  let cell = null;

  function flush() {
    if (!current.skip) {
      const text = squash(current.parts);
      if (text) sections.push({ heading: current.heading, text, level: current.level });
    }
    current.parts = [];
  }

  const parser = new Parser(
    {
      onopentag(name, attribs) {
        if (skipDepth > 0) {
          skipDepth += 1;
          return;
        }
        if (isChrome(name, attribs || {})) {
          skipDepth = 1;
          return;
        }
        if ((name === "h2" || name === "h3" || name === "h4") && tableDepth === 0) {
          heading = { level: Number(name.slice(1)), buf: [] };
          return;
        }
        if (name === "table") {
          tableDepth += 1;
          if (tableDepth === 1) {
            row = null;
            cell = null;
          }
          return;
        }
        if (tableDepth > 0 && name === "tr") {
          row = [];
          cell = null;
          return;
        }
        if (tableDepth > 0 && (name === "td" || name === "th")) cell = [];
      },
      ontext(text) {
        if (!text || skipDepth > 0) return;
        if (heading) {
          heading.buf.push(text);
          return;
        }
        if (cell) {
          cell.push(text);
          return;
        }
        if (tableDepth > 0) return;
        all.push(text);
        if (!current.skip) current.parts.push(text);
      },
      onclosetag(name) {
        if (skipDepth > 0) {
          skipDepth -= 1;
          return;
        }
        if (heading && (name === "h2" || name === "h3" || name === "h4")) {
          const title = squash(heading.buf).replace(/\[edit\]/gi, "").trim();
          heading = null;
          if (title) all.push(title);
          flush();
          current = {
            heading: title,
            level: Number(name.slice(1)),
            parts: [],
            skip: nonContent(title),
          };
          return;
        }
        if (tableDepth > 0 && (name === "td" || name === "th")) {
          const text = cell ? squash(cell) : "";
          cell = null;
          if (text) {
            all.push(text);
            if (row) row.push(text);
          }
          return;
        }
        if (tableDepth > 0 && name === "tr") {
          if (row && row.length && !current.skip) current.parts.push(row.join(" | "));
          row = null;
          cell = null;
          return;
        }
        if (name === "table" && tableDepth > 0) {
          tableDepth -= 1;
          if (tableDepth === 0) {
            row = null;
            cell = null;
          }
        }
      },
    },
    { decodeEntities: true },
  );

  return {
    write(html) {
      if (html) parser.write(html);
    },
    end() {
      parser.end();
    },
    finish() {
      flush();
      let rich = 0;
      for (const section of sections) {
        if (
          section.heading
          && section.heading !== "Introduction"
          && section.heading.length >= 3
          && section.text.length >= 100
        ) {
          rich += 1;
        }
      }
      // One real section keeps its heading, so that name is on every chunk.
      // A page with none is still one block. A short or chrome heading does
      // not count, and References and the other chrome names stay dropped.
      if (rich < 1) {
        const text = squash(all);
        return text ? [{ heading: "", text, level: 2 }] : [];
      }
      return sections;
    },
  };
}

export function articleFromHtml(raw) {
  const collector = createCollector();
  collector.write(raw);
  collector.end();
  return { sections: collector.finish() };
}

// iterByPath visits every user entry. A map file keeps its HTML pages first
// and then millions of tiles. Each step builds an Entry the V8 heap does not
// account for, so the walk collects them on a fixed count. The entry in hand
// stays live. Articles are not skipped, and a long page is still read in full.
const COLLECT_EVERY = 2048;
let resolvedGc = false;

export function resolveEntryGc() {
  if (resolvedGc !== false) return resolvedGc;
  if (typeof globalThis.gc === "function") {
    resolvedGc = globalThis.gc;
    return resolvedGc;
  }
  try {
    setFlagsFromString("--expose-gc");
    const found = runInNewContext("gc");
    resolvedGc = typeof found === "function" ? found : null;
  } catch {
    resolvedGc = null;
  }
  return resolvedGc;
}

export function collectEntries(visited, gc) {
  const count = Number(visited) || 0;
  if (count < COLLECT_EVERY || count % COLLECT_EVERY !== 0) return false;
  const run = gc === undefined ? resolveEntryGc() : gc;
  if (typeof run !== "function") return false;
  run({ type: "major", execution: "sync" });
  return true;
}

function eventTurn() {
  return new Promise((resolve) => {
    setImmediate(resolve);
  });
}

async function* entries(archive, afterPath) {
  // findByPath in this libzim build returns only that entry, not the tail.
  // One forward walk reaches the checkpoint. The archive then stays open.
  // gc() drops the JS wrapper only. The native entry is freed by a finalizer,
  // and that finalizer runs on a later turn of this thread. A tight walk
  // never takes that turn, so the await is what actually releases the memory.
  let skipping = Boolean(afterPath);
  let visited = 0;
  for (const entry of archive.iterByPath()) {
    visited += 1;
    if (collectEntries(visited)) {
      await eventTurn();
      await eventTurn();
    }
    if (skipping) {
      if (entry.path === afterPath) skipping = false;
      continue;
    }
    yield entry;
  }
  if (skipping) {
    throw new Error(`resume path was not found: ${afterPath}`);
  }
}

async function* selectedArticles(archive, afterPath, skip) {
  let seen = 0;
  const resumeSkip = !afterPath && (Number(skip) || 0) > 0;
  const limitSkip = Number(skip) || 0;
  for await (const entry of entries(archive, afterPath)) {
    if (!isArticle(entry)) continue;
    seen += 1;
    if (resumeSkip && seen <= limitSkip) continue;
    yield entry;
  }
}

async function prefetchArticles({ zimPath, afterPath, skip, budget, seq }) {
  const seqView = new Int32Array(seq);
  const archive = new Archive(zimPath);
  const state = { inflight: [], ahead: 0 };
  let articleSeq = 0;
  for await (const entry of selectedArticles(archive, afterPath || "", skip)) {
    if (Atomics.load(seqView, 1) === 1) return;
    articleSeq += 1;
    while (true) {
      if (Atomics.load(seqView, 1) === 1) return;
      const mainSeq = Atomics.load(seqView, 0);
      noteConsumed(state, mainSeq);
      if (articleSeq <= mainSeq) break;
      if (!lookaheadShouldWait(state.ahead, budget)) break;
      Atomics.wait(seqView, 0, mainSeq, 1000);
    }
    if (articleSeq <= Atomics.load(seqView, 0)) continue;
    let bytes = 1;
    try {
      bytes = Math.max(1, touchArticle(entry.item));
    } catch {
      bytes = 1;
    }
    noteConsumed(state, Atomics.load(seqView, 0));
    if (articleSeq <= Atomics.load(seqView, 0)) continue;
    notePrefetch(state, articleSeq, bytes);
  }
}

function startLookahead(zimPath, afterPath, skip) {
  const budget = clusterCacheBudget();
  if (!(budget > 0)) return null;
  const shared = new SharedArrayBuffer(8);
  const view = new Int32Array(shared);
  const worker = new Worker(new URL(import.meta.url));
  worker.on("error", (error) => {
    console.error(`cluster lookahead stopped: ${error && error.message ? error.message : error}`);
  });
  worker.postMessage({
    prefetch: true,
    zimPath,
    afterPath: afterPath || "",
    skip: Number(skip) || 0,
    budget,
    seq: shared,
  });
  console.error(`cluster lookahead ${budget} bytes`);
  return { worker, view };
}

function publishRead(lookahead, articleSeq) {
  if (!lookahead) return;
  Atomics.store(lookahead.view, 0, articleSeq);
  Atomics.notify(lookahead.view, 0);
}

function stopLookahead(lookahead) {
  if (!lookahead) return;
  Atomics.store(lookahead.view, 1, 1);
  Atomics.notify(lookahead.view, 0);
}

function payloadOf(msg) {
  if (msg.error) throw new Error(msg.error);
  return msg;
}

function startWorkers() {
  const workers = [];
  for (let slot = 0; slot < WORKERS; slot += 1) {
    const worker = new Worker(new URL(import.meta.url));
    const waiters = new Map();
    worker.on("message", (msg) => {
      const queue = waiters.get(msg.id);
      if (!queue || queue.length === 0) return;
      const resolve = queue.shift();
      if (queue.length === 0) waiters.delete(msg.id);
      resolve(msg);
    });
    let seq = 0;
    const post = (body) => new Promise((resolve) => {
      const queue = waiters.get(body.id) || [];
      queue.push(resolve);
      waiters.set(body.id, queue);
      worker.postMessage(body);
    });
    workers.push({
      run(html) {
        const id = `${slot}:${seq}`;
        seq += 1;
        return post({ id, html, end: true }).then(payloadOf);
      },
      open() {
        const id = `${slot}:${seq}`;
        seq += 1;
        return {
          write(html) {
            return post({ id, html, partial: true }).then(payloadOf);
          },
          finish() {
            return post({ id, end: true }).then(payloadOf);
          },
          abort() {
            waiters.delete(id);
            worker.postMessage({ id, abort: true });
          },
        };
      },
      terminate() {
        return worker.terminate();
      },
    });
  }
  return workers;
}

async function streamItem(worker, item) {
  const handle = worker.open();
  try {
    for (const piece of textSlices(item)) {
      await handle.write(piece);
    }
    return await handle.finish();
  } catch (error) {
    handle.abort();
    throw error;
  }
}

async function emit(line) {
  if (process.stdout.write(line)) return;
  await new Promise((resolve) => process.stdout.once("drain", resolve));
}

async function selfTest() {
  const sample = "<html><body><script>nope</script><nav>Menu</nav><div class=\"navbox\">Box</div><p>A short word.</p></body></html>";
  const simple = articleFromHtml(sample);
  const blob = simple.sections.map((section) => section.text).join(" ");
  if (!blob.includes("short word") || blob.includes("nope") || blob.includes("Menu") || blob.includes("Box")) {
    throw new Error(`simple parse failed: ${blob}`);
  }
  const alpha = "a".repeat(120);
  const beta = "b".repeat(120);
  const rich = articleFromHtml(
    `<html><body><h2>Alpha</h2><p>${alpha}</p><h2>Beta</h2><p>${beta}</p><table><tr><td>Left</td><td>Right</td></tr></table></body></html>`,
  );
  if (rich.sections.length < 2) throw new Error(`rich parse failed: ${rich.sections.length}`);
  const richBlob = rich.sections.map((section) => `${section.heading} ${section.text}`).join("\n");
  if (!richBlob.includes("Alpha") || !richBlob.includes(alpha) || !richBlob.includes("Left | Right")) {
    throw new Error(`rich text failed: ${richBlob.slice(0, 200)}`);
  }
  const cites = "c".repeat(150);
  const body = "d".repeat(150);
  const filtered = articleFromHtml(
    `<html><body><p>Intro meat.</p><h2>One</h2><p>${body}</p><h2>References</h2><p>${cites}</p><h2>Two</h2><p>${alpha}</p></body></html>`,
  );
  const filteredBlob = filtered.sections.map((section) => `${section.heading} ${section.text}`).join("\n");
  if (filteredBlob.includes(cites) || !filteredBlob.includes(body) || !filteredBlob.includes("Intro meat") || !filteredBlob.includes(alpha)) {
    throw new Error(`chrome filter failed: ${filteredBlob.slice(0, 240)}`);
  }
  const lone = articleFromHtml(
    `<html><body><p>Intro meat.</p><h2>Ingredients</h2><p>${alpha}</p><table><tr><td>Lamb</td><td>2 lb</td></tr></table></body></html>`,
  );
  const loneBlob = lone.sections.map((section) => `${section.heading} ${section.text}`).join("\n");
  if (!lone.sections.some((section) => section.heading === "Ingredients") || !loneBlob.includes("Lamb | 2 lb") || !loneBlob.includes("Intro meat")) {
    throw new Error(`one section was flattened: ${loneBlob.slice(0, 240)}`);
  }
  const brief = articleFromHtml("<html><body><p>Intro meat.</p><h2>Hi</h2><p>short</p></body></html>");
  if (brief.sections.length !== 1 || brief.sections[0].heading !== "") {
    throw new Error(`a short heading was kept: ${JSON.stringify(brief.sections)}`);
  }
  const pad = "x".repeat(600 * 1024);
  const huge = articleFromHtml(`<html><body><h2>Start</h2><p>${pad}</p><h2>End</h2><p>TAIL_MARKER ${beta}</p></body></html>`);
  const hugeBlob = huge.sections.map((section) => `${section.heading} ${section.text}`).join("\n");
  if (!hugeBlob.includes("TAIL_MARKER") || !hugeBlob.includes("Start") || !hugeBlob.includes(beta)) {
    throw new Error("the tail of a long page was dropped");
  }
  const split = createCollector();
  const splitHtml = `<p>Hello</p><p>${"z".repeat(80)} TAIL_SPLIT</p>`;
  split.write(splitHtml.slice(0, 2));
  split.write(splitHtml.slice(2));
  split.end();
  const splitBlob = split.finish().map((section) => section.text).join(" ");
  if (!splitBlob.includes("Hello") || !splitBlob.includes("TAIL_SPLIT")) {
    throw new Error(`split parse failed: ${splitBlob}`);
  }
  const word = `café ${"y".repeat(40)}`;
  const encoded = Buffer.from(word);
  const fake = {
    size: encoded.length,
    getData(offset, count) {
      const start = Number(offset);
      return { data: encoded.subarray(start, start + Number(count)) };
    },
  };
  const joined = Array.from(textSlices(fake, 8)).join("");
  if (joined !== word) throw new Error(`slice decode failed: ${joined}`);
  const cut = cutUtf8(Buffer.from("café").subarray(0, 4));
  if (cut[1].toString("hex") !== "c3" || cut[0].toString() !== "caf") {
    throw new Error("utf-8 split failed");
  }
  const workers = startWorkers();
  try {
    const parsed = await workers[0].run(sample);
    const workerText = (parsed.sections || []).map((section) => section.text).join(" ");
    if (!workerText.includes("short word") || workerText.includes("nope")) {
      throw new Error(`worker parse failed: ${workerText}`);
    }
    const handle = workers[0].open();
    const streamedHtml = `<p>${"q".repeat(1000)} WORKER_TAIL</p>`;
    await handle.write(streamedHtml.slice(0, 10));
    await handle.write(streamedHtml.slice(10));
    const streamed = await handle.finish();
    const streamedText = (streamed.sections || []).map((section) => section.text).join(" ");
    if (!streamedText.includes("WORKER_TAIL")) {
      throw new Error(`worker stream dropped the tail: ${streamedText.slice(-80)}`);
    }
  } finally {
    await Promise.all(workers.map((worker) => worker.terminate()));
  }
  if (lookaheadBudget(0) !== 0 || lookaheadBudget(512) !== 256) {
    throw new Error("lookahead budget was not half of a positive cap");
  }
  if (lookaheadBudget(536870912) !== 268435456) {
    throw new Error("lookahead budget changed the library-sized cap");
  }
  const gcCalls = [];
  const fakeGc = () => {
    gcCalls.push(gcCalls.length + 1);
  };
  if (collectEntries(0, fakeGc) || collectEntries(2047, fakeGc) || collectEntries(2049, fakeGc)) {
    throw new Error("entry collection ran off its count");
  }
  if (gcCalls.length !== 0) throw new Error("gc ran before 2048 entries");
  if (!collectEntries(2048, fakeGc) || gcCalls.length !== 1) {
    throw new Error("gc did not run at 2048 entries");
  }
  if (!collectEntries(4096, fakeGc) || gcCalls.length !== 2) {
    throw new Error("gc did not run at the next 2048 entries");
  }
  if (collectEntries(2048, null)) throw new Error("entry collection ran without gc");
  const enabled = resolveEntryGc();
  if (typeof enabled !== "function") throw new Error("entry gc could not be enabled");
  enabled();
  let turned = false;
  setImmediate(() => {
    turned = true;
  });
  const walked = [];
  const fakeArchive = {
    iterByPath() {
      return Array.from({ length: 2048 }, (_, index) => ({ path: `p${index}` }));
    },
  };
  for await (const entry of entries(fakeArchive, "")) walked.push(entry.path);
  if (walked.length !== 2048 || walked[2047] !== "p2047") {
    throw new Error(`entry walk count ${walked.length}`);
  }
  if (!turned) throw new Error("entry walk never returned to the event loop");
  if (!lookaheadShouldWait(1, 0) || !lookaheadShouldWait(100, 100) || lookaheadShouldWait(40, 100)) {
    throw new Error("lookahead wait condition is wrong");
  }
  const gap = { inflight: [], ahead: 0 };
  notePrefetch(gap, 1, 40);
  notePrefetch(gap, 2, 60);
  if (gap.ahead !== 100 || !lookaheadShouldWait(gap.ahead, 100)) {
    throw new Error("prefetch bytes were not counted");
  }
  if (noteConsumed(gap, 1) !== 60) throw new Error("consuming one article left the wrong gap");
  if (noteConsumed(gap, 2) !== 0) throw new Error("consuming the gap did not clear it");
  notePrefetch(gap, 3, 10);
  noteConsumed(gap, 9);
  if (gap.ahead !== 0 || gap.inflight.length !== 0) {
    throw new Error("a passed article stayed in the lookahead");
  }
  if (typeof libzim.getClusterCacheMaxSize === "function") {
    const before = Number(libzim.getClusterCacheMaxSize());
    if (lookaheadBudget(before) !== Math.floor(before / 2)) {
      throw new Error("live cache cap did not split in half");
    }
    if (Number(libzim.getClusterCacheMaxSize()) !== before) {
      throw new Error("reading the lookahead budget changed the cache cap");
    }
  }
  console.error("self-test ok");
}

function spillPaths(spillId) {
  const root = "/turbo-state/spill";
  return {
    text: `${root}/${spillId}-text.jsonl`,
    ocr: `${root}/${spillId}-ocr.jsonl`,
    prints: `${root}/${spillId}.prints`,
    go: `${root}/${spillId}.go`,
  };
}

function entryBytes(entry) {
  return bytesOf(entry.item);
}

function textSections(kind, path, bytes) {
  if (kind === "svg" || path.toLowerCase().endsWith(".svg")) {
    const text = svgText(bytes.toString("utf8"));
    return text ? [{ heading: "", text, level: 2 }] : [];
  }
  if (path.toLowerCase().endsWith(".epub") || kind === "epub") {
    const sections = [];
    for (const file of zipFiles(bytes)) {
      if (!/\.(html|xhtml|htm)$/i.test(file.name)) continue;
      const parsed = articleFromHtml(file.data.toString("utf8"));
      sections.push(...(parsed.sections || []));
    }
    return sections;
  }
  if (path.toLowerCase().endsWith(".docx") || kind === "docx") {
    const xml = zipFiles(bytes).find((file) => file.name === "word/document.xml");
    if (!xml) return [];
    const text = xml.data.toString("utf8").replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
    return text ? [{ heading: "", text, level: 2 }] : [];
  }
  const text = bytes.toString("utf8");
  if (boilerplate(path, text)) return [];
  return text.trim() ? [{ heading: "", text, level: 2 }] : [];
}

async function emitArticle(emit, article) {
  await emit(`${JSON.stringify(article)}\n`);
}

async function readLaterEntry(archive, path) {
  const entry = archive.getEntryByPath(path);
  return {
    entry,
    title: (entry && (entry.title || entry.path)) || path,
    bytes: entryBytes(entry),
    mime: (entry && entry.item && entry.item.mimetype) || "",
  };
}

function pruneSpillList(listPath, keep) {
  const known = readList(listPath);
  const kept = known.filter((entryPath) => classifyEntry("", entryPath, 1024 * 1024) === keep);
  if (kept.length === known.length) return;
  resetList(listPath);
  for (const entryPath of kept) appendPath(listPath, entryPath);
  const dropped = known.length - kept.length;
  const name = keep === "ocr" ? "pictures" : "plain text files";
  console.error(`left ${dropped} ${name} out of this ZIM`);
}

function passStart(list, resumePath, passName) {
  if (!resumePath) return 0;
  const at = list.indexOf(resumePath);
  if (at >= 0) return at + 1;
  console.error(`resume path is not in the ${passName} list, reading from the start`);
  return 0;
}

async function runTextPass(archive, list, resumePath, index, emit) {
  const start = passStart(list, resumePath, "text");
  await emitArticle(emit, { type: "pass", pass: "text", total: list.length, done: start });
  await emitArticle(emit, { type: "phase", phase: "text" });
  for (const path of list.slice(start)) {
    index += 1;
    let sections = [];
    let problem = "";
    let title = path;
    try {
      const got = await readLaterEntry(archive, path);
      title = got.title;
      const kind = got.mime.includes("epub") ? "epub" : got.mime.includes("wordprocessingml") ? "docx" : "";
      sections = textSections(kind, path, got.bytes);
      if (!sections.length) problem = "empty";
    } catch (error) {
      problem = "read";
      console.error(`skip read ${path}: ${error.message || error}`);
    }
    const article = { type: "article", index, path, title, sections };
    if (problem) article.skip = problem;
    await emitArticle(emit, article);
  }
  return index;
}

async function cachedOcrArticle(spillId, path, ahead) {
  if (!spillId) return null;
  let ready = readAhead(spillId, path);
  if (ready) return ready;
  if (ahead && ahead.current === path && ahead.done.has(path)) {
    await ahead.done.get(path);
    ready = readAhead(spillId, path);
    if (ready) return ready;
  }
  // This file is not ready and not the one being read. Stop starting more,
  // and let this pass read the rest. The file already open still finishes.
  if (ahead && ahead.enabled) ahead.closed = true;
  return null;
}

function emitCachedOcr(cached, index, path) {
  const title = cached.title || path;
  if (cached.skip) {
    return { type: "article", index, path, title, sections: [], skip: cached.skip };
  }
  if (cached.kind === "image") {
    return {
      type: "article", index, path, title, kind: "image", ocr: cached.ocr || "", sections: [],
    };
  }
  return {
    type: "article",
    index,
    path,
    title,
    kind: "pdf",
    pages: cached.pages || [],
    sections: [],
  };
}

async function runOcrPass(archive, list, resumePath, index, emit, quality, prints, spillId, ahead) {
  const ocr = await import("/app/nomad-turbo-ocr.mjs");
  const profile = ocr.qualityProfile(quality);
  const start = passStart(list, resumePath, "ocr");
  await emitArticle(emit, { type: "pass", pass: "ocr", total: list.length, done: start });
  for (const path of list.slice(start)) {
    index += 1;
    const cached = await cachedOcrArticle(spillId, path, ahead);
    if (cached && (cached.skip || cached.kind === "pdf" || cached.kind === "image")) {
      const title = cached.title || path;
      const plain = cached.kind === "image"
        ? ""
        : (cached.pages || []).map((page) => page.layer || page.text || "").join("\n");
      if (cached.skip === "duplicate" || (cached.kind !== "image" && matchesPrint(plain, prints))) {
        await emitArticle(emit, { type: "phase", phase: "text" });
        await emitArticle(emit, {
          type: "article", index, path, title, sections: [], skip: "duplicate",
        });
        removeAhead(spillId, path);
        continue;
      }
      if (!cached.skip) {
        const drew = cached.kind === "image" || (cached.pages || []).some((page) => page.drawn);
        await emitArticle(emit, { type: "phase", phase: drew ? "ocr" : "text" });
      }
      await emitArticle(emit, emitCachedOcr(cached, index, path));
      removeAhead(spillId, path);
      if (ahead && ahead.done) ahead.done.delete(path);
      continue;
    }
    let title = path;
    try {
      const got = await readLaterEntry(archive, path);
      title = got.title;
      const kind = classifyEntry(got.mime, path, got.bytes.length);
      if (kind !== "ocr") {
        await emitArticle(emit, { type: "article", index, path, title, sections: [], skip: "empty" });
        continue;
      }
      const isPdf = got.mime.includes("pdf") || path.toLowerCase().endsWith(".pdf");
      if (isPdf) {
        const layer = await ocr.pdfLayer(got.bytes);
        const plain = (layer.pages || []).map((page) => page.text || "").join("\n");
        if (matchesPrint(plain, prints)) {
          await emitArticle(emit, { type: "phase", phase: "text" });
          await emitArticle(emit, {
            type: "article", index, path, title, sections: [], skip: "duplicate",
          });
          continue;
        }
        const willDraw = profile.drawFilled
          || (layer.pages || []).some((page) => !String(page.text || "").trim());
        await emitArticle(emit, { type: "phase", phase: willDraw ? "ocr" : "text" });
        const parts = await ocr.readPdf(got.bytes, profile);
        await emitArticle(emit, {
          type: "article",
          index,
          path,
          title,
          kind: "pdf",
          pages: parts.pages || [],
          sections: [],
        });
        continue;
      }
      await emitArticle(emit, { type: "phase", phase: "ocr" });
      const seen = await ocr.readImage(got.bytes, profile);
      await emitArticle(emit, {
        type: "article", index, path, title, kind: "image", ocr: seen, sections: [],
      });
    } catch (error) {
      console.error(`skip read ${path}: ${error.message || error}`);
      await emitArticle(emit, { type: "article", index, path, title, sections: [], skip: "read" });
    }
  }
  return index;
}

async function main() {
  if (process.argv.includes("--self-test")) {
    await selfTest();
    return;
  }
  const zimPath = arg("--zim");
  const skip = Number(arg("--skip", "0")) || 0;
  const afterPath = arg("--after-path");
  const limit = Number(arg("--limit", "0")) || 0;
  let index = Number(arg("--index-start", "0")) || 0;
  if (!zimPath) {
    console.error("missing --zim");
    process.exit(2);
  }

  const archive = new Archive(zimPath);
  await emit(
    JSON.stringify({
      type: "meta",
      totalArticles: archive.articleCount,
      title: meta(archive, "Title") || meta(archive, "Name"),
      creator: meta(archive, "Creator"),
      publisher: meta(archive, "Publisher"),
      date: meta(archive, "Date"),
      language: meta(archive, "Language"),
      description: meta(archive, "Description"),
    }) + "\n",
  );
  // Date lookup only. Do not walk articles, and do not share the extract container.
  if (process.argv.includes("--meta-only")) {
    return;
  }

  const workers = startWorkers();
  let lookahead = null;
  try {
    lookahead = startLookahead(zimPath, afterPath, skip);
  } catch (error) {
    console.error(`cluster lookahead stopped: ${error && error.message ? error.message : error}`);
  }
  let cursor = 0;
  const flight = [];
  let emitted = 0;
  let articleSeq = 0;
  const wantedPass = arg("--pass", "html") || "html";
  const spillId = arg("--spill-id");
  const resumeLater = arg("--resume-path");
  const quality = arg("--quality", "medium") || "medium";
  const lists = spillId ? spillPaths(spillId) : null;
  const queueOutside = process.argv.includes("--ocr-queue") && Boolean(lists) && wantedPass === "html";
  if (lists && wantedPass === "html" && !afterPath) {
    resetList(lists.text);
    resetList(lists.ocr);
    clearAhead(spillId);
    if (queueOutside) clearOcrQueue(spillId);
  }
  if (lists) {
    pruneSpillList(lists.text, "text");
    pruneSpillList(lists.ocr, "ocr");
  }
  // libzim is not reentrant. This read uses a second archive, one file at a
  // time, on this thread, and only between turns of the walk. Results go to
  // disk. The walk's archive is left alone.
  const ahead = {
    enabled: process.argv.includes("--ocr-ahead") && !queueOutside && Boolean(lists) && wantedPass === "html",
    closed: false,
    chain: Promise.resolve(),
    archive: null,
    current: "",
    done: new Map(),
    ocr: null,
  };
  let seenHtml = 0;
  let ocrCount = 0;
  const resumeSkip = !afterPath && skip > 0;

  async function enqueueZimPdf(entry, entryPath) {
    // The walk's archive is free on this turn. Copy the PDF out and let the
    // OCR queue read it. This walk keeps emitting pages.
    try {
      let bytes;
      let title = entryPath;
      if (entry) {
        bytes = entryBytes(entry);
        title = entry.title || entryPath;
      } else {
        const got = await readLaterEntry(archive, entryPath);
        bytes = got.bytes;
        title = got.title || entryPath;
        got.entry = null;
      }
      if (!bytes || !bytes.length) {
        console.error(`ocr enqueue skip ${entryPath}: empty`);
        return;
      }
      const file = writeOcrExport(spillId, entryPath, bytes);
      await emitArticle(emit, {
        type: "ocr-enqueue",
        path: entryPath,
        file,
        title,
        kind: "pdf",
      });
    } catch (error) {
      console.error(`ocr enqueue skip ${entryPath}: ${error && error.message ? error.message : error}`);
    }
  }

  function queueOcrAhead(entryPath) {
    if (!ahead.enabled || !entryPath || ahead.closed) return;
    let resolveDone = () => {};
    const finished = new Promise((resolve) => {
      resolveDone = resolve;
    });
    ahead.done.set(entryPath, finished);
    ahead.chain = ahead.chain.then(async () => {
      if (ahead.closed) return;
      ahead.current = entryPath;
      try {
        await readZimEntryAhead(entryPath);
      } catch (error) {
        console.error(`ocr ahead skip ${entryPath}: ${error && error.message ? error.message : error}`);
      } finally {
        if (ahead.current === entryPath) ahead.current = "";
        resolveDone();
      }
    });
  }

  async function readZimEntryAhead(entryPath) {
    if (readAhead(spillId, entryPath)) return;
    if (!ahead.archive) {
      try {
        ahead.archive = new Archive(zimPath);
      } catch (error) {
        ahead.closed = true;
        console.error(`ocr ahead skip ${entryPath}: ${error && error.message ? error.message : error}`);
        return;
      }
    }
    let got;
    try {
      got = await readLaterEntry(ahead.archive, entryPath);
    } catch (error) {
      console.error(`ocr ahead skip ${entryPath}: ${error && error.message ? error.message : error}`);
      return;
    }
    const title = got.title || entryPath;
    const bytes = got.bytes;
    const mime = got.mime;
    got.entry = null;
    const kind = classifyEntry(mime, entryPath, bytes.length);
    if (kind !== "ocr") {
      writeAhead(spillId, entryPath, { skip: "empty", title });
      return;
    }
    const isPdf = mime.includes("pdf") || entryPath.toLowerCase().endsWith(".pdf");
    try {
      if (!ahead.ocr) ahead.ocr = import("/app/nomad-turbo-ocr.mjs");
      const ocr = await ahead.ocr;
      const profile = ocr.qualityProfile(quality);
      if (isPdf) {
        const layer = await ocr.pdfLayer(bytes);
        const plain = (layer.pages || []).map((page) => page.text || "").join("\n");
        if (matchesPrint(plain, loadPrints(lists.prints))) {
          writeAhead(spillId, entryPath, { skip: "duplicate", title });
          return;
        }
        const parts = await ocr.readPdf(bytes, profile);
        writeAhead(spillId, entryPath, { kind: "pdf", title, pages: parts.pages || [] });
        return;
      }
      const seen = await ocr.readImage(bytes, profile);
      writeAhead(spillId, entryPath, { kind: "image", title, ocr: seen || "" });
    } catch (error) {
      console.error(`ocr ahead skip ${entryPath}: ${error && error.message ? error.message : error}`);
    }
  }

  async function flushOne() {
    const job = flight.shift();
    let parsed = { sections: [] };
    let skipReason = job.problem || "";
    try {
      parsed = await job.done;
    } catch (error) {
      skipReason = skipReason || "parse";
      console.error(`skip parse ${job.path}: ${error.message || error}`);
    }
    if (parsed.error) {
      skipReason = skipReason || "parse";
      console.error(`skip parse ${job.path}: ${parsed.error}`);
    }
    if (!skipReason && !(parsed.sections || []).length) skipReason = "empty";
    const article = {
      type: "article",
      index: job.index,
      path: job.path,
      title: job.title,
      sections: parsed.sections || [],
    };
    if (skipReason) article.skip = skipReason;
    await emit(JSON.stringify(article) + "\n");
    emitted += 1;
  }

  try {
    if (wantedPass === "html") {
      await emitArticle(emit, { type: "phase", phase: "html" });
      if (lists) {
        const known = readList(lists.ocr);
        ocrCount = known.length;
        if (ocrCount) {
          await emitArticle(emit, { type: "ocr-tally", total: ocrCount });
        }
        if (queueOutside) {
          for (const entryPath of known) {
            await enqueueZimPdf(null, entryPath);
          }
        } else if (ahead.enabled) {
          for (const entryPath of known) {
            if (!readAhead(spillId, entryPath)) queueOcrAhead(entryPath);
          }
        }
      }
    }
    for await (const entry of (wantedPass === "html" ? entries(archive, afterPath) : [])) {
      let earlyPath = "";
      let earlyMime = "";
      let earlySize = 0;
      let earlyRedirect = false;
      try {
        earlyPath = entry.path || "";
        if (skipByPath(earlyPath)) continue;
        earlyRedirect = Boolean(entry.isRedirect);
        earlyMime = (entry.item && entry.item.mimetype) || "";
        earlySize = itemSize(entry.item);
      } catch {
        continue;
      }
      const earlyKind = earlyRedirect ? "skip" : classifyEntry(earlyMime, earlyPath, earlySize);
      if (earlyKind === "text" && lists) {
        appendPath(lists.text, earlyPath);
        continue;
      }
      if (earlyKind === "ocr" && lists) {
        appendPath(lists.ocr, earlyPath);
        ocrCount += 1;
        await emitArticle(emit, { type: "ocr-tally", total: ocrCount });
        if (queueOutside) await enqueueZimPdf(entry, earlyPath);
        else if (ahead.enabled) queueOcrAhead(earlyPath);
        continue;
      }
      if (earlyKind !== "html") continue;
      seenHtml += 1;
      if (resumeSkip && seenHtml <= skip) continue;
      articleSeq += 1;
      index += 1;
      let raw = "";
      let path = "";
      let title = "";
      let problem = "";
      let large = null;
      try {
        path = entry.path || "";
        title = entry.title || path;
        const size = itemSize(entry.item);
        if (size > SLICE) large = entry.item;
        else raw = bytesOf(entry.item).toString("utf8");
      } catch (error) {
        problem = "read";
        console.error(`skip read ${path}: ${error.message || error}`);
      }
      const worker = workers[cursor % workers.length];
      cursor += 1;
      let done;
      if (problem) {
        done = Promise.resolve({ sections: [] });
      } else if (large) {
        while (flight.length) await flushOne();
        try {
          done = Promise.resolve(await streamItem(worker, large));
        } catch (error) {
          try {
            done = worker.run(bytesOf(large).toString("utf8"));
          } catch (again) {
            problem = "read";
            console.error(`skip read ${path}: ${error.message || error}`);
            done = Promise.resolve({ sections: [] });
          }
        }
      } else {
        done = raw ? worker.run(raw) : Promise.resolve({ sections: [] });
      }
      publishRead(lookahead, articleSeq);
      flight.push({
        index,
        path,
        title,
        problem,
        done,
      });
      if (flight.length >= workers.length) await flushOne();
      if (limit && emitted >= limit) break;
    }
    while (flight.length && !(limit && emitted >= limit)) await flushOne();
    if (lists && !limit) {
      if (wantedPass === "html") {
        await emitArticle(emit, {
          type: "ocr-tally",
          total: readList(lists.ocr).length,
          complete: true,
        });
      }
      const resumeText = wantedPass === "text" ? resumeLater : "";
      const resumeOcr = wantedPass === "ocr" ? resumeLater : "";
      if (wantedPass === "html" || wantedPass === "text") {
        const textRows = readList(lists.text);
        if (textRows.length) {
          index = await runTextPass(archive, textRows, resumeText, index, emit);
        }
      }
      if (wantedPass !== "done" && queueOutside) {
        const queued = readList(lists.ocr).length;
        if (queued) {
          console.error(
            `${queued} PDF${queued === 1 ? "" : "s"} from this ZIM went to the front of the OCR queue`,
          );
        }
      } else if (wantedPass !== "done") {
        const ocrRows = readList(lists.ocr);
        if (ocrRows.length) {
          clearGo(lists.go);
          await emitArticle(emit, { type: "hold", pass: "ocr" });
          await waitForGo(lists.go);
          index = await runOcrPass(
            archive,
            ocrRows,
            resumeOcr,
            index,
            emit,
            quality,
            loadPrints(lists.prints),
            spillId,
            ahead,
          );
        }
      }
    }
    ahead.closed = true;
    await ahead.chain;
    await emit(JSON.stringify({ type: "done", articles: index }) + "\n");
  } catch (error) {
    ahead.closed = true;
    throw error;
  } finally {
    ahead.closed = true;
    try {
      await ahead.chain;
    } catch (error) {
      console.error(`ocr ahead skip: ${error && error.message ? error.message : error}`);
    }
    stopLookahead(lookahead);
    const closing = workers.map((worker) => worker.terminate());
    if (lookahead) closing.push(lookahead.worker.terminate());
    await Promise.all(closing);
  }
}

const launchedDirectly = process.argv[1]
  && import.meta.url === pathToFileURL(process.argv[1]).href;

if (!isMainThread) {
  const open = new Map();
  let prefetching = false;
  parentPort.on("message", (msg) => {
    if (msg && msg.prefetch) {
      if (prefetching) return;
      prefetching = true;
      prefetchArticles(msg).catch((error) => {
        console.error(`cluster lookahead stopped: ${error && error.message ? error.message : error}`);
      });
      return;
    }
    if (msg.abort) {
      open.delete(msg.id);
      return;
    }
    try {
      let job = open.get(msg.id);
      if (!job) {
        job = createCollector();
        open.set(msg.id, job);
      }
      if (msg.html) job.write(msg.html);
      if (msg.end || !msg.partial) {
        job.end();
        open.delete(msg.id);
        parentPort.postMessage({ id: msg.id, sections: job.finish() });
      } else {
        parentPort.postMessage({ id: msg.id, more: true });
      }
    } catch (error) {
      open.delete(msg.id);
      parentPort.postMessage({ id: msg.id, sections: [], error: String(error) });
    }
  });
} else if (launchedDirectly) {
  main().catch((error) => {
    console.error(error && error.stack ? error.stack : error);
    process.exit(1);
  });
}

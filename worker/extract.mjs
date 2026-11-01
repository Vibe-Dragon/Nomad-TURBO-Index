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
import { isMainThread, parentPort, Worker } from "node:worker_threads";

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
      if (rich < 2) {
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

function* entries(archive, afterPath) {
  // findByPath in this libzim build returns only that entry, not the tail.
  // One forward walk reaches the checkpoint. The archive then stays open.
  let skipping = Boolean(afterPath);
  for (const entry of archive.iterByPath()) {
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

function* selectedArticles(archive, afterPath, skip) {
  let seen = 0;
  const resumeSkip = !afterPath && (Number(skip) || 0) > 0;
  const limitSkip = Number(skip) || 0;
  for (const entry of entries(archive, afterPath)) {
    if (!isArticle(entry)) continue;
    seen += 1;
    if (resumeSkip && seen <= limitSkip) continue;
    yield entry;
  }
}

function prefetchArticles({ zimPath, afterPath, skip, budget, seq }) {
  const seqView = new Int32Array(seq);
  const archive = new Archive(zimPath);
  const state = { inflight: [], ahead: 0 };
  let articleSeq = 0;
  for (const entry of selectedArticles(archive, afterPath || "", skip)) {
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
    for (const entry of selectedArticles(archive, afterPath, skip)) {
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
    await emit(JSON.stringify({ type: "done", articles: index }) + "\n");
  } finally {
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
      try {
        prefetchArticles(msg);
      } catch (error) {
        console.error(`cluster lookahead stopped: ${error && error.message ? error.message : error}`);
      }
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

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
const { Archive } = require("@openzim/libzim");
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

  const workers = startWorkers();
  let cursor = 0;
  const flight = [];
  let seen = 0;
  let emitted = 0;

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
    for (const entry of entries(archive, afterPath)) {
      if (!isArticle(entry)) continue;
      seen += 1;
      if (!afterPath && seen <= skip) continue;
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
    await Promise.all(workers.map((worker) => worker.terminate()));
  }
}

const launchedDirectly = process.argv[1]
  && import.meta.url === pathToFileURL(process.argv[1]).href;

if (!isMainThread) {
  const open = new Map();
  parentPort.on("message", (msg) => {
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

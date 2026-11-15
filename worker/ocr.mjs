// Read one uploaded PDF or image with the OCR libraries in the NOMAD image.
// This does not call NOMAD's embedder. Turbo chunks the text and embeds it.
//
// Every English PDF page keeps its text layer. A page that is clearly not
// English is left out. A file with no English page is not drawn and not
// returned. A scan with no text layer is judged on its first page. A page
// that already has text is left as that text unless --ocr-every-pdf is set.
// A page with no text is drawn either way. OCR words the text layer already
// covers are not stored again. Pages are read together, one Tesseract worker for each
// hardware thread after two threads are left free. A worker finishes the page
// it is on and waits while other programs are using the CPU, then takes a
// thread back when that work is gone. The process also runs at a lower
// priority, so a short request is served during a page.
// A budget file can lower that cap while Qdrant is writing. The file is
// missing when nothing else asked for the cores. NOMAD_TURBO_OCR_SOCKET, when
// set, sends the drawn page to the GPU reader. Two failures use Tesseract.
// pdf2pic draws a page at 768x512 unless it is told the size, so the reader
// used to ask GraphicsMagick for that size and then draw the page. That read
// every page twice. The size now comes from the PDF page box, and GraphicsMagick
// draws each page once from the page box. Low is 200 dpi with a 4000 px cap.
// Medium is 300 dpi and 8000 px. High is 400 dpi and 12000 px. Only Low still
// shrinks the width to 2000. sharp then does grayscale, normalize, and sharpen.
// A picture uses the same quality. A picture that is not English is read
// once and then left out. The traineddata file is the same one tesseract.js
// loads for that call. Do not download another at run time.
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { connect } from "node:net";
import { availableParallelism, setPriority, tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const require = createRequire("/app/package.json");
const sharp = require("sharp");
const gm = require("gm");
const { PDFParse } = require("pdf-parse");
const { createWorker, OEM } = require("tesseract.js");

// Medium is the everyday draw. High keeps tiny labels. The long side stops
// so one page cannot fill memory. drawFilled follows --ocr-every-pdf and
// stays off unless that flag is set, at every quality.
export function qualityProfile(name, drawFilled = false) {
  const key = String(name || "medium").toLowerCase();
  const draw = Boolean(drawFilled);
  if (key === "low") return { name: "low", dpi: 200, cap: 4000, shrink: true, drawFilled: draw };
  if (key === "high") return { name: "high", dpi: 400, cap: 12000, shrink: false, drawFilled: draw };
  return { name: "medium", dpi: 300, cap: 8000, shrink: false, drawFilled: draw };
}

let activeProfile = qualityProfile("medium");

// Linux USER_HZ. /proc tick counters use this, not the wall clock.
const CPU_HZ = 100;
// How often the host's other CPU use is sampled while pages are in flight.
const CPU_SAMPLE_MS = 2000;
// nice(1) value. Higher yields to ordinary host programs. Idle cores still
// run OCR at full speed, because a lower priority only matters while someone
// else is runnable.
const OCR_NICE = 10;

// Two threads stay free while the host is quiet. Programs that are actually
// running take their cores from the pool. A fractional core rounds to the
// nearest thread, so a quiet HTTP server does not park a worker.
export function ocrSlots(logicalCpus, otherCores) {
  const cpus = Math.max(1, Math.floor(Number(logicalCpus) || 1));
  const reserve = Math.min(2, Math.max(0, cpus - 1));
  const others = Number(otherCores);
  const busy = Number.isFinite(others) && others > 0 ? Math.round(others) : 0;
  return Math.max(0, cpus - Math.max(reserve, busy));
}

// The budget file is a ceiling. A missing file leaves the host sample alone.
// An empty file is not zero: the writer may be between updates.
export function clampSlots(slots, budgetText) {
  const current = Math.max(0, Math.floor(Number(slots) || 0));
  if (budgetText == null) {
    return current;
  }
  const text = String(budgetText).trim();
  if (!text) {
    return current;
  }
  const budget = Number(text);
  if (!Number.isFinite(budget)) {
    return current;
  }
  return Math.max(0, Math.min(current, Math.floor(budget)));
}

function readBudget() {
  const path = process.env.NOMAD_TURBO_OCR_BUDGET || "";
  if (!path) {
    return null;
  }
  try {
    return readFileSync(path, "utf8");
  } catch {
    return null;
  }
}

function recognizeSocket(socketPath, buffer) {
  return new Promise((resolve, reject) => {
    const client = connect(socketPath);
    let header = Buffer.alloc(0);
    let need = null;
    let settled = false;
    const finish = (err, text) => {
      if (settled) {
        return;
      }
      settled = true;
      client.destroy();
      if (err) {
        reject(err instanceof Error ? err : new Error(String(err)));
      } else {
        resolve(text || "");
      }
    };
    client.setTimeout(120000, () => finish(new Error("gpu ocr timed out")));
    client.on("error", (err) => finish(err));
    client.on("data", (data) => {
      header = Buffer.concat([header, data]);
      if (need === null) {
        if (header.length < 4) {
          return;
        }
        need = header.readUInt32BE(0);
        header = header.subarray(4);
      }
      if (need > 8 * 1024 * 1024) {
        finish(new Error("gpu ocr reply is too large"));
        return;
      }
      if (header.length < need) {
        return;
      }
      try {
        const parsed = JSON.parse(header.subarray(0, need).toString("utf8"));
        if (parsed.error) {
          finish(new Error(parsed.error));
        } else {
          finish(null, parsed.text || "");
        }
      } catch (err) {
        finish(err);
      }
    });
    client.on("connect", () => {
      const head = Buffer.alloc(4);
      head.writeUInt32BE(buffer.length);
      client.write(Buffer.concat([head, buffer]));
    });
  });
}

let pictureNoted = false;

function readerNote() {
  const reader = process.env.NOMAD_TURBO_OCR_READER || "";
  const device = process.env.NOMAD_TURBO_OCR_DEVICE || "";
  if (reader && device) {
    return `${reader} on ${device}`;
  }
  if (process.env.NOMAD_TURBO_OCR_SOCKET) {
    return "the GPU reader";
  }
  return "Tesseract on the CPU";
}

function poolLine(shown, cpus, other) {
  const host = other >= 0.5 ? `, other cpu ${other.toFixed(1)}` : "";
  if (process.env.NOMAD_TURBO_OCR_SOCKET) {
    return `ocr draws ${shown} pages at a time${host}. ${readerNote()} reads one page at a time`;
  }
  return `ocr workers ${shown} of ${cpus} threads${host}. ${readerNote()}`;
}

// GPU first, when a socket is configured. Two failures stay on Tesseract.
async function recognizePrepared(buffer, shrink, runner, pool) {
  const prepared = await preprocess(buffer, shrink);
  const socket = process.env.NOMAD_TURBO_OCR_SOCKET || "";
  if (socket && pool.gpuFailures < 2) {
    try {
      return await recognizeSocket(socket, prepared);
    } catch (err) {
      pool.gpuFailures += 1;
      console.error(`gpu ocr failed: ${err && err.message ? err.message : err}`);
    }
  }
  if (!pool.workers) {
    pool.workers = await startWorkers(Math.max(1, pool.ceiling));
    if (socket) {
      console.error("ocr using Tesseract on the CPU");
    }
  }
  const worker = pool.workers[runner % pool.workers.length];
  const result = await worker.recognize(prepared);
  return (result.data && result.data.text) || "";
}

// user + nice + system + irq + softirq. Idle and iowait are not a program
// waiting for a core.
export function busyTicks(statText) {
  const line = String(statText || "").split("\n").find((item) => item.startsWith("cpu "));
  if (!line) {
    return 0;
  }
  const parts = line.trim().split(/\s+/).slice(1).map(Number);
  return (parts[0] || 0) + (parts[1] || 0) + (parts[2] || 0) + (parts[5] || 0) + (parts[6] || 0);
}

// utime + stime + cutime + cstime. cutime picks up a GraphicsMagick child
// after Node has waited for it, so a render that exits between samples is
// still counted as OCR time.
export function procTicks(statText) {
  const text = String(statText || "");
  const end = text.lastIndexOf(")");
  if (end < 0) {
    return { ppid: 0, ticks: 0 };
  }
  const rest = text.slice(end + 2).trim().split(/\s+/);
  return {
    ppid: Number(rest[1]) || 0,
    ticks: (Number(rest[11]) || 0) + (Number(rest[12]) || 0) + (Number(rest[13]) || 0) + (Number(rest[14]) || 0),
  };
}

export function treeTicks(records, rootPid) {
  const children = new Map();
  const ticks = new Map();
  for (const row of records) {
    ticks.set(row.pid, row.ticks);
    if (!children.has(row.ppid)) {
      children.set(row.ppid, []);
    }
    children.get(row.ppid).push(row.pid);
  }
  let sum = 0;
  const stack = [rootPid];
  const seen = new Set();
  while (stack.length) {
    const pid = stack.pop();
    if (seen.has(pid)) {
      continue;
    }
    seen.add(pid);
    sum += ticks.get(pid) || 0;
    for (const child of children.get(pid) || []) {
      stack.push(child);
    }
  }
  return sum;
}

// Cores used by every process that is not this OCR process or its children.
export function otherCores(before, after) {
  const wall = Number(after.wallMs) - Number(before.wallMs);
  if (!(wall > 0)) {
    return 0;
  }
  const hz = Number(after.hz) || CPU_HZ;
  const otherTicks = Math.max(
    0,
    (Number(after.totalBusy) - Number(before.totalBusy)) - (Number(after.ownBusy) - Number(before.ownBusy)),
  );
  return otherTicks / ((wall / 1000) * hz);
}

function readProcRecords() {
  const records = [];
  let names = [];
  try {
    names = readdirSync("/proc");
  } catch {
    return records;
  }
  for (const name of names) {
    if (!/^[0-9]+$/.test(name)) {
      continue;
    }
    try {
      const parsed = procTicks(readFileSync(`/proc/${name}/stat`, "utf8"));
      records.push({ pid: Number(name), ppid: parsed.ppid, ticks: parsed.ticks });
    } catch {
      // The process exited between the directory listing and the read.
    }
  }
  return records;
}

function readCpuSample() {
  const ownBusy = treeTicks(readProcRecords(), process.pid);
  const totalBusy = busyTicks(readFileSync("/proc/stat", "utf8"));
  return { wallMs: Date.now(), totalBusy, ownBusy, hz: CPU_HZ };
}

// Calls apply(slots, otherCores) about every two seconds. The first call is
// the first real delta, so a busy host is noticed within one sample.
function watchHostCpu(cpus, ceiling, apply) {
  let previous = null;
  try {
    previous = readCpuSample();
  } catch (err) {
    console.error(`ocr cpu sample failed: ${err && err.message ? err.message : err}`);
    return () => {};
  }
  let ema = null;
  const timer = setInterval(() => {
    let sample = null;
    try {
      sample = readCpuSample();
    } catch (err) {
      console.error(`ocr cpu sample failed: ${err && err.message ? err.message : err}`);
      return;
    }
    const cores = otherCores(previous, sample);
    previous = sample;
    ema = ema === null ? cores : ema * 0.5 + cores * 0.5;
    apply(Math.min(ceiling, ocrSlots(cpus, ema)), ema);
  }, CPU_SAMPLE_MS);
  return () => clearInterval(timer);
}

function yieldToHost() {
  try {
    setPriority(process.pid, OCR_NICE);
  } catch (err) {
    console.error(`ocr priority unchanged: ${err && err.message ? err.message : err}`);
  }
}

// jobs are independent. runners is the number of live loops. getSlots() can
// fall to 0, which parks every loop after its current job. wake() is handed
// the function that releases parked loops when the slot count changes.
export async function runPool({ jobs, runners, getSlots, wake, work }) {
  const results = new Array(jobs);
  let next = 0;
  let active = 0;
  let failure = null;
  let waiters = [];
  const notify = () => {
    const pending = waiters;
    waiters = [];
    for (const resolve of pending) {
      resolve();
    }
  };
  if (wake) {
    wake(notify);
  }
  const park = () => new Promise((resolve) => {
    waiters.push(resolve);
  });
  function takeJob() {
    if (failure) {
      return "stop";
    }
    if (next >= jobs) {
      return "done";
    }
    if (active >= getSlots()) {
      return "wait";
    }
    active += 1;
    const index = next;
    next += 1;
    return index;
  }
  async function loop(runner) {
    for (;;) {
      const job = takeJob();
      if (job === "stop" || job === "done") {
        return;
      }
      if (job === "wait") {
        await park();
        continue;
      }
      try {
        results[job] = await work(job, runner);
      } catch (err) {
        if (!failure) {
          failure = err;
        }
        return;
      } finally {
        active -= 1;
        notify();
      }
    }
  }
  const width = Math.max(1, Math.min(runners, Math.max(jobs, 1)));
  await Promise.all(Array.from({ length: jobs === 0 ? 0 : width }, (_, index) => loop(index)));
  if (failure) {
    throw failure;
  }
  return results;
}

function longWords(text) {
  return String(text || "").toLowerCase().match(/[a-z0-9]{3,}/g) || [];
}

// Words the page image has that the text layer does not.
export function freshWords(layer, ocr) {
  const known = new Set(longWords(layer));
  const seen = new Set();
  const fresh = [];
  for (const word of longWords(ocr)) {
    if (known.has(word) || seen.has(word)) {
      continue;
    }
    seen.add(word);
    fresh.push(word);
  }
  return fresh;
}

export function mergePage(layer, ocr) {
  const written = String(layer || "").trim();
  const seen = String(ocr || "").trim();
  if (!seen) {
    return written;
  }
  if (!written) {
    return seen;
  }
  // A few OCR misreads are not a second copy of the page. A caption or a
  // scanned section has several words the text layer never contained.
  if (freshWords(written, seen).length < 3) {
    return written;
  }
  return `${written}\n${seen}`;
}

async function preprocess(buffer, shrink) {
  let image = sharp(buffer).grayscale().normalize().sharpen();
  if (shrink) image = image.resize({ width: 2000, fit: "inside" });
  return image.toBuffer();
}

async function startWorker(index) {
  return createWorker("eng", OEM.LSTM_ONLY, {
    langPath: "/tessdata",
    cachePath: `/tmp/tess-cache-${process.pid}-${index}`,
    gzip: true,
  });
}

async function startWorkers(count) {
  const settled = await Promise.allSettled(
    Array.from({ length: count }, (_, index) => startWorker(index)),
  );
  const workers = [];
  let failure = null;
  for (const item of settled) {
    if (item.status === "fulfilled") {
      workers.push(item.value);
    } else if (!failure) {
      failure = item.reason;
    }
  }
  if (failure) {
    await stopWorkers(workers);
    throw failure;
  }
  return workers;
}

async function stopWorkers(workers) {
  await Promise.all((workers || []).filter(Boolean).map((worker) => worker.terminate().catch(() => {})));
}

async function ocrBuffer(worker, buffer, shrink) {
  const prepared = await preprocess(buffer, shrink);
  const result = await worker.recognize(prepared);
  return (result.data && result.data.text) || "";
}

async function pdfDocument(buffer) {
  const parser = new PDFParse({ data: buffer });
  try {
    let text = { pages: [], total: 0 };
    try {
      text = await parser.getText();
    } catch (err) {
      console.error(`pdf text layer failed: ${err && err.message ? err.message : err}`);
    }
    let boxes = [];
    let infoTotal = 0;
    try {
      const info = await parser.getInfo({ parsePageInfo: true });
      boxes = Array.isArray(info && info.pages) ? info.pages : [];
      infoTotal = Number(info && info.total) || 0;
    } catch (err) {
      console.error(`pdf page size failed: ${err && err.message ? err.message : err}`);
    }
    const pages = Array.isArray(text && text.pages) ? text.pages : [];
    const byNum = new Map(
      pages.map((page) => [Number(page.num) || 0, String(page.text || "").trim()]),
    );
    const sizes = new Map();
    for (const box of boxes) {
      const num = Number(box.pageNumber) || 0;
      const width = Number(box.width) || 0;
      const height = Number(box.height) || 0;
      if (num > 0 && width > 0 && height > 0) {
        sizes.set(num, { width, height });
      }
    }
    const total = Math.max(Number(text && text.total) || 0, infoTotal, pages.length, sizes.size);
    return { total, byNum, sizes };
  } finally {
    await parser.destroy();
  }
}

function gmIdentify(target, format) {
  return new Promise((resolve, reject) => {
    gm(target).identify(format, (err, value) => {
      if (err) {
        reject(err);
        return;
      }
      resolve(String(value).trim());
    });
  });
}

function capPixels(width, height, cap) {
  const w = Math.max(1, Math.round(Number(width) || 1));
  const h = Math.max(1, Math.round(Number(height) || 1));
  const limit = Math.max(1, Number(cap) || activeProfile.cap);
  const scale = Math.min(1, limit / w, limit / h);
  return {
    width: Math.max(1, Math.round(w * scale)),
    height: Math.max(1, Math.round(h * scale)),
  };
}

// Points at the quality's dpi, then the long-side cap. The draw uses this size directly.
export function rasterPixels(pointsW, pointsH, profile = activeProfile) {
  const dpi = profile.dpi;
  return capPixels(
    (Number(pointsW) || 612) * dpi / 72,
    (Number(pointsH) || 792) * dpi / 72,
    profile.cap,
  );
}

// One GraphicsMagick convert. Density is an input option, so the page is
// rastered at 200 dpi. An exact size scales that draw down; it does not
// open the page a second time.
function renderPage(pdfPath, page, pixels) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let command = gm(`${pdfPath}[${page - 1}]`).density(activeProfile.dpi, activeProfile.dpi).units("PixelsPerInch");
    if (pixels) {
      command = command.resize(pixels.width, pixels.height, "!");
    }
    command.quality(50).compress("jpeg").stream("png", (err, stdout, stderr) => {
      if (err) {
        reject(err);
        return;
      }
      const errors = [];
      if (stderr) {
        stderr.on("data", (chunk) => errors.push(chunk));
      }
      stdout.on("data", (chunk) => chunks.push(chunk));
      stdout.on("error", reject);
      stdout.on("end", () => {
        const buffer = Buffer.concat(chunks);
        if (!buffer.length) {
          const detail = Buffer.concat(errors).toString("utf8").trim();
          reject(new Error(detail || `pdf page ${page} had no image`));
          return;
        }
        resolve(buffer);
      });
    });
  });
}

// Function words only. Nothing is downloaded to recognize a language.
// True is English, false is clearly some other language, and null is too
// little text to judge. A short caption stays null so it is kept.
const ENGLISH_WORDS = new Set(
  "the of and to that for with this from have are was were been which their there would could should about into than then these those what when where who will your they them his her its but not you our has had can may also such only other over after before because while is".split(/\s+/),
);
const OTHER_WORDS = new Set(
  "le la les des une est dans pour que qui pas sur avec sont cette ces aux mais nous vous leur comme aussi etre ete dont donc chez entre apres avant sous sans peut elle ils du au se ne un ce sa ses quand alors bien encore ici tout fait tres tre et ou es il par de und der die das den dem ein eine nicht ist sich auf fur fuer mit von auch nach bei aus wie oder aber wenn nur noch kann wird sind durch gegen ohne zwischen el las del por con una para como mas pero sus este esta entre cuando muy tambien sobre hay son fue estan het van een niet voor naar ook nog wel zijn che sono della degli nella das dass fur uber für über été très où après être".split(/\s+/),
);
for (const word of ENGLISH_WORDS) {
  OTHER_WORDS.delete(word);
}

export function englishProse(text) {
  const raw = String(text || "");
  let letters = 0;
  let latin = 0;
  for (const ch of raw) {
    if (!/\p{L}/u.test(ch)) {
      continue;
    }
    letters += 1;
    if (/\p{Script=Latin}/u.test(ch)) {
      latin += 1;
    }
  }
  const nonLatin = letters - latin;
  if (letters >= 40 && nonLatin >= 12 && nonLatin / letters >= 0.2) {
    return false;
  }
  const words = raw.toLowerCase().match(/[a-zà-ÿœæ]+/gu) || [];
  if (letters < 40 || words.length < 8) {
    return null;
  }
  let english = 0;
  let other = 0;
  let accented = 0;
  for (const word of words) {
    if (/[à-ÿœæ]/u.test(word)) {
      accented += 1;
    }
    if (ENGLISH_WORDS.has(word)) {
      english += 1;
    } else if (OTHER_WORDS.has(word)) {
      other += 1;
    }
  }
  if (english >= 4 && english >= other) {
    return true;
  }
  if (english >= 8) {
    return true;
  }
  if (other >= 4 && other > english * 2) {
    return false;
  }
  if (words.length >= 12 && accented >= 4 && accented / words.length >= 0.1 && english <= 1) {
    return false;
  }
  return null;
}

// keep is the pages to read. sample means the text layer was too thin to
// judge, so the first kept page has to be read before the rest. skipped is
// how many pages were clearly not English.
export function selectEnglishPages(pageNumbers, byNum) {
  const labels = pageNumbers.map((page) => ({
    page,
    verdict: englishProse((byNum && byNum.get(page)) || ""),
  }));
  const judged = labels.filter((row) => row.verdict !== null);
  const english = judged.some((row) => row.verdict === true);
  const foreign = labels.filter((row) => row.verdict === false);
  if (judged.length && !english) {
    return { keep: [], sample: false, skipped: foreign.length };
  }
  const drop = new Set(foreign.map((row) => row.page));
  return {
    keep: pageNumbers.filter((page) => !drop.has(page)),
    sample: judged.length === 0 && pageNumbers.length > 0,
    skipped: foreign.length,
  };
}

export async function pdfLayer(buffer) {
  const doc = await pdfDocument(buffer);
  const pages = [];
  const total = Math.max(doc.total, ...doc.byNum.keys(), 0);
  for (let page = 1; page <= total; page += 1) {
    pages.push({ page, text: doc.byNum.get(page) || "" });
  }
  return { total, pages };
}

export async function readPdf(buffer, profile = activeProfile) {
  activeProfile = profile;
  const doc = await pdfDocument(buffer);
  const byNum = doc.byNum;
  const layerChars = [...byNum.values()].reduce((sum, text) => sum + text.length, 0);
  const filled = profile.drawFilled ? "every page" : "pages with no text layer";
  const knownPages = doc.total > 0 ? Array.from({ length: doc.total }, (_, index) => index + 1) : [];
  if (knownPages.length && !selectEnglishPages(knownPages, byNum).keep.length) {
    console.error(`pdf text layer ${layerChars} chars, quality ${profile.name}`);
    console.error("ocr skip, text is not english");
    return { pages: [], drew: false, skip: "language" };
  }
  console.error(`pdf text layer ${layerChars} chars, quality ${profile.name}, reading ${filled}`);
  const dir = mkdtempSync(join(tmpdir(), "nomad-turbo-pdf-"));
  const pdfPath = join(dir, "input.pdf");
  try {
    writeFileSync(pdfPath, buffer);
    let foundPages = knownPages;
    if (!foundPages.length) {
      foundPages = (await gmIdentify(pdfPath, "%p "))
        .split(/\s+/)
        .map((item) => Number(item))
        .filter((item) => Number.isInteger(item) && item > 0);
    }
    if (!foundPages.length) {
      throw new Error("pdf had no pages");
    }
    const choice = selectEnglishPages(foundPages, byNum);

    async function drawListed(listed) {
      const pageNumbers = listed;
      const pendingDraw = pageNumbers.filter((page) => profile.drawFilled || !(byNum.get(page) || "").trim());
      if (!pendingDraw.length) {
        return {
          pages: pageNumbers.map((page) => ({ page, layer: byNum.get(page) || "", ocr: "", drawn: false })),
          drew: false,
        };
      }
      const cpus = Math.max(1, availableParallelism() || 1);
      const ceiling = Math.max(1, Math.min(pageNumbers.length, ocrSlots(cpus, 0)));
      const pool = { workers: null, ceiling, gpuFailures: 0 };
      if (!process.env.NOMAD_TURBO_OCR_SOCKET) {
        pool.workers = await startWorkers(ceiling);
      }
      console.error(poolLine(clampSlots(ceiling, readBudget()), cpus, -1));
      let slots = ceiling;
      let notify = () => {};
      let shown = ceiling;
      let pending = null;
      let pendingHits = 0;
      const stopWatch = watchHostCpu(cpus, ceiling, (next, other) => {
        next = clampSlots(next, readBudget());
        if (next === shown) {
          pending = null;
          pendingHits = 0;
          return;
        }
        // A one-thread flicker is sampled twice before the pool moves. Giving
        // up two or more threads happens on the sample that saw it.
        const sudden = shown - next >= 2;
        if (next === pending) {
          pendingHits += 1;
        } else {
          pending = next;
          pendingHits = 1;
        }
        if (!sudden && pendingHits < 2) {
          return;
        }
        pending = null;
        pendingHits = 0;
        shown = next;
        slots = next;
        console.error(poolLine(next, cpus, other));
        notify();
      });
      try {
        let done = 0;
        const rows = await runPool({
          jobs: pageNumbers.length,
          runners: pool.ceiling,
          getSlots: () => clampSlots(slots, readBudget()),
          wake: (fn) => {
            notify = fn;
          },
          work: async (index, runner) => {
            const page = pageNumbers[index];
            const layer = byNum.get(page) || "";
            if (!profile.drawFilled && layer.trim()) {
              done += 1;
              console.error(`ocr page ${done} of ${pageNumbers.length} text`);
              return { page, layer, ocr: "", drawn: false };
            }
            const box = doc.sizes.get(page);
            // The page box is in points. Drawing at that pixel size is the only
            // GraphicsMagick read. A page with no box is drawn at the quality's
            // dpi and scaled down afterwards, still once.
            const pixels = box ? rasterPixels(box.width, box.height, profile) : null;
            let rendered = await renderPage(pdfPath, page, pixels);
            let meta = await sharp(rendered).metadata();
            let width = meta && meta.width ? meta.width : 0;
            let height = meta && meta.height ? meta.height : 0;
            if (!pixels && (width > profile.cap || height > profile.cap)) {
              const fitted = capPixels(width, height, profile.cap);
              rendered = await sharp(rendered)
                .resize({ width: fitted.width, height: fitted.height, fit: "fill" })
                .toBuffer();
              width = fitted.width;
              height = fitted.height;
            }
            let seen = "";
            try {
              seen = await recognizePrepared(rendered, profile.shrink, runner, pool);
            } catch (err) {
              console.error(`ocr page failed ${page}: ${err && err.message ? err.message : err}`);
            }
            done += 1;
            console.error(`ocr page ${done} of ${pageNumbers.length} ${width}x${height}`);
            return { page, layer, ocr: seen, drawn: true };
          },
        });
        const pages = rows.filter(Boolean);
        return { pages, drew: pages.some((page) => page.drawn) };
      } finally {
        stopWatch();
        await stopWorkers(pool.workers);
      }
    }

    if (!choice.keep.length) {
      console.error("ocr skip, text is not english");
      return { pages: [], drew: false, skip: "language" };
    }
    if (choice.skipped) {
      const noun = choice.skipped === 1 ? "page" : "pages";
      console.error(`ocr left out ${choice.skipped} ${noun} that are not english`);
    }
    if (choice.sample) {
      const first = await drawListed([choice.keep[0]]);
      const row = first.pages[0] || {};
      const sampleText = [row.layer, row.ocr].filter(Boolean).join("\n");
      if (englishProse(sampleText) === false) {
        console.error("ocr skip, text is not english");
        return { pages: [], drew: false, skip: "language" };
      }
      if (choice.keep.length === 1) {
        return first;
      }
      const rest = await drawListed(choice.keep.slice(1));
      const pages = first.pages.concat(rest.pages);
      return { pages, drew: pages.some((page) => page.drawn) };
    }
    return await drawListed(choice.keep);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

export async function readImage(buffer, profile = activeProfile) {
  activeProfile = profile;
  const pool = { workers: null, ceiling: 1, gpuFailures: 0 };
  let told = false;
  while (clampSlots(1, readBudget()) < 1) {
    if (!told) {
      console.error("ocr waiting while Qdrant is busy");
      told = true;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  if (!pictureNoted) {
    pictureNoted = true;
    console.error(`ocr reading one picture. ${readerNote()}`);
  }
  try {
    const meta = await sharp(buffer).metadata();
    const width = meta && meta.width ? meta.width : 0;
    const height = meta && meta.height ? meta.height : 0;
    if (width > 0 && height > 0 && (width < 8 || height < 8)) {
      console.error(`ocr skip picture ${width}x${height}`);
      return "";
    }
    return await recognizePrepared(buffer, profile.shrink, 0, pool);
  } finally {
    await stopWorkers(pool.workers);
  }
}

function joinedText(parts) {
  return (parts.pages || [])
    .map((page) => [page.layer, page.ocr].filter(Boolean).join("\n"))
    .filter(Boolean)
    .join("\n\n");
}

function arg(name) {
  const index = process.argv.indexOf(name);
  if (index < 0 || index + 1 >= process.argv.length) {
    return "";
  }
  return process.argv[index + 1];
}

function testDrawChoice() {
  const medium = qualityProfile("medium");
  if (medium.drawFilled || medium.dpi !== 300 || medium.shrink) {
    throw new Error("medium should leave a page that already has text");
  }
  const high = qualityProfile("high", false);
  if (high.drawFilled || high.cap !== 12000 || high.shrink) {
    throw new Error("high should leave a page that already has text");
  }
  const forced = qualityProfile("low", true);
  if (!forced.drawFilled || forced.dpi !== 200 || !forced.shrink || forced.cap !== 4000) {
    throw new Error("ocr every pdf should still draw at the chosen quality");
  }
}

function testRaster() {
  const low = qualityProfile("low");
  const letter = rasterPixels(612, 792, low);
  if (letter.width !== 1700 || letter.height !== 2200) {
    throw new Error(`letter page ${letter.width}x${letter.height}`);
  }
  const wide = rasterPixels(1600, 400, low);
  if (wide.width !== 4000 || wide.height !== 1000) {
    throw new Error(`wide page ${wide.width}x${wide.height}`);
  }
  const medium = qualityProfile("medium");
  const sharper = rasterPixels(612, 792, medium);
  if (sharper.width !== 2550 || sharper.height !== 3300) {
    throw new Error(`medium letter ${sharper.width}x${sharper.height}`);
  }
  const missing = rasterPixels(0, 0, low);
  if (missing.width !== 1700 || missing.height !== 2200) {
    throw new Error(`default page ${missing.width}x${missing.height}`);
  }
}

function testMerge() {
  const body = "The shelter manual explains water, food, and ventilation for a family of six.";
  if (mergePage(body, body) !== body) {
    throw new Error("a page of real text should not be pasted twice");
  }
  if (mergePage(body, "the the the") !== body) {
    throw new Error("short OCR noise should not be added");
  }
  if (mergePage("", "HELLO") !== "HELLO") {
    throw new Error("a page with no text layer should keep the OCR");
  }
  const both = mergePage(body, `${body}\nEMERGENCY RATION CHART`);
  if (!both.includes("shelter manual") || !both.includes("EMERGENCY RATION CHART")) {
    throw new Error("real text and image text should both be kept");
  }
}

function testSlots() {
  if (ocrSlots(12, 0) !== 10) {
    throw new Error(`quiet 12-thread host should run 10 workers, got ${ocrSlots(12, 0)}`);
  }
  if (ocrSlots(12, 0.4) !== 10 || ocrSlots(12, 2.4) !== 10) {
    throw new Error("the reserved threads should absorb a light host");
  }
  if (ocrSlots(12, 2.6) !== 9 || ocrSlots(12, 4) !== 8) {
    throw new Error(`host load should park workers, got ${ocrSlots(12, 2.6)} and ${ocrSlots(12, 4)}`);
  }
  if (ocrSlots(12, 12) !== 0 || ocrSlots(12, 14) !== 0) {
    throw new Error("a full host should park every worker");
  }
  if (ocrSlots(2, 0) !== 1 || ocrSlots(1, 0) !== 1 || ocrSlots(1, 1) !== 0) {
    throw new Error("a small host should keep one worker until that core is taken");
  }
  if (ocrSlots(Number.NaN, Number.NaN) !== 1) {
    throw new Error("missing cpu info should still run one worker");
  }
  if (clampSlots(10, null) !== 10 || clampSlots(10, "") !== 10 || clampSlots(10, "nope") !== 10) {
    throw new Error("a missing budget should leave the host sample alone");
  }
  if (clampSlots(10, "2") !== 2 || clampSlots(10, "0") !== 0 || clampSlots(2, "8") !== 2) {
    throw new Error("the budget is a ceiling, including zero while Qdrant is refused");
  }
}

function testCpuMath() {
  const ticks = procTicks("10 (ocr worker) S 1 0 0 0 0 0 0 0 0 0 5 6 7 8\n");
  if (ticks.ppid !== 1 || ticks.ticks !== 26) {
    throw new Error(`proc ticks parsed ${JSON.stringify(ticks)}`);
  }
  const busy = busyTicks("cpu  100 20 30 400 5 6 7 8 0 0\ncpu0 1 0 0 0 0 0 0 0\n");
  if (busy !== 163) {
    throw new Error(`busy ticks ${busy}`);
  }
  const own = treeTicks([
    { pid: 10, ppid: 1, ticks: 26 },
    { pid: 11, ppid: 10, ticks: 4 },
    { pid: 12, ppid: 11, ticks: 1 },
    { pid: 99, ppid: 1, ticks: 50 },
  ], 10);
  if (own !== 31) {
    throw new Error(`tree ticks ${own}`);
  }
  const quiet = otherCores(
    { wallMs: 0, totalBusy: 0, ownBusy: 0 },
    { wallMs: 1000, totalBusy: 1000, ownBusy: 800, hz: 100 },
  );
  if (Math.abs(quiet - 2) > 0.001) {
    throw new Error(`other cores ${quiet}`);
  }
  const skewed = otherCores(
    { wallMs: 0, totalBusy: 0, ownBusy: 0 },
    { wallMs: 2000, totalBusy: 500, ownBusy: 800, hz: 100 },
  );
  if (skewed !== 0) {
    throw new Error(`own time ahead of the host sample should not look like other work, got ${skewed}`);
  }
}

async function testPoolPauses() {
  const blocks = [];
  let started = 0;
  let active = 0;
  let maxActive = 0;
  let slots = 2;
  let notify = () => {};
  const done = runPool({
    jobs: 5,
    runners: 4,
    getSlots: () => slots,
    wake: (fn) => {
      notify = fn;
    },
    work: async (index) => {
      started += 1;
      active += 1;
      maxActive = Math.max(maxActive, active);
      let resolve = null;
      const gate = new Promise((finish) => {
        resolve = finish;
      });
      blocks.push({ index, resolve });
      await gate;
      active -= 1;
      return index;
    },
  });
  if (started !== 2 || maxActive !== 2) {
    throw new Error(`pool started ${started} with ${maxActive} active`);
  }
  slots = 0;
  notify();
  const first = blocks.splice(0, blocks.length);
  for (const item of first) {
    item.resolve();
  }
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));
  if (started !== 2) {
    throw new Error(`paused pool started ${started}`);
  }
  slots = 2;
  notify();
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));
  if (started !== 4 || maxActive !== 2) {
    throw new Error(`resumed pool started ${started}, max ${maxActive}`);
  }
  const timeoutAt = Date.now() + 1000;
  let results = null;
  while (Date.now() < timeoutAt && results === null) {
    slots = 4;
    notify();
    const batch = blocks.splice(0, blocks.length);
    for (const item of batch) {
      item.resolve();
    }
    results = await Promise.race([
      done.then((value) => value),
      new Promise((resolve) => setImmediate(() => resolve(null))),
    ]);
  }
  if (!results || results.join(",") !== "0,1,2,3,4") {
    throw new Error(`page order ${results && results.join(",")}`);
  }
}

function testLanguage() {
  const english = "The shelter manual explains water, food, and ventilation for a family of six. This guide is for people who have to stay inside with their children when the weather is bad.";
  if (englishProse(english) !== true) {
    throw new Error(`english prose ${englishProse(english)}`);
  }
  const french = "Le guide explique comment stocker l'eau et la nourriture pour les familles. Les pages suivantes decrivent les etapes dans la maison et pour les enfants qui sont dans la cuisine avec une lampe.";
  if (englishProse(french) !== false) {
    throw new Error(`french prose ${englishProse(french)}`);
  }
  const german = "Die Anleitung erklaert wie eine Familie Wasser und Nahrung lagert und sich auf den Winter vorbereitet. Der Text ist nicht fuer den Garten und nicht fuer die Schule und auch nicht fuer das Buero.";
  if (englishProse(german) !== false) {
    throw new Error(`german prose ${englishProse(german)}`);
  }
  const spanish = "La guia explica como guardar el agua y la comida para una familia. Este texto es para las personas que estan en la casa y para los ninos que viven con sus padres.";
  if (englishProse(spanish) !== false) {
    throw new Error(`spanish prose ${englishProse(spanish)}`);
  }
  const cjk = "この文書は家族のための水と食料の保管方法を説明します。冬に備えて家の中で準備をする手順です。";
  if (englishProse(cjk) !== false) {
    throw new Error(`cjk prose ${englishProse(cjk)}`);
  }
  const arabic = "يشرح هذا الدليل كيفية تخزين الماء والطعام للعائلة خلال فصل الشتاء داخل المنزل ومع الأطفال.";
  if (englishProse(arabic) !== false) {
    throw new Error(`arabic prose ${englishProse(arabic)}`);
  }
  if (englishProse("Fig. 1") !== null) {
    throw new Error(`figure caption ${englishProse("Fig. 1")}`);
  }
  if (englishProse("Pump diagram") !== null) {
    throw new Error(`short caption ${englishProse("Pump diagram")}`);
  }
  if (englishProse(`${english} ${french}`) !== true) {
    throw new Error("a page with English and another language should stay");
  }
  const mixed = selectEnglishPages(
    [1, 2, 3],
    new Map([[1, french], [2, english], [3, "Fig. 1"]]),
  );
  if (mixed.keep.join(",") !== "2,3" || mixed.sample || mixed.skipped !== 1) {
    throw new Error(`mixed pages ${JSON.stringify(mixed)}`);
  }
  const foreign = selectEnglishPages([1, 2], new Map([[1, french], [2, german]]));
  if (foreign.keep.length || foreign.sample) {
    throw new Error(`foreign file ${JSON.stringify(foreign)}`);
  }
  const scan = selectEnglishPages([1, 2, 3], new Map([[1, ""], [2, ""], [3, ""]]));
  if (!scan.sample || scan.keep.join(",") !== "1,2,3") {
    throw new Error(`scan ${JSON.stringify(scan)}`);
  }
  if (englishProse("C6H12O6 NaCl H2O") !== null) {
    throw new Error(`formula ${englishProse("C6H12O6 NaCl H2O")}`);
  }
}

async function main() {
  if (process.argv.includes("--self-test")) {
    testRaster();
    testDrawChoice();
    testMerge();
    testSlots();
    testCpuMath();
    testLanguage();
    await testPoolPauses();
    const first = readCpuSample();
    await new Promise((resolve) => setTimeout(resolve, 40));
    const second = readCpuSample();
    const live = otherCores(first, second);
    if (!Number.isFinite(live) || live < 0 || !(first.totalBusy > 0) || !(first.ownBusy > 0)) {
      throw new Error(`live cpu sample ${live} total ${first.totalBusy} own ${first.ownBusy}`);
    }
    const image = arg("--image");
    if (image) {
      const text = await readImage(readFileSync(image));
      if (!/HELLO/i.test(text)) {
        throw new Error(`image read ${JSON.stringify(text)}`);
      }
    }
    const pdf = arg("--pdf");
    if (pdf) {
      const text = joinedText(await readPdf(readFileSync(pdf), qualityProfile("low")));
      if (!/HELLO/i.test(text)) {
        throw new Error(`pdf read ${JSON.stringify(text)}`);
      }
    }
    console.error("self-test ok");
    return;
  }
  yieldToHost();
  const file = arg("--file");
  const kind = arg("--kind");
  const profile = qualityProfile(arg("--quality") || "medium", process.argv.includes("--ocr-every-pdf"));
  if (!file || (kind !== "pdf" && kind !== "image")) {
    throw new Error("usage: --file PATH --kind pdf|image");
  }
  const buffer = readFileSync(file);
  if (kind === "image") {
    const text = await readImage(buffer, profile);
    if (englishProse(text) === false) {
      console.error("ocr skip, text is not english");
      if (arg("--format") === "pages") {
        process.stdout.write(JSON.stringify({ ocr: "", drew: true, pages: [], skip: "language" }));
        return;
      }
      return;
    }
    if (arg("--format") === "pages") {
      process.stdout.write(JSON.stringify({ ocr: text, drew: true, pages: [] }));
      return;
    }
    process.stdout.write(text);
    return;
  }
  const parts = await readPdf(buffer, profile);
  if (arg("--format") === "pages") {
    process.stdout.write(JSON.stringify(parts));
    return;
  }
  process.stdout.write(joinedText(parts));
}

const launched = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (launched) {
  main().catch((err) => {
    console.error(err && err.stack ? err.stack : String(err));
    process.exit(1);
  });
}

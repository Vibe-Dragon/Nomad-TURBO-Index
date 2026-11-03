// Read one uploaded PDF or image with the OCR libraries in the NOMAD image.
// This does not call NOMAD's embedder. Turbo chunks the text and embeds it.
//
// Every PDF page keeps its text layer and is also rendered and read. A page
// of real text is not pasted twice: OCR is added when it has words the text
// layer does not. Pages are read together, one Tesseract worker for each
// hardware thread after two threads are left free. A worker finishes the page
// it is on and waits while other programs are using the CPU, then takes a
// thread back when that work is gone. The process also runs at a lower
// priority, so a short request is served during a page.
// pdf2pic draws a page at 768x512 unless it is told the size, so the reader
// used to ask GraphicsMagick for that size and then draw the page. That read
// every page twice. The size now comes from the PDF page box, and GraphicsMagick
// draws each page once at 200 dpi. A long side past 4000 pixels is scaled down
// in that same draw. sharp then does NOMAD's preprocess.
// A picture follows processImageFile: grayscale, normalize, sharpen, fit
// inside 2000 pixels, then English LSTM. The traineddata file is the same
// one tesseract.js loads for that call. Do not download another at run time.
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { availableParallelism, setPriority, tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const require = createRequire("/app/package.json");
const sharp = require("sharp");
const gm = require("gm");
const { PDFParse } = require("pdf-parse");
const { createWorker, OEM } = require("tesseract.js");

// NOMAD renders a PDF page at this density. The long side stops here so one
// page cannot fill memory. Both are applied in the one GraphicsMagick draw.
const RASTER_DPI = 200;
const RASTER_CAP = 4000;

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

async function preprocess(buffer) {
  return sharp(buffer)
    .grayscale()
    .normalize()
    .sharpen()
    .resize({ width: 2000, fit: "inside" })
    .toBuffer();
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
  await Promise.all(workers.filter(Boolean).map((worker) => worker.terminate().catch(() => {})));
}

async function ocrBuffer(worker, buffer) {
  const prepared = await preprocess(buffer);
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

function capPixels(width, height) {
  const w = Math.max(1, Math.round(Number(width) || 1));
  const h = Math.max(1, Math.round(Number(height) || 1));
  const scale = Math.min(1, RASTER_CAP / w, RASTER_CAP / h);
  return {
    width: Math.max(1, Math.round(w * scale)),
    height: Math.max(1, Math.round(h * scale)),
  };
}

// Points at 200 dpi, then the long-side cap. The draw uses this size directly.
export function rasterPixels(pointsW, pointsH) {
  return capPixels(
    (Number(pointsW) || 612) * RASTER_DPI / 72,
    (Number(pointsH) || 792) * RASTER_DPI / 72,
  );
}

// One GraphicsMagick convert. Density is an input option, so the page is
// rastered at 200 dpi. An exact size scales that draw down; it does not
// open the page a second time.
function renderPage(pdfPath, page, pixels) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let command = gm(`${pdfPath}[${page - 1}]`).density(RASTER_DPI, RASTER_DPI).units("PixelsPerInch");
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

async function readPdf(buffer) {
  const doc = await pdfDocument(buffer);
  const byNum = doc.byNum;
  const layerChars = [...byNum.values()].reduce((sum, text) => sum + text.length, 0);
  console.error(`pdf text layer ${layerChars} chars, reading every page`);
  const dir = mkdtempSync(join(tmpdir(), "nomad-turbo-pdf-"));
  const pdfPath = join(dir, "input.pdf");
  try {
    writeFileSync(pdfPath, buffer);
    let pageNumbers = doc.total > 0 ? Array.from({ length: doc.total }, (_, index) => index + 1) : [];
    if (!pageNumbers.length) {
      pageNumbers = (await gmIdentify(pdfPath, "%p "))
        .split(/\s+/)
        .map((item) => Number(item))
        .filter((item) => Number.isInteger(item) && item > 0);
    }
    if (!pageNumbers.length) {
      throw new Error("pdf had no pages");
    }
    const cpus = Math.max(1, availableParallelism() || 1);
    const ceiling = Math.max(1, Math.min(pageNumbers.length, ocrSlots(cpus, 0)));
    console.error(`ocr workers ${ceiling} of ${cpus} threads`);
    const workers = await startWorkers(ceiling);
    let slots = ceiling;
    let notify = () => {};
    let shown = ceiling;
    let pending = null;
    let pendingHits = 0;
    const stopWatch = watchHostCpu(cpus, ceiling, (next, other) => {
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
      const host = other >= 0.5 ? `, other cpu ${other.toFixed(1)}` : "";
      console.error(`ocr workers ${next} of ${cpus} threads${host}`);
      notify();
    });
    try {
      let done = 0;
      const rows = await runPool({
        jobs: pageNumbers.length,
        runners: workers.length,
        getSlots: () => slots,
        wake: (fn) => {
          notify = fn;
        },
        work: async (index, runner) => {
          const page = pageNumbers[index];
          const box = doc.sizes.get(page);
          // The page box is in points. Drawing at that pixel size is the only
          // GraphicsMagick read. A page with no box is drawn at 200 dpi and
          // scaled down afterwards, still once.
          const pixels = box ? rasterPixels(box.width, box.height) : null;
          let rendered = await renderPage(pdfPath, page, pixels);
          let meta = await sharp(rendered).metadata();
          let width = meta && meta.width ? meta.width : 0;
          let height = meta && meta.height ? meta.height : 0;
          if (!pixels && (width > RASTER_CAP || height > RASTER_CAP)) {
            const fitted = capPixels(width, height);
            rendered = await sharp(rendered)
              .resize({ width: fitted.width, height: fitted.height, fit: "fill" })
              .toBuffer();
            width = fitted.width;
            height = fitted.height;
          }
          const merged = mergePage(byNum.get(page) || "", await ocrBuffer(workers[runner], rendered));
          done += 1;
          console.error(`ocr page ${done} of ${pageNumbers.length} ${width}x${height}`);
          return merged;
        },
      });
      return rows.filter(Boolean).join("\n\n");
    } finally {
      stopWatch();
      await stopWorkers(workers);
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

async function readImage(buffer) {
  const worker = await startWorker(0);
  try {
    return await ocrBuffer(worker, buffer);
  } finally {
    await worker.terminate();
  }
}

function arg(name) {
  const index = process.argv.indexOf(name);
  if (index < 0 || index + 1 >= process.argv.length) {
    return "";
  }
  return process.argv[index + 1];
}

function testRaster() {
  const letter = rasterPixels(612, 792);
  if (letter.width !== 1700 || letter.height !== 2200) {
    throw new Error(`letter page ${letter.width}x${letter.height}`);
  }
  const wide = rasterPixels(1600, 400);
  if (wide.width !== 4000 || wide.height !== 1000) {
    throw new Error(`wide page ${wide.width}x${wide.height}`);
  }
  const missing = rasterPixels(0, 0);
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

async function main() {
  if (process.argv.includes("--self-test")) {
    testRaster();
    testMerge();
    testSlots();
    testCpuMath();
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
      const text = await readPdf(readFileSync(pdf));
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
  if (!file || (kind !== "pdf" && kind !== "image")) {
    throw new Error("usage: --file PATH --kind pdf|image");
  }
  const buffer = readFileSync(file);
  const text = kind === "pdf" ? await readPdf(buffer) : await readImage(buffer);
  process.stdout.write(text);
}

const launched = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (launched) {
  main().catch((err) => {
    console.error(err && err.stack ? err.stack : String(err));
    process.exit(1);
  });
}

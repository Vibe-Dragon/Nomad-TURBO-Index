// Read one uploaded PDF or image with the OCR libraries in the NOMAD image.
// This does not call NOMAD's embedder. Turbo chunks the text and embeds it.
//
// Every PDF page keeps its text layer and is also rendered and read. A page
// of real text is not pasted twice: OCR is added when it has words the text
// layer does not. pdf2pic's own default size is 768x512, which squashes a
// page before OCR.
// The render here is 200 dpi, the density NOMAD asks for, with the long side
// capped so one page cannot fill memory. sharp then does NOMAD's preprocess.
// A picture follows processImageFile: grayscale, normalize, sharpen, fit
// inside 2000 pixels, then English LSTM. The traineddata file is the same
// one tesseract.js loads for that call. Do not download another at run time.
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const require = createRequire("/app/package.json");
const sharp = require("sharp");
const gm = require("gm");
const { PDFParse } = require("pdf-parse");
const { fromPath } = require("pdf2pic");
const { createWorker, OEM } = require("tesseract.js");

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

async function startWorker() {
  return createWorker("eng", OEM.LSTM_ONLY, {
    langPath: "/tessdata",
    cachePath: "/tmp/tess-cache",
    gzip: true,
  });
}

async function ocrBuffer(worker, buffer) {
  const prepared = await preprocess(buffer);
  const result = await worker.recognize(prepared);
  return (result.data && result.data.text) || "";
}

async function pdfPages(buffer) {
  const parser = new PDFParse({ data: buffer });
  try {
    const data = await parser.getText();
    const pages = Array.isArray(data && data.pages) ? data.pages : [];
    return pages.map((page) => ({
      num: Number(page.num) || 0,
      text: String(page.text || "").trim(),
    }));
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

// 200 dpi, then stop a single page from becoming an enormous bitmap.
function rasterPixels(pointsW, pointsH) {
  let width = Math.max(1, Math.round((Number(pointsW) || 612) * 200 / 72));
  let height = Math.max(1, Math.round((Number(pointsH) || 792) * 200 / 72));
  const cap = 4000;
  const scale = Math.min(1, cap / width, cap / height);
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

async function readPdf(buffer) {
  let layers = [];
  try {
    layers = await pdfPages(buffer);
  } catch (err) {
    console.error(`pdf text layer failed: ${err && err.message ? err.message : err}`);
  }
  const byNum = new Map(layers.map((page) => [page.num, page.text]));
  const layerChars = layers.reduce((sum, page) => sum + page.text.length, 0);
  console.error(`pdf text layer ${layerChars} chars, reading every page`);
  const dir = mkdtempSync(join(tmpdir(), "nomad-turbo-pdf-"));
  const pdfPath = join(dir, "input.pdf");
  try {
    writeFileSync(pdfPath, buffer);
    const pageNumbers = (await gmIdentify(pdfPath, "%p "))
      .split(/\s+/)
      .map((item) => Number(item))
      .filter((item) => Number.isInteger(item) && item > 0);
    if (!pageNumbers.length) {
      throw new Error("pdf had no pages");
    }
    const worker = await startWorker();
    try {
      const parts = [];
      for (let i = 0; i < pageNumbers.length; i += 1) {
        const page = pageNumbers[i];
        const size = (await gmIdentify(`${pdfPath}[${page - 1}]`, "%w %h")).split(/\s+/);
        const pixels = rasterPixels(size[0], size[1]);
        // A new converter each page. setOptions on a shared one keeps the
        // first size, and pdf2pic's default size is 768x512.
        const convert = fromPath(pdfPath, {
          quality: 50,
          density: 200,
          format: "png",
          width: pixels.width,
          height: pixels.height,
          preserveAspectRatio: false,
          savePath: "/tmp",
          saveFilename: "nomad-turbo-page",
        });
        const rendered = await convert(page, { responseType: "buffer" });
        if (!rendered || !rendered.buffer) {
          throw new Error(`pdf page ${page} had no image`);
        }
        const meta = await sharp(rendered.buffer).metadata();
        console.error(
          `ocr page ${i + 1} of ${pageNumbers.length} ${meta.width}x${meta.height}`,
        );
        const merged = mergePage(byNum.get(page) || "", await ocrBuffer(worker, rendered.buffer));
        if (merged) {
          parts.push(merged);
        }
      }
      return parts.join("\n\n");
    } finally {
      await worker.terminate();
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

async function readImage(buffer) {
  const worker = await startWorker();
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

async function main() {
  if (process.argv.includes("--self-test")) {
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

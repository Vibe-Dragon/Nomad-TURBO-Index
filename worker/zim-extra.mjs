// Later ZIM passes: EPUB and Word, then PDFs.
// Pictures are left out. HTML is handled by extract.mjs. These helpers do not touch libzim.
import { createHash } from "node:crypto";
import { inflateRawSync } from "node:zlib";
import { appendFileSync, existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

const HTML = new Set(["text/html", "application/xhtml+xml"]);
const PDF = new Set(["application/pdf"]);
const EPUB = new Set(["application/epub+zip"]);
const DOCX = new Set(["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]);
const TEXT = new Set(["text/plain", "text/markdown", "text/x-markdown", "text/vtt"]);
const SVG = new Set(["image/svg+xml"]);
const IMAGES = new Set([
  "image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp",
  "image/tiff", "image/tif", "image/bmp", "image/x-ms-bmp",
]);
const GENERIC = new Set(["", "application/octet-stream", "binary/octet-stream"]);
const EXT = {
  ".html": "html", ".htm": "html", ".xhtml": "html", ".pdf": "ocr",
  ".epub": "text", ".docx": "text", ".txt": "text", ".md": "text",
  ".markdown": "text", ".vtt": "text", ".svg": "text",
  ".jpg": "ocr", ".jpeg": "ocr", ".png": "ocr", ".gif": "ocr",
  ".webp": "ocr", ".tif": "ocr", ".tiff": "ocr", ".bmp": "ocr",
};
const IMAGE_EXT = new Set([".jpg", ".jpeg", ".png", ".gif", ".webp", ".tif", ".tiff", ".bmp"]);
const BOILER = /(?:^|\/)(readme|license|licence|changelog|notice|contributing|code_of_conduct)(?:[._-]|$)/i;
const FNV_OFFSET = 14695981039346656037n;
const FNV_PRIME = 1099511628211n;
const MASK = (1n << 64n) - 1n;

function suffix(path) {
  const name = String(path || "").toLowerCase();
  const dot = name.lastIndexOf(".");
  return dot < 0 ? "" : name.slice(dot);
}

function mimeKind(mime) {
  const name = String(mime || "").split(";", 1)[0].trim().toLowerCase();
  if (GENERIC.has(name)) return "";
  if (HTML.has(name)) return "html";
  if (PDF.has(name) || IMAGES.has(name)) return "ocr";
  if (EPUB.has(name) || DOCX.has(name) || TEXT.has(name) || SVG.has(name)) return "text";
  return "skip";
}

function isPicture(mime, path) {
  const name = String(mime || "").split(";", 1)[0].trim().toLowerCase();
  if (IMAGES.has(name) || SVG.has(name)) return true;
  const ext = suffix(path);
  return IMAGE_EXT.has(ext) || ext === ".svg";
}

function isBook(mime, path) {
  const name = String(mime || "").split(";", 1)[0].trim().toLowerCase();
  if (EPUB.has(name) || DOCX.has(name)) return true;
  const ext = suffix(path);
  return ext === ".epub" || ext === ".docx";
}

export function skipByPath(path) {
  // Map tiles are skipped without opening the entry. Opening every tile is
  // what holds the walk after the HTML pages are done.
  const lowered = String(path || "").replaceAll("\\", "/").toLowerCase();
  if (!lowered) return false;
  if (lowered.startsWith("tiles/") || lowered.includes("/tiles/")) return true;
  const dot = lowered.lastIndexOf(".");
  const slash = lowered.lastIndexOf("/");
  if (dot > slash && (IMAGE_EXT.has(lowered.slice(dot)) || lowered.slice(dot) === ".svg")) return true;
  return false;
}

export function classifyEntry(mime, path, _size) {
  const lowered = String(path || "").replaceAll("\\", "/").toLowerCase();
  if (lowered.startsWith("tiles/") || lowered.includes("/tiles/")) return "skip";
  if (isPicture(mime, lowered)) return "skip";
  let kind = mimeKind(mime);
  if (!kind) kind = EXT[suffix(lowered)] || "skip";
  if (kind === "text" && !isBook(mime, lowered)) return "skip";
  return kind || "skip";
}

export function boilerplate(path, text) {
  const name = String(path || "").replaceAll("\\", "/").split("/").pop();
  if (!BOILER.test(`/${name}`)) return false;
  const words = String(text || "").match(/[A-Za-z0-9]+/g) || [];
  return words.length < 200;
}

export function svgText(raw) {
  return String(raw || "")
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function findEocd(buf) {
  const mark = Buffer.from([0x50, 0x4b, 0x05, 0x06]);
  return buf.lastIndexOf(mark);
}

export function zipFiles(buffer) {
  const buf = Buffer.isBuffer(buffer) ? buffer : Buffer.from(buffer);
  const eocd = findEocd(buf);
  if (eocd < 0 || eocd + 22 > buf.length) return [];
  const count = buf.readUInt16LE(eocd + 10);
  const offset = buf.readUInt32LE(eocd + 16);
  const files = [];
  let cursor = offset;
  for (let index = 0; index < count && cursor + 46 <= buf.length; index += 1) {
    if (buf.readUInt32LE(cursor) !== 0x02014b50) break;
    const method = buf.readUInt16LE(cursor + 10);
    const compressed = buf.readUInt32LE(cursor + 20);
    const nameLen = buf.readUInt16LE(cursor + 28);
    const extraLen = buf.readUInt16LE(cursor + 30);
    const commentLen = buf.readUInt16LE(cursor + 32);
    const local = buf.readUInt32LE(cursor + 42);
    const name = buf.slice(cursor + 46, cursor + 46 + nameLen).toString("utf8");
    cursor += 46 + nameLen + extraLen + commentLen;
    if (name.endsWith("/")) continue;
    if (local + 30 > buf.length) continue;
    const localName = buf.readUInt16LE(local + 26);
    const localExtra = buf.readUInt16LE(local + 28);
    const start = local + 30 + localName + localExtra;
    const raw = buf.slice(start, start + compressed);
    let data = raw;
    if (method === 8) {
      try {
        data = inflateRawSync(raw);
      } catch {
        continue;
      }
    } else if (method !== 0) {
      continue;
    }
    files.push({ name, data });
  }
  return files;
}

export function fnv64(text) {
  let value = FNV_OFFSET;
  for (const byte of Buffer.from(String(text), "utf8")) {
    value ^= BigInt(byte);
    value = (value * FNV_PRIME) & MASK;
  }
  return value;
}

export function fingerprint(text) {
  const tokens = String(text || "").toLowerCase().match(/[a-z0-9]{2,}/g) || [];
  if (!tokens.length) return [0n, 0];
  const bits = new Array(64).fill(0);
  for (const token of tokens) {
    const hashed = fnv64(token);
    for (let index = 0; index < 64; index += 1) {
      bits[index] += (hashed >> BigInt(index)) & 1n ? 1 : -1;
    }
  }
  let value = 0n;
  for (let index = 0; index < 64; index += 1) {
    if (bits[index] > 0) value |= 1n << BigInt(index);
  }
  return [value, tokens.length];
}

function popcount(value) {
  let count = 0;
  let left = value & MASK;
  while (left) {
    count += Number(left & 1n);
    left >>= 1n;
  }
  return count;
}

export function sameWork(left, right) {
  const [leftHash, leftCount] = left;
  const [rightHash, rightCount] = right;
  if (leftCount < 200 || rightCount < 200) return false;
  const short = Math.min(leftCount, rightCount);
  const long = Math.max(leftCount, rightCount);
  if (short / long < 0.8) return false;
  return popcount(leftHash ^ rightHash) <= 3;
}

export function loadPrints(path) {
  if (!existsSync(path)) return [];
  const rows = [];
  for (const line of readFileSync(path, "utf8").split("\n")) {
    const parts = line.trim().split(/\s+/);
    if (parts.length !== 2) continue;
    try {
      rows.push([BigInt(`0x${parts[0]}`), Number(parts[1])]);
    } catch {
      // A broken line is ignored. The rest of the list still applies.
    }
  }
  return rows;
}

export function matchesPrint(text, prints) {
  const current = fingerprint(text);
  return prints.some((row) => sameWork(current, row));
}

export function readList(path) {
  if (!existsSync(path)) return [];
  const rows = [];
  for (const line of readFileSync(path, "utf8").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const value = JSON.parse(trimmed);
      if (typeof value === "string" && value) rows.push(value);
    } catch {
      rows.push(trimmed);
    }
  }
  return rows;
}

export function resetList(path) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, "");
}

export function appendPath(path, entry) {
  mkdirSync(dirname(path), { recursive: true });
  appendFileSync(path, `${JSON.stringify(entry)}\n`);
}

export function aheadFile(spillId, entryPath) {
  const id = createHash("sha256").update(String(entryPath || "")).digest("hex").slice(0, 16);
  return `/turbo-state/ocr-ahead/${spillId}/${id}.json`;
}

export function writeAhead(spillId, entryPath, payload) {
  const dest = aheadFile(spillId, entryPath);
  mkdirSync(dirname(dest), { recursive: true });
  const tmp = `${dest}.tmp`;
  writeFileSync(tmp, JSON.stringify(payload));
  renameSync(tmp, dest);
}

export function readAhead(spillId, entryPath) {
  const dest = aheadFile(spillId, entryPath);
  if (!existsSync(dest)) return null;
  try {
    const value = JSON.parse(readFileSync(dest, "utf8"));
    return value && typeof value === "object" ? value : null;
  } catch {
    rmSync(dest, { force: true });
    return null;
  }
}

export function removeAhead(spillId, entryPath) {
  rmSync(aheadFile(spillId, entryPath), { force: true });
}

export function clearAhead(spillId) {
  if (!spillId) return;
  rmSync(`/turbo-state/ocr-ahead/${spillId}`, { recursive: true, force: true });
}

export function ocrExportFile(spillId, entryPath) {
  const id = createHash("sha256").update(String(entryPath || "")).digest("hex").slice(0, 16);
  return `/turbo-state/ocr-queue/${spillId}/${id}.pdf`;
}

export function writeOcrExport(spillId, entryPath, bytes) {
  const dest = ocrExportFile(spillId, entryPath);
  mkdirSync(dirname(dest), { recursive: true });
  const tmp = `${dest}.tmp`;
  writeFileSync(tmp, bytes);
  renameSync(tmp, dest);
  return dest;
}

export function clearOcrQueue(spillId) {
  if (!spillId) return;
  rmSync(`/turbo-state/ocr-queue/${spillId}`, { recursive: true, force: true });
}

export function clearGo(path) {
  rmSync(path, { force: true });
}

export async function waitForGo(path) {
  while (!existsSync(path)) {
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
}

# Upstream Project NOMAD pull requests

When Turbo takes up a pull request from [Project NOMAD](https://github.com/Crosstalk-Solutions/project-nomad), add it here. Record the number, what the pull request changed, and how Turbo handled it. A pull request Turbo already covers is marked covered. A pull request that is only planned stays marked planned until the code lands, and the entry is then updated with what actually shipped.

Turbo does not copy NOMAD's patch into the admin container. The handling below is the Turbo reader (`worker/ocr.mjs`, `worker/extract.mjs`) and the Python that launches it.

## 1570 — embed each chunk with its article and section heading

- Pull request: https://github.com/Crosstalk-Solutions/project-nomad/pull/1570
- Issue: https://github.com/Crosstalk-Solutions/project-nomad/issues/1569
- Also the retrieval half of https://github.com/Crosstalk-Solutions/project-nomad/issues/1546
- Author: rcronk. Opened 2026-11-07. Commit `7b03b23`. Open against `dev`.
- Status: **covered**.

Stock `embedAndStoreText` embedded `search_document: ` plus the bare section text. The title and heading were already stored on the point. The embedder never read them, so an ingredient list or a dosage table sat far from a question that names the subject.

Turbo puts that line on the text sent to the embedder only. The stored chunk, the payload, and the point id stay the bare section text. A section heading that differs from the title becomes `Title > Heading`. A file with only a name, including a plain upload and a picture, uses that name once. A PDF page keeps its page label, so the line is `bandaging.pdf > page 4`. The line is counted in the token budget. A chunk that would no longer fit is split, and the stored words are kept. HTML, EPUB, captions, PDF pages, pictures, and other uploads all go through `prepare_article`, so they all get the line.

What we did not copy: leaving a plain upload with no heading line. Those files are named, and the name is the subject the page often never repeats.

A page with a single real section used to be stored as one block with a blank heading, so the line was only the article name. That page now keeps the section, and the line is `Title > Heading`. A table on the page keeps its rows. A page with no real section is still one block.

## 1567 — keep a PDF page's aspect ratio when rasterizing it for OCR

- Pull request: https://github.com/Crosstalk-Solutions/project-nomad/pull/1567
- Issue: https://github.com/Crosstalk-Solutions/project-nomad/issues/1566
- Author: rcronk. Opened 2026-11-07. Commit `fb4ee27`.
- Status: **covered**. The page-box draw was already in Turbo. The dpi follow-up below has shipped.

Stock `convertPDFtoImages` calls pdf2pic without a size. pdf2pic then resizes every page to 768×512 and ignores aspect ratio (`gm` resize with `!`), so a 200 dpi portrait page is squashed into a landscape thumbnail before preprocess and Tesseract. The pull request passes `preserveAspectRatio: true` and a 1700 px minimum on both width and height. 1700 px is a letter page (8.5 in) at 200 dpi. The pull request's own check: Emergency-Childbirth mean confidence 57.6 to 83.4 and dictionary words 74.7% to 97.9%; Bandaging-and-Splinting 72.9 to 81.6 and 90.2% to 97.2%. Related pull request #1460 renames the method to `convertPDFPagesToImages` and needs the same options.

Turbo does not use pdf2pic for the draw. `worker/ocr.mjs` reads the page box from pdf-parse and draws that page once with GraphicsMagick at 200 dpi. `rasterPixels` turns the box into pixels at 200 dpi, and a long side past 4000 px is scaled down in that same draw, both axes by the same factor. A letter page is 1700×2200. A page with no box is drawn at 200 dpi and only scaled down afterwards, still once. Aspect ratio comes from the page box, so the 768×512 squash does not apply.

What we did not copy: the pdf2pic flags, the 1700 px floor on both axes, and the method rename. A non-letter page stays its own shape.

Follow-up, shipped with the OCR quality control. 200 dpi is enough for body text and too low once the page is shrunk for Tesseract. `preprocess` used to resize OCR input to a width of 2000 px. A letter page drawn at 200 dpi is 1700×2200 and was scaled up slightly. A 17-inch-wide sheet drawn at 200 dpi was then scaled to 2000 px across, about 118 dpi, which dropped the small labels on a schematic. Medium, the saved default, draws at 300 dpi with an 8000 px cap and does not shrink the width. High is 400 dpi and 12000 px. Low keeps the old 200 dpi draw, the 4000 px cap, and the width-2000 shrink, and it skips a page that already has a text layer. Grayscale, normalize, and sharpen stay. A letter page at 300 dpi is 2550×3300. Text rotated on the drawing, or sitting on a wire, can still be missed. Density does not fix that. The control is the OCR group on the Index page, between Measure and the embed batch.

## 1536 — index the PDFs inside ZIM files

- Pull request: https://github.com/Crosstalk-Solutions/project-nomad/pull/1536
- Issue: https://github.com/Crosstalk-Solutions/project-nomad/issues/1445
- Author: YordiLorenzo. Draft, opened 2026-11-03. The ZIM PDF commit is `db6e6ab`. It also carries #1460 (per-page OCR, `ef54af2`) and #1464 (bundled English traineddata, `8fb0d69`).
- Status: **shipped**. One forward walk writes text and OCR paths to disk, then a text pass and an OCR pass open those paths with `getEntryByPath`. HTML is embedded during the walk. The file bar restarts for each pass: green HTML, blue text, orange while a page is recognized, purple while that text is stored. OCR quality, the spelling rules, and the duplicate checks below are what the code does. A short ZIM picture is dropped after it is read. An uploaded picture is kept. OCR text is not written back into the file or the ZIM.

Stock ZIM extraction reads HTML only, so a PDF-shelf ZIM (zimgit, Medicine Essentials, FAS Military Medicine) stored one or two chunks. The pull request reads each `application/pdf` entry with the same reader as an uploaded PDF: per-page text, OCR only on a page whose text layer is empty, then joins pages into passages of about 3000 characters labelled `page 4` or `pages 5-7`. A PDF that cannot be read is skipped with a warning. The rest of the ZIM continues. The pull request's own check: `fas-military-medicine_en_2025-06.zim` went from 1 chunk to 5,615 (26 articles, 1,134 pages OCR'd, about 53 minutes, almost all of it extraction on one core). `zimgit-medicine_en_2024-08.zim` went from 2 chunks to 3,034 in 75 seconds. libzim counts that second file as 1 article, so the progress bar sat near the start until the end. `section_title` is the page range. `article_title` is the entry path. Duplicate copies of one PDF inside a ZIM are left as duplicates (that is #1441). A tesseract.js recognize failure can still escape and take the queue down. That is not fixed in the pull request.

Turbo already does the two dependencies for uploaded files. #1464 matches the vendored `worker/tessdata/eng.traineddata.gz` and the rule that OCR does not download a model at run time. #1460's per-page read matches Turbo's page loop. Turbo's upload policy is stricter than the pull request: every page keeps its text layer and is also rendered, and OCR text is appended only when it has words the layer lacks (`mergePage`).

The gap is `worker/extract.mjs`. `isArticle` keeps `text/html` and `application/xhtml+xml` only. A PDF entry is never emitted.

### Plan

Index every text-bearing entry in the ZIM, not only HTML and not only PDFs. Decided while reviewing this pull request: EPUBs and other readable entries are in scope. Video, audio, fonts, scripts, stylesheets, and map tiles are not. Later, pictures inside a ZIM were left out, and so was plain text inside a ZIM. HTML, PDFs, EPUBs, and Word documents remain. Uploaded pictures are unchanged. When CPU OCR overlaps GPU indexing, a PDF found in a ZIM goes to the front of the OCR queue. The GPU finishes that ZIM's pages, embeds OCR text that is already ready, then the next ZIM.

1. In `worker/extract.mjs`, keep an entry when it can yield text:
   - HTML and XHTML, as today.
   - `application/pdf`, read with the PDF page reader.
   - `application/epub+zip`, unzipped and read as the HTML inside, the same way an uploaded EPUB is read.
   - Picture types already accepted for uploads (`jpeg`, `png`, `gif`, `webp`, `tif`, `bmp`). SVG is read as XML text, not OCR.
   - Plain text, markdown, and `text/vtt` captions.
   - A docx entry, if one appears, through the same document XML read uploads use.
2. Skip what is not readable text. Map tiles (a path containing `tiles/`) are the important case: a map ZIM can hold tens of millions of them, and OCR on a tile is noise. Also skip an image under about 8 KB (icons and spacers) and an image whose OCR comes back with fewer than 8 words of 3 or more characters (a photo). Video, audio, fonts, CSS, and JavaScript stay skipped. A file named README, LICENSE, CHANGELOG, NOTICE, CONTRIBUTING, or CODE_OF_CONDUCT is skipped when its body is under about 200 words. A long one is the manual, and it is indexed.
3. Read PDF and picture bytes with libzim. Do not run them through the HTML parser, and do not decode a PDF as UTF-8 slices.
4. Call the existing reader from `worker/ocr.mjs` inside the extract container. `start_extract` bind-mounts `ocr.mjs` and `worker/tessdata` the same way the OCR container already does. Network stays off. One entry does not start its own container.
5. A PDF page keeps its text layer. The page image is also read. Merge and spelling follow the rules below. The same rules run for uploads and for ZIM PDFs. How hard the page is drawn follows the OCR quality setting.
6. Join PDF page texts into sections of about 3000 characters. The section heading is `page N` or `pages N-M`. Emit the same article JSON the HTML path already emits: `title` is the entry title or path, `sections[].heading` is the page range. `packing.py` already stores that heading as `section_title`, so a hit can say where in the PDF it came from. An EPUB's chapters stay headings from its HTML. A picture is one article, titled by its entry path.
7. A page whose recognize call throws is logged and stored as an empty page. The next page continues. An entry that cannot be opened is emitted with `skip` set, the same way a bad HTML page is skipped today, and the ZIM continues.
8. The lookahead thread keeps its one-byte touch. It does not pull a PDF or a picture into the prefetch worker.
9. The checkpoint remembers the pass and the entry inside it. A stop during HTML resumes that article. A stop during a later pass resumes that path in the on-disk list and does not read the HTML again.
10. While a page is being read, the reader prints the same `ocr page N of M` line uploads already print. The extract stderr path logs that line at debug, and sets the OCR session note, so the Index row shows pages done / total pages and pages per minute in the Articles and art/min columns. Other OCR lines stay info.
11. Each pass fills the file's own bar from empty to full, then the next pass starts again at empty. HTML uses the article count. The text pass and the OCR pass use the counts written during the walk. The file is indexed when the last pass that has work is done. The bar color and the status word say which pass is running. See passes below.
12. A second copy of the same text is not stored. See dedupe below. The first copy is indexed in full.

OCR text is not written back into the PDF, the picture, or the ZIM. The words are what gets chunked and stored in the knowledge base. A re-index that starts over reads the file again. A sidecar cache of page text, keyed by path, size, mtime, and reader settings, can skip that repeat. Rewriting a ZIM to embed a text layer means rewriting the whole archive. Rewriting an uploaded PDF changes the file the user added.

### Passes

One forward walk of the ZIM. HTML is read and embedded during that walk, at today's pace. Every other text path, and every PDF or picture worth reading, is appended to a list on disk. The paths are not kept in RAM. A map file's tiles are not written down. `findByPath` in this libzim returns one entry, so the later passes open those paths directly and do not walk the directory again.

Then the text pass: EPUB, plain text, markdown, captions, docx, SVG text. Short READMEs and the other boilerplate names above are dropped. Anything that matches an HTML article already stored is dropped. The rest is embedded.

Then the OCR pass: PDFs and pictures. A PDF's text layer is read first. When that layer already matches an article from an earlier pass, the page is not drawn. Otherwise the quality setting runs, and the merged text is embedded if it is real content. Pictures are read here. A photo that comes back with fewer than 8 words is dropped after the read.

OCR can run while HTML is still embedding when the two are on different devices. CPU OCR then keeps a couple of cores while Qdrant is writing, one while Qdrant is optimizing, and none when Qdrant refuses a write. A quiet collection gives those cores back. The same device does not do both at once, so OCR waits until that device is between writes.

The file's progress bar is painted per pass, because the stock bar has one theme color for every row:

| Pass | Bar | Status |
| --- | --- | --- |
| HTML | green | HTML |
| Other text, and a PDF text layer that did not need a draw | blue | text |
| Recognizing a page or a picture | orange | OCR |
| Storing the words that recognition produced | purple | OCR text |

A pass with nothing in its list is skipped. Orange and purple trade places on each PDF and each picture: orange while that entry is read, purple while its chunks are embedded, then orange again for the next one. An uploaded PDF or picture does the same. A PDF whose text layer already matches an indexed article, or whose quality setting never draws the page, stays on blue and then the normal embed color for text. It does not turn orange.

### OCR quality

One saved setting, default medium. It applies to PDF pages and pictures, in uploads and in ZIMs. The page is still drawn once from its page box, with aspect ratio kept. Grayscale, normalize, and sharpen stay.

| Setting | Draw | Long side | Which pages |
| --- | --- | --- | --- |
| Low | 200 dpi | 4000 px | A page that already has a text layer is not drawn. An empty page is read. |
| Medium | 300 dpi | 8000 px | Every page is drawn. The width-2000 shrink stays off. |
| High | 400 dpi | 12000 px | Every page is drawn. The width-2000 shrink stays off. |

Medium is the everyday setting, including small print. High is for schematics and tiny labels. A 17-inch sheet at 400 dpi is about 6800 px and stays under the high cap. High reads about 1.8 times as many pixels as medium, so it is slower. Vertical lettering and type drawn on a wire can still be missed at every setting.

### Spelling

The text layer is kept as read. An OCR word is dropped when it is within one edit of a layer word: `catheer` loses to `catheter`, and the misspelling is not stored. An OCR word that is a dictionary word and is not a one-edit neighbor of a layer word is new text (a caption, a label) and is stored. An OCR word that is in no dictionary, and is not a one-edit neighbor of a layer word, is stored unchanged. That is a place name, a drug name, a species name, or a part number. It is not rewritten to the nearest English word.

On a page with no text layer, a lowercase word of at least 5 letters is corrected only when the dictionary has exactly one word within one edit. Two candidates (`form` and `from`) leave the OCR token as read. A capitalized word and any token that contains a digit are never rewritten, so a place name on a scan stays as Tesseract read it.

The word list ships with Turbo. Nothing is downloaded at run time. A technical word that is missing from that list is kept, the same way a place name is kept.

### Dedupe

`point_id` includes the entry path, so the same paragraph from the HTML article and from the EPUB is two points today. Before a chunk is embedded:

- An article fingerprint (simhash of its words) is compared with articles already indexed. A match skips the new article. Both sides need about 200 words, and the shorter text must be at least 80% of the longer one, so a shared preface does not drop a different book. The same PDF at two paths, and a Gutenberg book stored as HTML and again as EPUB, collapse to one copy. The first copy wins. HTML is indexed first, so that copy is the one that stays, with its title and headings.
- A chunk hash (normalized text) is kept for the whole library. A chunk of about 400 characters or more that was already stored is not embedded again, including when it comes from another file. The first copy's citation stays. Adjacent chunks overlap on purpose and are not the same text, so both stay.

The first copy is indexed in full. A later copy of the same words adds nothing.

`tests/test_passes.py` checks page headings `page 4` and `pages 5-7`, that a PDF, an EPUB, and a large picture are kept while a map tile and a tiny icon are not, and that a page heading lands on `section_title`. A real PDF-shelf ZIM is the manual check after that.

The open window loads Python at startup. The new mount flags load the next time Turbo is opened. `extract.mjs` itself is read at the next Start. Do not restart an indexing run to pick this up.

## 1460 and 1464

These two are dependencies of #1536. They are recorded here so a later pass does not port them again.

- #1460 (https://github.com/Crosstalk-Solutions/project-nomad/pull/1460) stopped treating a PDF as one string. pdf-parse joins pages with a `-- N of M --` separator, and a short-document OCR gate then saw those separators and skipped real scans. Turbo already reads every page's text layer on its own and does not use that whole-document gate.
- #1464 (https://github.com/Crosstalk-Solutions/project-nomad/pull/1464) ships `eng.traineddata` with the app, sets `langPath`, and stops the runtime download from the jsDelivr CDN. Turbo already vendors `worker/tessdata/eng.traineddata.gz` and refuses a run-time download.

## Indexing pull requests already covered

Checked against the open and merged Project NOMAD pull requests on 2026-11-07. These are the indexing changes Turbo already does. A pull request about the stock queue, the Stored Files screen, or chat wording is not listed here.

- #1044 (https://github.com/Crosstalk-Solutions/project-nomad/pull/1044), merged. Drop `See also`, `References`, `External links`, and the other non-content headings, and keep table cells separated. Status: **covered**. `worker/extract.mjs` skips those headings and joins a table row with ` | `. Navigation, scripts, and the other chrome classes are left out as well.
- #1386 (https://github.com/Crosstalk-Solutions/project-nomad/pull/1386), open. Stop walking a ZIM from the first entry on every batch. Status: **covered**. One walk stays open and the checkpoint remembers the entry path. A restart walks forward to that path once. This libzim `findByPath` does not return the rest of the archive, so the resume is that one forward walk, not a skip of N articles on every batch.
- #1242 (https://github.com/Crosstalk-Solutions/project-nomad/pull/1242), merged. Do not treat a batch of empty pages as the end of the ZIM. Status: **covered**. An empty page is marked skipped and the walk continues through the rest of the file.
- #1441 (https://github.com/Crosstalk-Solutions/project-nomad/pull/1441), open. Store each chunk's text once inside one file. Status: **covered**, and the check is wider than the pull request. A chunk of about 400 characters or more is stored once for the whole library. A later copy of a long article is dropped too. The rules are under #1536, Dedupe.
- #1517 (https://github.com/Crosstalk-Solutions/project-nomad/pull/1517), draft. Embed several ZIM chunks at once when the embedder is on a GPU. Status: **covered**. Turbo sends a hardware-sized batch in one call, on CPU or on a card. It does not keep a fixed 8 chunks in flight inside the stock loop.
- #1541 (https://github.com/Crosstalk-Solutions/project-nomad/pull/1541), draft. Upsert one article's points together, at most 256 per request. Status: **covered** by the same kind of batching. Turbo sends 128 points per request, and a write can overlap the next embed call. A long article is split across requests.
- #1496 (https://github.com/Crosstalk-Solutions/project-nomad/pull/1496), open, and #873 (https://github.com/Crosstalk-Solutions/project-nomad/pull/873), merged. Do not pause a second between ZIM batches when the embedder is not NOMAD's own CPU Ollama. Status: **covered**. Turbo does not insert that pause. It uses its own embed server.
- #763 (https://github.com/Crosstalk-Solutions/project-nomad/pull/763), #890 (https://github.com/Crosstalk-Solutions/project-nomad/pull/890), and #994 (https://github.com/Crosstalk-Solutions/project-nomad/pull/994), all merged. Keep a dense chunk inside the embedder's context, and do not retry a bad call 30 times. Status: **covered** by splitting. A chunk that would pass 2048 tokens is split before the call, and the stored words are kept. A failed call is tried 3 times. A 400 splits that input instead of retrying it. Turbo does not send `num_ctx: 8192`.
- #352 (https://github.com/Crosstalk-Solutions/project-nomad/pull/352), merged. Do not retry an embed job 30 times when Ollama is not installed. Status: **covered** by the same retry cap. Turbo does not queue stock `file-embeddings` jobs for its own work.
- #1135 (https://github.com/Crosstalk-Solutions/project-nomad/pull/1135), merged. Do not create the Qdrant payload indexes again on every document. Status: **covered**. Turbo checks the collection when a run needs it. An index that already exists is left alone.
- #257 (https://github.com/Crosstalk-Solutions/project-nomad/pull/257), merged. Index EPUB uploads. Status: **covered** for uploads and for EPUB entries inside a ZIM. Turbo reads the HTML inside the archive. It does not follow the OPF spine order.
- #1100 (https://github.com/Crosstalk-Solutions/project-nomad/pull/1100), merged. Read a `.docx` as text instead of as a zip of XML. Status: **covered**. Uploads read `word/document.xml` word by word. A `.docx` inside a ZIM uses the same file with the tags stripped. Turbo does not use mammoth.
- #1227 (https://github.com/Crosstalk-Solutions/project-nomad/pull/1227), merged. Delete Qdrant points when the file is gone. Status: **covered** by Complete repair. That scan lists the sources in the index and deletes points for a library file that is no longer on disk. It does not run by itself during indexing. A file that is still on disk is left in place.

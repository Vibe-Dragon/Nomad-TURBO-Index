# How Turbo indexing differs from stock NOMAD

Checked against Project NOMAD 1.35.2, the image `ghcr.io/crosstalk-solutions/project-nomad:v1.35.2`. That is the latest GitHub release as of this page. Turbo runs beside that install. It does not patch the admin container.

## Dirent-cursor batch seek

Project NOMAD 1.35.2 uses a dirent-cursor batch seek. It seeks the next ZIM batch with a dirent cursor. `openZimBatchRange` takes `resumeAtDirent`, opens `archive.iterByPath()`, and returns `iterByPath().offset(resumeAtDirent, range.size)`. The batch is 50 articles. The job then queues the next batch with the next cursor. A batch that has no cursor still walks from the start of the archive until it has skipped the saved article count, and the log says so.

Turbo still does this differently. One reader stays open for the file. The checkpoint stores the entry path and, once known, its dirent in `iterByPath()` order. A restart seeks with `offset(dirent, size)` and checks that the entry is still that path. A checkpoint that only has a path walks forward once, records the dirent, and stays there. Turbo does not open the archive again on every batch. While that read is going, a second reader loads upcoming clusters into the cache libzim already chose, up to half of that cache. It does not raise the cache. `findByPath` still returns one entry. A cursor that lands on a different path, or past the end of the archive, walks from the start instead.

## Own embed server

Stock queues `file-embeddings` jobs and embeds 8 chunks per call through Ollama. When that embedder is not on a GPU, the next ZIM batch waits one second.

Turbo leaves that queue paused and talks to one `llama-server` it started from the Ollama image NOMAD already has. The model is loaded once. The chunk count is how many chunks the next HTTP call sends. It does not start a second server. That server can open a second sequence when the card has room for another 2048-token context. Each sequence keeps that context. Long calls that get slower send it back to one sequence. Turbo does not insert the one-second pause.

The call size is a chunk count. There is no fixed ceiling and no graphics-card name in the math. **Recommended** is the safe call for the computer running Turbo. **Auto** follows finished embed calls and can go above that start. A call already in flight finishes at the old width.

## Long pages

Stock reads the HTML it cleaned and chunks that text at about 1500 tokens, with about 150 tokens of overlap. A chunk that estimates over 1600 tokens is truncated. The tail of that chunk is not stored.

Turbo does not bring a page-size cap back. The reader keeps the body of the page, parses a long page in slices so one book does not force a giant DOM into memory, and `split_chunks` walks every character. A piece that would not fit the embedder is split, and the stored words are kept. The chunk size is still about 1500 tokens, with about 150 tokens of overlap.

## Chrome

Leaving out navigation and the other furniture is not a Turbo-only change. Stock 1.35.2 removes the same tags and classes Turbo skips: `script`, `style`, `nav`, `header`, `footer`, `noscript`, `iframe`, `svg`, and the classes `navbox`, `sidebar`, `infobox`, `mw-editsection`, `reference`, `reflist`, `toc`, `noprint`, `mw-jump-link`, `mw-headline-anchor`, `navbar`, `hatnote`, `ambox`, `sistersitebox`, `portal`, `geo-nondefault`, and `authority-control`. Both drop the headings See also, References, External links, Further reading, Notes, Bibliography, and Navigation, and both join a table row with ` | `. Both also remove `[role="navigation"]` and `#coordinates`. Turbo's reader treats a `navigation` role and an id of `coordinates` as chrome, the same way stock's selector list does.

## Article and section headings

Pull request [1570](https://github.com/Crosstalk-Solutions/project-nomad/pull/1570) is still open, so 1.35.2 still embeds `search_document: ` plus the bare section text. The title and heading are stored on the point. The embedder does not read them.

Turbo puts that line on the text sent to the embedder only. The stored chunk stays the bare section text. A section heading that differs from the title becomes `Title > Heading`. The line is counted in the token budget. A chunk that would no longer fit is split.

## Upload PDFs and pictures

Stock reads an upload inside the admin container. A PDF is one text extract of the whole file. If that extract is under 100 characters, every page is rasterized with pdf2pic and no page size, which squashes the page. Pull request [1567](https://github.com/Crosstalk-Solutions/project-nomad/pull/1567), which would keep the aspect ratio, is still open and is not in 1.35.2. A picture is resized to a width of 2000 and read with Tesseract. A ZIM walk emits `text/html` and `application/xhtml+xml` only. Pull request [1536](https://github.com/Crosstalk-Solutions/project-nomad/pull/1536), which would index PDFs inside a ZIM, is still an open draft. Pull request [1460](https://github.com/Crosstalk-Solutions/project-nomad/pull/1460), per-page OCR, is still open.

Turbo reads plain text, DOCX, and EPUB on the host. For a PDF or a picture it runs the OCR libraries that are already in the NOMAD image, then chunks and embeds that text with its own server. It does not hand the file to NOMAD's embedder. Every PDF page keeps its text layer. A page is drawn when it has no text, or when **OCR every PDF** is on. The draw uses the page box, so the aspect ratio stays. **OCR** on the Index page is Low, Medium, or High. Medium is the default: 300 dpi and an 8000 pixel cap, with no width-2000 shrink. A picture is read either way. A PDF or picture that is not English is left out. The recognized words stay in the knowledge base. They are not written back into the file.

**ZIM documents** stays off until it is turned on. Off, PDFs, Word documents, and EPUBs inside the ZIM stay unread. Uploaded PDFs and pictures are still read. On, those inner files are listed during the page walk and read after every ZIM's pages and every upload are stored. Pictures inside a ZIM are left out, and so is plain text inside a ZIM. A ZIM that is already indexed stays as it is until it is walked again.

## Ingest policy

On Start, Turbo remembers the ingest policy, sets it to Manual, and pauses only the `file-embeddings` queue. It waits until that queue has no active job, then reads the saved offsets. Asking NOMAD a question still searches. Other queues are not paused.

On Stop, the policy goes back to the value found at Start. A run that found Manual stays Manual. Unfinished files are handed to stock NOMAD only when the run started on Always, and a half-finished file in that case stays Manual so NOMAD does not start that file over.

Stop does not delete the index. Delete is a separate button. It asks you to type DELETE, removes stored vectors and ingest rows, and does not start indexing. ZIM files and uploads stay on disk.

The vectors Turbo stores are the ones stock NOMAD can search. The model, the prefix, and the `source` path are in the [stock contract](stock-contract.md).

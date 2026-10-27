# NOMAD Turbo

Nomad TURBO Index speeds up a stock [Project NOMAD](https://github.com/Crosstalk-Solutions/project-nomad) knowledge base from outside the admin container. It detects the machine it is on and sizes one embed call from that hardware. This copy was tested with Project NOMAD 1.35.1.

## How Turbo indexing differs from stock NOMAD

- Turbo runs outside the admin container and does not patch it.
- Turbo uses its own embed server and larger calls instead of stock's small embed jobs.
- Turbo keeps the meat of a long HTML page and chunks it instead of cutting the page short.
- Turbo resumes a ZIM from the article it stopped on. It does not count from the first article on every batch.
- Turbo leaves out navigation, scripts, and similar chrome.
- Turbo reads upload PDFs and pictures itself.
- Turbo pauses only file embedding and restores the ingest policy found at start.
- The call size scales with detected hardware and can be set to 110%.

## Turbo runs outside the admin container and does not patch it

Stock indexing lives inside the NOMAD admin container. Turbo never copies itself into that image and never edits compose. It asks Docker which containers are already running, then does its own work beside them.

The embed server is a separate container Turbo creates and is the only container Turbo removes. NOMAD's admin, Redis, MySQL, Qdrant, and Ollama containers stay up.

```python
name = "nomad-turbo-embed"
_remove_ours(name)
command = [
    "docker",
    "run",
    "-d",
    "--name",
    name,
    "--label",
    "nomad-turbo=embed",
```

`_remove_ours` refuses any name that does not start with `nomad-turbo-`. A porter who wants this inside stock NOMAD would call the same chunking and store helpers from the admin process. This package does not do that port.

## Turbo uses its own embed server and larger calls instead of stock's small embed jobs

Stock NOMAD queues small embed jobs on `file-embeddings` and talks to Ollama one batch at a time. Turbo leaves that worker idle and talks to one `llama-server` it started from the same Ollama image NOMAD already has. The model is loaded once. The slider changes how many chunks the next HTTP call sends. It does not start a second server.

```python
def batch_width(self, percent: int) -> int:
    """Chunks the next call sends. One server. The width follows the computer running Turbo."""
    gpu_total = self.total_mib if self.use_gpu and self.total_mib > 0 else None
    return embed_call_budget(
        percent,
        gpu_total_mib=gpu_total,
        host_total_mib=self.host_total_mib or 0,
        cpu_count=self.cpu_count,
    )
```

A call that is already in flight finishes at the old width. The next call uses the new one.

## Turbo keeps the meat of a long HTML page and chunks it instead of cutting the page short

Stock can stop reading a page once it hits a size cap. Turbo does not bring that cap back. The ZIM reader keeps the body of the page no matter how long it is, and `split_chunks` walks every character of that text.

```javascript
// The meat of a page is indexed no matter how long the page is. Navigation,
// scripts, author boxes, and similar chrome are left out. A long page is
// parsed in slices so one book cannot force a giant DOM into memory. The
// text is still stored in full and chunked later. Do not bring back a size cap.
```

```python
def split_chunks(text: str) -> list[str]:
    """Split the whole sanitized page into stock-sized chunks.

    The walk covers every character. A long page becomes many chunks instead
    of being cut off. fit_embed_inputs splits a piece that is still too long
    for the embedder. The prefix is applied later, not here.
    """
```

The chunk size is still stock's: about 1500 tokens, with about 150 tokens of overlap. A piece that is still too long for the embedder is split, and the rest of the file continues.

## Turbo leaves out navigation, scripts, and similar chrome

The reader skips the tags and classes that are page furniture. The article text is what gets chunked and stored.

```javascript
const SKIP_TAGS = new Set([
  "script", "style", "nav", "header", "footer", "noscript", "iframe", "svg",
]);

const SKIP_CLASS = new Set([
  "navbox", "sidebar", "infobox", "mw-editsection", "reference", "reflist",
  "toc", "noprint", "mw-jump-link", "mw-headline-anchor", "navbar", "hatnote",
  "ambox", "sistersitebox", "portal", "geo-nondefault", "authority-control",
]);
```

Sections whose heading is "References", "See also", "External links", and the same kind of footer are left out too. The body of the article is not.

## Turbo resumes a ZIM from the saved article instead of counting from the first article on every batch

Stock NOMAD stores how many articles it has finished. The next batch opens the ZIM, starts at the first article, and counts forward until it reaches that number. The batch after that opens the file and counts from the first article again.

Turbo stores the path of the last article it checkpointed. One reader opens the ZIM and walks forward until it reaches that path. Articles it passes are not parsed. The file stays open, and later embed calls continue with the next article. They do not go back to the start and count up to the saved number.

This build of libzim can look up one path, and that lookup returns only that entry, not the rest of the file. The reader cannot jump to the saved article and then iterate the tail. It walks forward once, when the reader opens the file, and then stays there. A checkpoint that has no path still uses the article count.

```javascript
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
```

When the checkpoint has a path, the count skip is left at 0 and the path is what the reader uses:

```python
    proc = start_extract(
        install,
        str(work.host),
        0 if work.path else work.articles,
        work.path,
        work.articles,
    )
```

A crash resumes after the last article that was checkpointed. The checkpoint is kept. An article path that is already in the index is not embedded again. A plain upload that already has points is left as indexed.

## Turbo reads upload PDFs and pictures itself

Stock embeds an upload from inside the admin container. Turbo reads plain text, DOCX, and EPUB on the host. For a PDF or a picture it runs the OCR libraries that are already in the NOMAD image, then chunks and embeds that text with its own server. It does not hand the file to NOMAD's embedder.

```javascript
// Read one uploaded PDF or image with the OCR libraries in the NOMAD image.
// This does not call NOMAD's embedder. Turbo chunks the text and embeds it.
```

Every PDF page keeps its text layer and is also rendered at 200 dpi. OCR text is added when the page image has words the text layer does not. A picture is grayscale, normalized, sharpened, fit inside 2000 pixels, and read with the English model shipped in `worker/tessdata`. A file the reader cannot open stays marked "needs NOMAD" and is not queued.

## Turbo pauses only file embedding and restores the ingest policy found at start

On Start, Turbo remembers the ingest policy, sets it to Manual, and pauses only the `file-embeddings` queue. It waits until that queue has no active job, then reads the saved offsets. Asking NOMAD a question still searches. Other queues are not paused. Re-embed and Reset in NOMAD still delete stored points.

```python
        previous = get_policy(install)
        (state_dir() / "policy").write_text(previous)
        set_policy(install, "Manual")
        log.info(
            "NOMAD ingest policy was %s. Turbo keeps Manual until this run stops, then puts that back.",
            previous,
        )
        if session:
            session.set_phase("waiting")
            session.set_note("Waiting for NOMAD to finish its current embed batches.")
        queuectl(install, "pause")
```

On Stop, the policy goes back to the value found at Start. A run that found Manual stays Manual. Unfinished files are handed to stock NOMAD only when the run started on Always, and a half-finished file in that case stays Manual so NOMAD does not start that file over from the first article.

```python
def stock_should_continue(previous: str | None) -> bool:
    """Hand unfinished files to stock NOMAD only when the run started on Always.

    Manual means stock indexing stays off. Turbo keeps the checkpoint.
    """
    return remembered_policy(previous) == "Always"
```

Stop does not delete the index. Delete is a separate button. It asks you to type DELETE, removes stored vectors and ingest rows, and does not start indexing. ZIM files and uploads stay on disk.

## The call size scales with detected hardware and can be set to 110%

There is no fixed chunk ceiling and no graphics-card name in the math. When a GPU is present, the safe 100% call is computed from that card's memory. Otherwise it is computed from system RAM and the CPU count. The same inputs always give the same count. A machine with more memory gets a 100% call at least as large as a machine with less. A machine with no GPU still gets a finite plan.

The slider runs from 10% to 100% of that budget. 110% is an overclock past it: strictly more chunks than 100% on the same hardware. A new install does not start at 110%. The default is 55% (`reserve_ratio` 0.45) until you move the slider. One embed server stays up either way. 110% can be slower if the card is already full. It is there for people who want to try past the safe size.

```python
full = safe_call_chunks(
    gpu_total_mib=gpu_total_mib,
    host_total_mib=host_total_mib,
    cpu_count=cpu_count,
)
pct = clamp_boost_percent(percent)
if pct > 100:
    extra = max(1, full * (pct - 100) // 100)
    return full + extra
```

`python3 -m nomad_turbo plan` prints the hardware-derived size. It does not pause NOMAD or start containers.

## Disclaimer

I update this when I get to it. It can fall behind Project NOMAD, and it may not match the version you are running. Treat an old copy as untried on a NOMAD release it was not written for.

I am not responsible if Turbo messes up your index, your search results, or anything else on your machine. Use it only if you accept that. Back up what you care about before you click **Start indexing** or **Delete index**.

## FAQ

### Is this official Project NOMAD?

Stock Project NOMAD does not ship this. Turbo is a side tool that runs next to a NOMAD install you already have.

### Will it change my NOMAD containers?

Turbo does not edit compose and does not copy itself into the admin image. It starts one extra container, `nomad-turbo-embed`, and that is the only container it removes. The admin, Redis, MySQL, Qdrant, and Ollama containers stay up.

### Will installing it, or opening the window, start indexing or delete my library?

Installing the `.deb` does not start indexing and does not delete stored points, ZIM files, or uploads. Opening the window scans the library list. The icon does not start or stop a run.

### Can this mess up my knowledge index?

Turbo writes vectors into the same Qdrant collection NOMAD searches. A bug, a NOMAD update this copy has not caught up with, or **Delete index** can change or remove those points. I am not responsible if your index ends up wrong or empty. Keep your own backup.

### What does Delete index remove?

You have to type DELETE. It removes stored knowledge points and ingest rows. Indexing does not start. ZIM files and uploads stay on disk.

### What happens to Manual and Always?

**Start indexing** remembers the ingest policy, sets it to Manual, and pauses only the `file-embeddings` queue. Asking NOMAD a question still searches. **Stop** puts the saved policy back. A run that found Manual does not hand unfinished files to stock NOMAD. A run that found Always can hand unfinished files back, and a half-finished file in that case stays Manual so NOMAD does not start that file over from the first article.

### Do I need a particular graphics card?

A GPU is used when the machine has one. With no GPU, the call size comes from system RAM and the CPU count. The package does not depend on a specific driver or a specific card. It installs on Debian, Ubuntu, and Linux Mint. It needs Python 3, GTK, Docker usable by your user, and NOMAD already running.

### What does the slider do, and what is 110%?

The slider is a share of the budget detected on your computer. 10% to 100% stays inside that budget. 110% is an overclock past it. A new install starts at 55%. One embed server stays up. The slider changes how many chunks the next call sends. 110% can be slower when the card is already busy.

### Will this stay current when I update NOMAD?

It stays current when I update it. A new NOMAD release can change paths, the index format, or the embed worker, and this copy will not know until I change it.

### What gets left out of a page?

Navigation, scripts, author boxes, and the same kind of chrome are left out. The body of a long page is kept and split into chunks. There is no size cap that cuts the rest of the page off.

### What if a file says "needs NOMAD"?

Turbo could not read that file, so it is not queued. The file stays on disk.

### Why are Start indexing, Delete index, Scan, and Start repair off?

The Qdrant line under the page switch has not said the collection answered. Those buttons stay off until it reads `Qdrant is ready for changes.` Rescan does not wait. The messages are in Common log lines.

### What does an HTTP error in the log mean?

`qdrant 503` and `HTTP 503` are different programs. The first is the index. The second is Turbo's embed server. A timeout keeps the current file. An HTTP 400 from the embedder is one bad chunk, and the rest of the file continues. Common log lines has the wording.

### What does Repair delete?

**Scan** changes nothing. It does not pause embedding. **Start repair** pauses file embedding, then deletes points only for ZIM and upload files that are already gone. It also removes that file's ingest row, checkpoint, and leftover embed jobs. Points for a file that is still on disk stay. A count that does not match is reported and left in place. Repair does not embed files again, and it does not wipe the whole index. The Repair section describes the page.

### Is there a license?

Yes. This project is under the MIT License. The terms are in the `LICENSE` file. Copyright 2026 Vibe-Dragon.

## Install the package

The `.deb` is the one-file installer for Debian, Ubuntu, and Linux Mint. Open it from the file manager and install it. That does not start indexing and does not delete anything already stored.

The package depends on Python 3 and GTK. It does not depend on a specific NVIDIA driver or a specific graphics card. After install, the menu entry **Nomad TURBO Index** runs `/usr/bin/nomad-turbo`. Docker has to be usable by your user, and NOMAD has to be running. Turbo finds that install when it starts.

To build the package from this tree:

```bash
packaging/build-deb.sh
```

The script writes `dist/nomad-turbo_<version>_all.deb`. It does not install the package.

## Use

Click the icon. The window scans the library. Opening it does not start or stop indexing.

**Start indexing** pauses NOMAD's file embedding and continues from the saved article. The reader opens the ZIM once, walks to that article, and keeps going from there. **Rescan** reads the library again. It does not pause NOMAD or start indexing. **Stop** finishes the embed call already running. It does not delete the index. When the run started on Manual, unfinished files are not handed back to stock NOMAD. **Delete index** sits at the right end of the button row. You must type DELETE. Indexing does not start. ZIM files and uploads stay.

The **Embed batch** slider is the boost. The line under it reads `Set 55% · <chunks> chunks per call. Embed server: idle.` The chunk count is the budget for that setting on your machine. Once a run has started the server, that line says `One embed server is running.` Above 100% it adds `Overclock past the detected budget.` The next line is `Card memory in use: <used> / <total> MiB`, or `No GPU detected. The embed server will use the CPU.` Hover any button, checkbox, column heading, progress bar, or those lines for what they do.

**Nomad TURBO Index** at the top right opens this repository. The Qdrant line sits under the Index / Repair switch. **Start indexing**, **Delete index**, **Scan**, and **Start repair** stay off until that line says `Qdrant is ready for changes.` **Rescan** does not wait.

The window edges resize the whole window. Each page has a drag bar between the file list and the controls. The log bar sits under the Log row. Column edges resize. Those positions are remembered.

If Turbo dies while the embed queue is paused, the next time the window opens it puts the ingest policy back and resumes NOMAD embedding. If a repair delete may still be running, embedding stays paused. `nomad-turbo recover` does the same thing.

**Stay on after the last file** keeps Turbo in control when the queue is empty, so files that arrive later are indexed by Turbo. Off, which is the default, Turbo stops and the ingest policy returns to the value found at Start. When the run started on Manual, stock NOMAD does not start embedding.

Closing the window while a run is going hides it to the tray. The tray menu can show the window or stop the run. The tray icon is the plain logo when idle, the logo on fire while indexing, and blue while a repair is running. Clicking the desktop icon brings the window back. It does not start or stop the run.

## Index

The Index page is the library list.

- **File.** Blue names are ZIM files. Amber names are uploads. Hover a row for the full path.
- **Kind.** ZIM library, or a file uploaded to the knowledge base.
- **Status.** Waiting, indexing, indexed, partial, failed, or needs NOMAD.
- **Progress.** How far this file is through its articles.
- **Articles.** Stored so far, and the total when it is known.
- **Chunks.** Embedded pieces stored for this file. A long page is many chunks.
- **Size.** The file on disk.
- **Rate.** Articles per minute swing with page length. Chunks per second stay steadier.
- **ETA.** Time left for this file at its recent article rate.

The file being indexed has its own bar. The bar under it is the whole library, weighed by file size, and the time left for what remains. The stats line counts articles per minute, chunks per minute, skipped pages, and bad pages. Skipped pages had no text. Bad pages could not be read or parsed. A long page is indexed in full.

The list keeps the active file near the top. Click a heading to sort. Drag a column edge to resize it. That width is remembered.

## Resource spikes while indexing

A run trades work between the embed server, Qdrant, and the disk. One of them getting busy while another goes quiet is that handoff. The link at the top right only opens this repository. It does not start a job, and it does not talk to Qdrant or the embed server.

### The GPU stays busy and articles per minute drop

A long page becomes many chunks. The embed server spends that time on the GPU. Articles per minute fall because the page takes a while. Chunks per minute on the stats line are the sign the run is still working. A quiet disk during that stretch is normal.

### The GPU is quieter and the embed server uses more CPU

Short pages leave the graphics card waiting between calls. The embed server then spends more of its time on the CPU, packing those small chunks. Dictionaries and other short-article libraries look like this. Chunks per minute can stay healthy while the GPU graph sits well under full.

### CPU jumps for a few minutes and the GPU drops

Qdrant builds a new segment when the one it is writing fills. That build is CPU work: the payload indexes, then the search graph. Turbo waits until each batch is stored, so the embed server sits idle and GPU use falls until the build finishes. The collection can show yellow while those new points are not in the search graph yet. Closing a browser does not stop the build. Leave Qdrant running. A restart replays the unfinished write and can use more memory than the build did.

### CPU stays high after the Knowledge Base page is closed, and the article count stops

Project NOMAD's Knowledge Base page asks Qdrant to set `active` on every stored point that does not already have it. The page can time out and be closed while Qdrant is still doing that rewrite. Turbo waits on the current file, and the GPU drops because new batches are not being stored. Points Turbo stores already include `active`. The rewrite is the older points. Leave Qdrant running, and leave that page closed during a run.

### Memory climbs over a long run

Qdrant keeps the search graph in RAM. The graph grows as more of the library is stored. That climb is the index. A restart loads the graph again from disk, and a large index can use more memory on the way back up than it did while it was running.

### The disk is busy and the next batch waits

Reading a ZIM and writing the index share the disk. When the disk is saturated, Turbo waits before the next upsert. A download onto the same disk slows both.

### Start spends a short while loading the model

The embed server loads the model once when a run starts. The log says `embedder not ready yet` until that load finishes. Card memory steps up by the model and stays there. The usage line then says `One embed server is running.`

### One page holds the article count

A very long page is read and split before its chunks are stored. The article count sits on that page while the reader and the embed server work through it. The next article shows up when that page has been stored. Chunks per minute still move if the page is producing chunks.

## Log

The Log row shows and hides the log. **Errors**, **Info**, and **Debug** only change which lines the pane draws. The log file still records every line. **Follow** jumps to the newest line and stays there. Scroll up to hold your place. Turn Follow off and the page stays where it is while new lines arrive.

**Export log** saves that file, including lines the checkboxes are hiding. The suggested name is `nomad-turbo-log.txt`.

On a long run the pane drops lines it has already shown, so the window still answers the close button. The file keeps the session.

## Repair

Repair is the second page. It does not embed files, and it does not wipe the collection. It finds index entries whose library file is gone, and **Start repair** can delete those entries. **Delete index** is the other button, on the Index page, and it asks you to type DELETE.

**Quick repair** and **Complete repair** are the two modes. Both **Scan** and **Start repair** use the mode that is selected.

**Scan** only reads. It does not pause NOMAD, does not change the ingest policy, and does not delete points. Quick compares the files on disk with ingest rows and checkpoints. Complete also asks Qdrant to list every `source` in the collection, when the collection is answering. If it is not answering, the scan still compares files, rows, and checkpoints, and it says the source list did not return. Stop is checked between those steps. The source list itself runs until Qdrant answers.

The list has File, Problem, Points, Progress, Status, and Elapsed. A scan marks a missing library file **Would delete** and a file that is still on disk **Report only**. Hover a row for its path and the detail. The line under the bars counts the library, ingest rows, checkpoints, and, on a complete scan, the sources Qdrant listed.

**Start repair** is what changes the index. It remembers the ingest policy, sets Manual, pauses `file-embeddings`, and waits until that queue has no active job. The tray icon turns blue. The desktop icon does not change. It then repairs one file at a time.

What a repair deletes:

- Points for a ZIM or an upload that is no longer on disk.
- That file's ingest row, Turbo checkpoint, and leftover NOMAD embed jobs.
- On a complete repair, orphan points whose `source` is a library path that is gone, even when the only trace was the index itself.

What a repair leaves in place:

- Points for a file that is still on disk. A point count that does not match the ingest row is reported, not deleted.
- A file that has points and no ingest row.
- An ingest row that says indexed when the source list has no points for it.
- Points stored outside `zim/` and `kb_uploads/`. NOMAD's own docs live outside those folders, so they stay.
- An unfinished file whose point count disagrees with its checkpoint. Deleting those points would throw away the progress the checkpoint is holding.

Stop works between files, or between scan steps. A delete that was already sent runs until the index finishes it. Stop does not wipe the collection. If a delete does not return, the embed queue stays paused so a second delete is not started on top of it. Start repair again when the index is idle. While a download is in progress, or the disk is busy, repair waits instead of deleting. Closing the window during a repair hides it to the tray. The run keeps going.

## Commands

The desktop icon runs `nomad-turbo toggle`. It opens or raises the window. It does not start or stop indexing.

- `nomad-turbo plan` prints the call size for the computer it is running on. It does not pause NOMAD or start containers.
- `nomad-turbo status` prints whether indexing is running, or only the window is open.
- `nomad-turbo stop` stops after the embed call that is already running.
- `nomad-turbo recover` puts the ingest policy back and resumes NOMAD embedding after a crash that left the queue paused. A repair delete that may still be running leaves embedding paused.

## Common log lines

These are the lines people meet when Qdrant or the embed server is not ready to take work. The wording is the wording in the window and the log.

### Qdrant is not ready for changes

The status line under the page switch is one of these until the collection answers:

- `Qdrant is missing. Not ready for changes.` The container state replaces `missing` when Docker reports one.
- `Qdrant is running and has not passed the health check. Not ready for changes.`
- `Qdrant is running. The ready check has not passed. Not ready for changes.`
- `Qdrant is still writing. Do not restart it or delete the index.`

A large collection can be up and still on the last of those while it finishes a write. **Start indexing**, **Delete index**, **Scan**, and **Start repair** stay off. The second line names the container, how long it has been up, the health check, the ready check, and whether the collection answered. When the collection does not answer, that line also says a restart would replay the write, and it says whether the open segment is still gaining vectors. When Docker returned a log line, it is included as `Last log:`.

`Qdrant is ready for changes.` is the line that turns those buttons on.

### Qdrant stopped taking a write

A write that does not reach Qdrant is tried again. The log says `qdrant write timed out. Trying that batch again.`

If it is still not taking the write, Turbo stays on that file. The status line says `Qdrant is still writing this file. Do not restart Qdrant or delete the index. Turbo is waiting here.` The log says `qdrant is not taking writes. Staying on <file> at article <n>: qdrant unreachable: ...`

`qdrant 400: ...`, `qdrant 503: ...`, and the other `qdrant <code>:` lines are answers Qdrant did return. Turbo does not treat those as "wait on this article." It tries that file once more from the same article (`will try <file> again from article <n> after an error`). A second failure at that same article is `file failed`.

### The embed server is not ready

Before Start, the usage line says `Embed server: idle.` After Start, Turbo waits while `llama-server` loads the model. New waits are logged as `embedder not ready yet: ...`. That wait is about three minutes. The usage line then says `One embed server is running.`

A call that gets HTTP 408, 429, 500, 502, 503, or 504 is logged as `embedder call failed (HTTP 503: ...). Trying again.` The code in the message is the code that came back. Turbo tries that call up to three times. The error that remains is `embedder HTTP 503: ...`. The file is then tried once more from the same article. A second failure at that article is `file failed`.

HTTP 400 is one input the embedder refused, not a dead server. The log says `splitting one chunk of <file> (<path>) that did not fit the embedder`. If that one chunk is still refused, the log says `skipping one chunk of <file> (<path>) the embedder rejected`. That chunk counts as bad. The rest of the file continues.

`the embed server is not running` means a call was ready and the server process was already gone.

## Stock contract

Turbo stores vectors stock NOMAD can search. Do not change these if you port the indexer:

- Model `nomic-embed-text:v1.5`, 768 dimensions, cosine distance, collection `nomad_knowledge_base`.
- The embedder sees `search_document: ` plus the chunk. The stored payload does not include that prefix. NOMAD queries with `search_query: `.
- `source` is the container path, such as `/app/storage/zim/some.zim`.

```python
def embed_input(chunk: str, prefix: str = DOCUMENT_PREFIX) -> tuple[str, str]:
    """Return (prefixed model input, payload text without the prefix).

    The model sees `search_document: ` plus the chunk, which is the stock
    NOMAD document prefix. The text stored in Qdrant omits that prefix.
    NOMAD queries with `search_query: ` through its own Ollama. Turbo does
    not change either prefix.
    """
```

## What this does not do

It does not upgrade NOMAD, edit compose, or copy files into the admin container. It does not start one embed server per slider notch. It does not delete points unless you type DELETE or you run a repair for a file that is gone.

The license is MIT. See `LICENSE`.

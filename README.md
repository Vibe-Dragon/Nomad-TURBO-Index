# NOMAD Turbo

Nomad TURBO Index speeds up a stock Project NOMAD knowledge base from outside the admin container. It detects the machine it is on and sizes one embed call from that hardware.

## How Turbo indexing differs from stock NOMAD

- Turbo runs outside the admin container and does not patch it.
- Turbo uses its own embed server and larger calls instead of stock's small embed jobs.
- Turbo keeps the meat of a long HTML page and chunks it instead of cutting the page short.
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
    """Chunks the next call sends. One server. The width follows this machine."""
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

### What does Repair delete?

**Scan** changes nothing. **Quick repair** and **Complete repair** remove points for library files that are already gone. Points for a file that is still on disk stay. Repair does not embed files again.

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

**Start indexing** pauses NOMAD's file embedding and continues from the saved place. **Rescan** reads the library again. It does not pause NOMAD or start indexing. **Stop** finishes the embed call already running. It does not delete the index. When the run started on Manual, unfinished files are not handed back to stock NOMAD. **Delete index** sits at the right end of the button row. You must type DELETE. Indexing does not start. ZIM files and uploads stay.

The **Embed batch** slider is the boost. The line under it shows the chunk count that setting will send. Hover any button, checkbox, column heading, progress bar, or that line for what it does.

**Stay on after the last file** keeps Turbo in control when the queue is empty, so files that arrive later are indexed by Turbo. Off, which is the default, Turbo stops and the ingest policy returns to the value found at Start. When the run started on Manual, stock NOMAD does not start embedding.

The **Repair** tab checks the index. **Scan** changes nothing: it does not pause embedding or delete points. **Quick repair** and **Complete repair** remove points for library files that are gone. Points for a file that is still on disk stay. Repair does not embed files again.

Closing the window while a run is going hides it to the tray. The tray icon is the plain logo when idle, the logo on fire while indexing, and blue while a repair is running. Clicking the desktop icon brings the window back. It does not start or stop the run.

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

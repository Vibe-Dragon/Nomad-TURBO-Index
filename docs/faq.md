# FAQ

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

**Start indexing** remembers the ingest policy, sets it to Manual, and pauses only the `file-embeddings` queue. Asking NOMAD a question still searches. **Stop** puts the saved policy back. A run that found Manual does not hand unfinished files to stock NOMAD. A run that found Always can hand unfinished files back, and a half-finished file in that case stays Manual so NOMAD does not start that file over.

### Do I need a particular graphics card?

A GPU is used when the machine has one. With no GPU, the call size comes from system RAM and the CPU count. The package does not depend on a specific driver or a specific card. It installs on Debian, Ubuntu, and Linux Mint. It needs Python 3, GTK, Docker usable by your user, and NOMAD already running.

### What does the chunk count do?

The field is how many chunks the next embed call sends. **Recommended** sets it to the safe call detected from the memory and CPU of the computer running Turbo. That is a start, not a maximum. Typing a number changes the next call. The call already running is left alone. One embed server stays up. A larger count does not change GPU clocks. A saved percent from an older version opens as the chunk count that percent meant. **Auto** follows finished embed calls. It raises the count while a larger count is still faster, and a tie does not force another climb. It can go above the old top. It steps down when a larger count is clearly slower, or the call fails. A brief slow patch that the next calls do not repeat is left alone. Article length does not move the count, and a pace from another article length is not used. Another publication's pace is not a verdict.

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

### What does Then index do?

Then index does nothing until a measurement is already going. **Then index** does not start indexing and it does not start a measurement. During a measurement it arms that same pass to keep indexing after the samples. It does not restart the pass or clear the samples already taken. The half shows that it took effect. **Measure** by itself still stops when it started from idle.

### Why are some controls buttons?

**Host system graphs**, **Hide finished**, **Stay on**, **Parallel OCR**, **ZIM documents**, **Auto**, **Errors**, **Info**, **Debug**, **Follow**, the export session choice, and **Quick repair** / **Complete repair** are buttons rather than checkboxes. **Parallel OCR**, **OCR workers**, and **ZIM documents** are on the Settings page. **Parallel OCR** is experimental and off until it is turned on. Off, one file is read or indexed at a time. **OCR workers** is a slider and a number under Parallel OCR. The top leaves two threads free, and the next page uses that number. **ZIM documents** is off until it is turned on. Off, a ZIM indexes its pages, and uploaded PDFs and pictures are still read. **Stay on** is the label. Its tip keeps the sentence "Stay on after the last file, and index new files as they arrive."

### Does the vertical bar cut the buttons off?

No. Dragging the vertical bar left wraps the Index controls, the Repair controls, and the log row so each control stays visible. The bar stops at that squeeze limit. The window still asks for at most 480 by 280. If the window is already that small, the host side shrinks.

### Does Follow fall behind?

With **Follow** on, new lines leave the newest line in view after the scroll settles, including when several updates arrive together. Scrolling away holds your place until you return to the end or turn **Follow** on again. **Follow** off does not jump.

### How can I see which file is indexing?

The row above the folder tree names it. That file's row in the folder tree also carries a tree mark the other file rows do not. The tree mark goes away when that file is no longer indexing.

### Is there a license?

Yes. This project is under the MIT License. The terms are in the `LICENSE` file. Copyright 2026 Vibe-Dragon.

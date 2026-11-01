# Changelog

## 1.1.5

### Moderate

- **Then index** is the other half of **Measure**. It does nothing until a measurement is already going. It does not start indexing and it does not start a measurement. During a measurement it arms that same pass to keep indexing after the samples, including a pass that started while idle, without restarting the pass or clearing the samples already taken. The half shows that it took effect. **Measure** by itself still stops when it started from idle. During a run, **Measure** still samples and then keeps indexing.
- **Follow** keeps the newest log line in view when several lines arrive before the page has scrolled. Being behind only because new lines arrived is still following. Scrolling away holds that place until you return to the end or turn **Follow** on again. **Follow** off does not jump.
- The vertical bar that sets the host pane width reflows the Index controls, the Repair controls, and the log row so each control stays fully visible. The bar stops when the main area is as narrow as those controls can go, wrapped onto extra lines, and the list gives up that height. The file headings scroll with the folder list, so those columns do not push the buttons off. The window still asks for at most 480 by 280. If the window is already that small while the host pane is open, the host side shrinks.

### Minor

- The on/off and either/or controls are buttons rather than checkboxes. That covers **Host system graphs**, **Hide finished**, **Stay on**, **Auto**, **Errors**, **Info**, **Debug**, **Follow**, the export session choice, and Repair's **Quick repair** and **Complete repair**. The stay-on button is labeled **Stay on**. Its tip keeps the full sentence. Index groups **Start indexing**, **Stop**, **Rescan**, and **Stay on**, then the **Measure** / **Then index** control, then **Auto**, the chunk field, **Recommended**, and the server count. **Delete index** stays apart at the end.
- The folder-tree row of the file that is indexing carries a tree mark the other file rows do not. Status text is not that mark. The pinned row above the tree still names the file. The tree mark goes away when that file is no longer indexing. A closed folder does not gain a badge.
- Repair uses the same buttons. **Quick repair** and **Complete repair** are one choice, and Quick is still the default. **Scan**, **Start repair**, and **Stop** are grouped. **Optimize stuck segments** stays on its own row.

## 1.1.4

### Moderate

- The embed batch is a chunk count. Typing it changes the next embed call. **Recommended** sets the count to the safe call detected from the memory and CPU of the computer running Turbo. A saved percent opens as the chunk count that percent meant. **Auto** follows finished embed calls. It can go above the old top while a larger count is still faster. It steps down when a larger count is clearly slower, when the embed server fails the call, or when free card memory is low. A brief slow patch that later calls do not repeat does not move the count. Article length does not move the count, and a pace from another article length is not used. Another publication's pace is not a verdict.
- One embed server can open a second sequence when the card has room for another 2048-token context. Each sequence keeps that context. Long calls that get slower send it back to one sequence. The log says `embed sequences`.
- **Debug** starts off in the log pane. The session log file still records those lines.

### Minor

- The main window opens at the size and place it had when it last closed, and maximized if it was maximized. The log bar is placed again after the window has that size.
- The file list and the controls under it are one section. The bar above the log sizes that section by changing the list height. The controls stay against the top of the bar, and the log row stays against the bottom of it. **Host system graphs** is a checkbox. It opens a side pane, and a vertical bar sets that pane's width. Whether that pane is open, that width, and the log bar are remembered.
- The file list shows File type, then Size, and splits the pace into art/min and chk/s. Order chooses how unfinished files are indexed. The list is a folder tree for kb_uploads and zim. Folders start closed. A closed folder is not loaded into the list. Opening one loads its rows, and closing it drops them. The refresh updates the file being indexed and the library bar. An open folder stays open. The file being indexed stays in its own row above that tree. The library bar under the list is the full width of the list. Dragging a column edge moves that row and the tree together, and extra width stays on the last column. Hide finished takes indexed files out of the tree. Right-click a file for Index now or priority. Measure samples each unfinished file for about five minutes and fills in the ETAs. The window does not walk the whole library on every article, and saving file facts does not stop the window. While a file is indexing, the library folders are not walked again until that file finishes, and a progress save writes that file's facts without rewriting every other file. The chunk-count line and the card-memory line are gone. The number of embed servers running sits beside **Recommended**. **Export log** is on the log row, with **Log help** to its right. **Measure** is on the button row. **Then index**, beside **Measure**, samples the same way and then keeps indexing. **Measure** by itself still stops when it started while idle. Stopping a measure leaves **Measure** clickable once the run is idle. **Order** and **Hide finished** sit in the header beside **Host system graphs**, and the order is remembered. Each disk in that pane has a read graph and a write graph. A file that continues names the stored point count when its article paths are not listed again.
- The Qdrant status line no longer reads the container log. That read does not return, and a copy left behind keeps Docker busy after the window is closed.
- A PDF or picture stays on the file line while OCR reads it. Its row says OCR, and that row's progress is the page. The bar shows the page, and each page is written to the log as it finishes. That read does not use up the five-minute Measure sample. A file left before any of it was stored stays waiting.
- Right-click a file for Details. Size, location, dates, article and chunk counts, slowest and fastest speeds, the last measured ETA, and priority are saved with that file. Opening Turbo again shows them. A file that leaves the library is dropped from that record.
- The ZIM reader loads upcoming clusters while it reads the current page. It uses half of the cache libzim already chose and does not raise that cache. If the prefetch stops, the file continues.

## 1.1.3

### Moderate

- **Auto** remembers the pace of the size just below the current one, from finished embed calls. A long stay at one size no longer forgets that neighbor. It steps down when the current calls are clearly slower than that neighbor. A fast sample from earlier in the climb does not pull the slider down. A step that is not faster does not keep going. A small slowdown is still left alone. 150% remains only the top of the slider.

## 1.1.2

### Moderate

- **Auto** follows finished embed calls in both directions. It steps down when a larger call is clearly slower or the embed server struggles. A slowdown smaller than about a fifth is left alone. It tries one step higher again after about a minute and a half and more finished calls, and sooner when the pace of the work changes a lot. A probe that keeps up waits again before the next step. One odd slow call does not cut the batch. Card memory still stops a climb when little is free. 150% remains only the top of the slider.
- Routine optimization lines in the log are one sentence. An optimizer error keeps the full detail. **Errors**, **Info**, and **Debug** still only hide lines in the pane. Each window start keeps the previous session log and removes older ones. **Export log** can save the current session or that previous one. **Log help**, beside About, explains the lines Turbo writes on purpose.

## 1.1.1

### Moderate

- The line under the Index buttons recommends a batch size from finished embed calls. **Auto**, beside the slider, raises the call while those calls keep up and steps back when a larger call is slower or the embed server struggles. The stop is the fastest size measured. 150% remains only the top of the slider. The advice does not guess from the card.

## 1.1.0

### Major

- Indexing pauses before a new upsert when a plain segment is past the 4 GiB vector cap, or when free space is under twice the segment that would be copied. Points already stored stay. The pause is what stops that segment from growing past the size a later rewrite can copy.
- The embed-batch slider goes to 150%. 10% to 100% stay inside the budget detected on the computer running Turbo. Above 100% sends more chunks than the safe call. It does not change GPU clocks. 150% adds half of the safe call, rounded down to a whole number of chunks. A new install still starts at 55%. Moving the slider during a run changes the next embed call. The call already in flight is left alone. One embed server stays up.

### Moderate

- Repair has **Optimize stuck segments**. It asks Qdrant to rewrite an oversized or fully-deleted segment only when no optimization is already running, free space is at least twice that segment, and the in-memory graph fits in spare memory. Otherwise the click is refused and the collection is left unchanged. The reason is on the Repair status line and in the log.
- The Index page shows an optimization bar while indexing is paused for that rewrite. Each new pause starts the clock over. The Repair page shows whether an optimization is running, waiting for an update, refused, or in error. Qdrant does not report a percent for the in-memory build, so the bar moves without one.
- While indexing is running, a Qdrant problem is written to the log once. That covers an optimizer error, a red collection, a collection that stops answering, a failed status check, a failed point count, and a failed read of stored paths. When the collection is answering again, the log says so.

### Minor

- **Embed batch** sits on the button row, between **Export log** and **Delete index**, so the file bar and the library bar have the row above.
- Every Optimize click is written to the log: checking, refused, started, failed, or not started.
- The next major milestone is **Speed up segment optimization**. Segment builds are the bottleneck after the write hold. That work is not in this version.

## 1.0.3

### Minor

- Host opens a separate window for CPU, memory, GPU, and video memory. Each disk the library uses shows its size and its read and write speed, including a folder that lives on another disk. It stays off the Index page. The window remembers its size and place.
- The 110% end of the embed-batch slider is a label past the track, so it stays clear of the 100% mark.

## 1.0.2

### Moderate

- Index and Repair put the single-line controls on shared rows, so the file list gets at least half the page at the default window size when the log is open and the splits have not been dragged. Extra height still goes to the list. At 480 by 280 the list stays on screen and the control area stays mapped, so those controls can be scrolled into view.
- The header shows the package version. An About page, beside Index and Repair, names that version, Project NOMAD 1.35.1, and links to this repository and the Project NOMAD repository.

### Minor

- The corner link opens this repository. The README links to Project NOMAD.
- The README describes the CPU, GPU, memory, and disk spikes a run can show, and which of them mean the index is still working.

## 1.0.1

### Major

- A stored point now includes `active`. Stock NOMAD writes that flag, and opening the Knowledge Base page sets `active: true` on every point that does not have it. That rewrite holds the collection until it finishes, so Turbo stays on the current file and the article count does not move. Qdrant keeps writing the open segment, and the in-memory search graph grows with the index. New points carry the flag, so that page no longer has a missing field to rewrite. The flag is false only when that file's ingest row is turned off.

### Moderate

- While the collection does not answer, the Qdrant status says not to restart Qdrant or delete the index. It says a restart would replay the unfinished write. It also says whether the open segment is still gaining vectors, read from the segment file so the check does not take the collection lock.
- The indexing status line says Turbo is waiting on the current file, and not to restart Qdrant or delete the index.

### Minor

- The main window resizes from its edges again. List, control, and log minimums no longer add up to the whole screen.
- Shrinking the window keeps a file list on screen. Extra space goes to the list. The log and the controls keep the size set by their drag bar.
- Index and Repair each have a drag bar between the file list and the controls under it.
- The log drag bar sits under the Log row and sizes the log inside the window.
- The drag bars are a thin strip. The Qdrant status sits under the page switch, away from the log bar.
- Column edges still resize. Those positions are remembered.
- The main window links to the Project NOMAD repository.
- The README describes the Repair page, and how a ZIM resumes from the saved article instead of counting from the first article on every batch.
- The README names Project NOMAD 1.35.1 as the tested version, and describes the Index page, the log, the commands, and the Qdrant and embed-server errors.

## 1.0.0

### Major

- First published package of Nomad TURBO Index.
- One embed server. The slider sets the next call from 10% to 110% of the budget detected on the computer running Turbo.

### Moderate

- Index and Repair pages, and a tray icon while a run is active.

### Minor

- Delete index asks for the word DELETE before it removes the collection.

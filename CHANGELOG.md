# Changelog

## Unreleased

## 1.1.0

### Major

- Indexing pauses before a new upsert when a plain segment is past the 4 GiB vector cap, or when free space is under twice the segment that would be copied. Points already stored stay. The pause is what stops that segment from growing past the size a later rewrite can copy.
- The embed-batch slider goes to 150%. 10% to 100% stay inside the budget detected on the computer running Turbo. Above 100% is an overclock. 150% adds half of the safe call, rounded down to a whole number of chunks. A new install still starts at 55%. Moving the slider during a run changes the next embed call. The call already in flight is left alone. One embed server stays up.

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

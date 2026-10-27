# Changelog

## Unreleased

## 1.0.1

- The main window resizes from its edges again. List, control, and log minimums no longer add up to the whole screen.
- Shrinking the window keeps a file list on screen. Extra space goes to the list. The log and the controls keep the size set by their drag bar.
- Index and Repair each have a drag bar between the file list and the controls under it.
- The log drag bar sits under the Log row and sizes the log inside the window.
- The drag bars are a thin strip. The Qdrant status sits under the page switch, away from the log bar.
- Column edges still resize. Those positions are remembered.
- The main window links to the Project NOMAD repository.
- The README describes the Repair page, and how a ZIM resumes from the saved article instead of counting from the first article on every batch.
- The README names Project NOMAD 1.35.1 as the tested version, and describes the Index page, the log, the commands, and the Qdrant and embed-server errors.
- A stored point now includes `active`. Stock NOMAD writes that flag, and opening the Knowledge Base page sets `active: true` on every point that does not have it. On this machine that rewrite held the collection for about 25 minutes while Turbo stayed on one Wikispecies article. Qdrant kept writing the open segment, and its in-memory search graph climbed from about 14 GB to about 22 GB. New points carry the flag, so that page no longer has a missing field to rewrite. The flag is false only when that file's ingest row is turned off.
- While the collection does not answer, the Qdrant status says not to restart Qdrant or delete the index. It says a restart would replay the unfinished write. It also says whether the open segment is still gaining vectors, read from the segment file so the check does not take the collection lock.
- The indexing status line says Turbo is waiting on the current file, and not to restart Qdrant or delete the index.

## 1.0.0

- First published package of Nomad TURBO Index.
- One embed server. The slider sets the next call from 10% to 110% of the budget detected on that computer.
- Index and Repair pages, a tray icon while a run is active, and Delete index after typing DELETE.

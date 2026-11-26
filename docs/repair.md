# Repair


Repair is the second page. It does not embed files, and it does not wipe the collection. It finds index entries whose library file is gone, and **Start repair** can delete those entries. **Delete index** is the other button, on the Index page, and it asks you to type DELETE.

**Quick repair** and **Complete repair** are buttons, and exactly one is selected. Quick is the default. Both **Scan** and **Start repair** use the mode that is selected. **Scan**, **Start repair**, and **Stop** are grouped beside that choice. **Optimize stuck segments** stays on its own row. The file list takes the spare height on this page, the same way the Index list does. The controls wrap when the page is narrow, the same way the Index buttons do. The longer repair instructions are on the status line's tip.

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

**Optimize stuck segments** is on its own row under those buttons, with a bar beside it. It does not need a file in the repair list. It asks Qdrant to rewrite an oversized plain segment, or a fully-deleted segment still on disk, only when that copy is safe: no optimization is already open, free space is at least twice the segment, and the in-memory graph fits in spare memory. A refusal leaves the collection unchanged and says why. The bar can read **Already optimizing**, **Waiting for an update**, **Not started**, or **Optimizer error**. It pulses when a build is open. Qdrant does not report a percent. Do not restart Qdrant to clear that bar.

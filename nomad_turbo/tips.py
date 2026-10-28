"""Plain-language tooltips for the Index and Repair pages.

The window and the tests share this table, so the wording can be checked
without opening a window. Each sentence describes what that control does.
"""

from __future__ import annotations

TIPS: dict[str, str] = {
    "qdrant": (
        "Whether Qdrant is running and the collection is answering. "
        "While a write is still moving, this says so and says not to restart Qdrant or delete the index. "
        "Ready for changes means the collection answered."
    ),
    "qdrant_gate": (
        "Qdrant is not ready for changes yet. This stays off until the collection answers."
    ),
    "col_file": "Blue names are ZIM files. Amber names are uploads. Hover a row for its full path.",
    "col_kind": "ZIM library or a file uploaded to the knowledge base.",
    "col_status": "Waiting, indexing, indexed, partial, failed, or needs NOMAD.",
    "col_progress": "How far this file is through its articles.",
    "col_articles": "Articles stored so far, and the total when it is known.",
    "col_chunks": "Embedded pieces stored for this file. A long page is many chunks.",
    "col_size": "Size of the file on disk.",
    "col_rate": (
        "This file's recent pace. Articles per minute swing with page length. "
        "Chunks per second stay steadier."
    ),
    "col_eta": "Time left for this file at its recent article rate.",
    "file_now": "The file being indexed right now.",
    "file_bar": (
        "Progress of the file being indexed. The library bar beside it is the whole library, weighed by file size."
    ),
    "library_bar": "Share of the library finished, by file size, and the ETA for what is left.",
    "index_optimize": (
        "Shown while indexing is paused so a segment can be optimized. "
        "The clock starts over each time a new pause begins. "
        "Qdrant does not report how far the segment build is, so the bar moves without a percent."
    ),
    "stats": (
        "Articles per minute swing with page length. Chunks per minute is the steadier pace. "
        "Skipped pages had no text. Bad pages could not be read or parsed. "
        "A long page is indexed in full and stored as many chunks."
    ),
    "slider": (
        "How many chunks the next embed call sends, as a percent of the budget detected on the computer running Turbo. "
        "10% to 100% stays inside that budget. 150% is an overclock past that budget. "
        "Moving it during a run changes the next embed call. "
        "One embed server stays up. A call that is already running is left alone."
    ),
    "usage": (
        "The chunk count this setting will send on the next embed call, "
        "and whether the one embed server is running."
    ),
    "card": "Video memory in use on the card. Turbo's embed server is part of this number.",
    "keep": (
        "Off: when the queue is empty, Turbo stops and the ingest policy returns "
        "to the value found at Start. When the run started on Manual, stock NOMAD "
        "does not start embedding. On: Turbo keeps control and indexes files that show up later."
    ),
    "start": "Pause NOMAD's file embedding and index from the saved place. This does not delete anything.",
    "stop": (
        "Stop after the embed call that is already running. This does not delete the index. "
        "When the run started on Manual, unfinished files are not handed back to stock NOMAD. "
        "The ingest policy returns to the value found at Start, except a half-finished file "
        "that started on Always stays Manual."
    ),
    "rescan": "Read the library again and refresh this list. This does not pause NOMAD or start indexing.",
    "export": "Save the whole session log, including lines the checkboxes are hiding.",
    "delete": (
        "Remove stored knowledge points and ingest rows. You must type DELETE. "
        "Indexing does not start. ZIM files and uploads stay on disk."
    ),
    "errors": (
        "Warnings and errors. Unchecking this only hides them in the pane. The log file still records them."
    ),
    "info": "Ordinary progress lines. Unchecking this only hides them in the pane.",
    "debug": "Detail lines, including one line per article. Unchecking this only hides them in the pane.",
    "follow": (
        "On jumps to the newest line and stays there. Scroll up to hold your place. "
        "Off keeps the page where it is while new lines arrive."
    ),
    "log": (
        "Show or hide the log. The bar under this row sizes the log inside the window. "
        "Each page has its own bar between the file list and the controls. "
        "The window edges resize the whole window. Those positions are remembered."
    ),
    "project": (
        "Open this project's repository."
    ),
    "nomad_repo": "Open the Project NOMAD repository.",
    "version": "This copy of Nomad TURBO Index. About names the Project NOMAD version it was tested with.",
    "host": (
        "Open a separate window for CPU, memory, GPU, video memory, and the disks the library uses. "
        "Resize that window and place it where you want. The size and place are remembered."
    ),
    "host_disk": (
        "Used space and size of each disk the library uses, and how fast that disk is reading and writing. "
        "A folder on its own disk, such as Qdrant, is a separate line."
    ),
    "legend": (
        "Blue names are ZIM files. Amber names are files you uploaded. "
        "Drag a column edge to resize it. Drag a bar to resize the list, the controls, or the log. "
        "Click a heading to sort."
    ),
    "repair_page": (
        "Scan checks the library and the index and changes nothing. "
        "Start repair removes index entries for library files that are gone. "
        "NOMAD keeps embedding until you press Start repair. "
        "Quick compares files, ingest rows, and checkpoints. "
        "Complete also lists sources in the index when Qdrant is answering. "
        "Start repair pauses file embedding, then deletes points for files that are gone. "
        "Points for a file that is still on disk stay there. "
        "The tray icon turns blue while a repair is running. Stop is here and in the tray. "
        "Optimize stuck segments rewrites an oversized or fully-deleted segment when that is safe."
    ),
    "repair_quick": (
        "Compare files on disk with ingest rows and checkpoints. "
        "Delete index points for ZIM and upload files that are gone."
    ),
    "repair_complete": (
        "Also list every source in the index. Delete orphan points for missing library files. "
        "When a file is still on disk, show a count that disagrees and leave those points in place."
    ),
    "repair_col_file": "The library file this row is repairing. Hover a row for the full path and the live spec.",
    "repair_col_problem": "Why this file is in the list.",
    "repair_col_points": (
        "Points in the index when a complete repair listed them. "
        "Otherwise the chunk count from the ingest row."
    ),
    "repair_col_progress": "The bar moves while this file is being deleted. It fills when the file is finished.",
    "repair_col_status": "Waiting, deleting, repaired, noted, failed, stopped, would delete, or report only.",
    "repair_col_elapsed": "Time spent on this file. It updates while the delete is running.",
    "repair_file_now": "The file repair is working on right now.",
    "repair_file_bar": "Progress of the file being repaired. The bar beside it is the whole repair or scan.",
    "repair_overall_bar": "During a scan, how far the check has got. During a repair, how many files are finished.",
    "repair_stats": "Live counts for the library, ingest rows, checkpoints, index sources, and what a repair would do.",
    "repair_scan": (
        "Check the library and the index without pausing NOMAD or deleting anything. "
        "Stop is checked between steps. Listing sources runs until Qdrant answers."
    ),
    "repair_start": (
        "Pause NOMAD file embedding and repair the index. The tray icon turns blue. "
        "The desktop icon stays the same."
    ),
    "repair_stop": (
        "Stop between files, or between scan steps. A delete that was already sent runs until the index finishes it. "
        "Listing sources runs until Qdrant answers. This does not delete the whole index."
    ),
    "repair_optimize": (
        "Rewrite stuck index segments when that is safe. "
        "An oversized plain segment, or fully-deleted segments still on disk, can be optimized. "
        "This stays available when the file list is empty. "
        "It runs only when free space is at least twice the segment, no optimization is already running, "
        "and the in-memory graph fits in spare memory. Otherwise the index is left unchanged and the reason is shown. "
        "File embedding pauses only while this runs, then resumes. Points and the collection stay. "
        "The result is written to the log."
    ),
    "repair_optimize_bar": (
        "Whether Qdrant is optimizing. Already optimizing means a build is open and this button will not start another. "
        "Waiting for an update means Qdrant will not run until Optimize stuck segments sends one. "
        "Not started means the click was refused and the reason is on the status line and in the log. "
        "Qdrant does not report a percent for a segment build."
    ),
}

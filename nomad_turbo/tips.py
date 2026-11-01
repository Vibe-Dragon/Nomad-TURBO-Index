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
    "col_file": (
        "Blue names are ZIM files. Amber names are uploads. Hover a row for its full path. "
        "A red asterisk to the left of the name means that file is prioritized."
    ),
    "col_type": "The file type: zim, pdf, docx, epub, txt, md, rtf, or an image extension such as jpg or png.",
    "col_status": "Waiting, indexing, OCR, indexed, partial, failed, or needs NOMAD.",
    "col_progress": "How far this file is through its articles.",
    "col_articles": "Articles stored so far, and the total when it is known.",
    "col_art_min": "Articles per minute for the file being indexed. Other rows stay blank.",
    "col_chunks": "Embedded pieces stored for this file. A long page is many chunks.",
    "col_chk_s": "Chunks per second for the file being indexed. This pace stays steadier than articles per minute.",
    "col_size": "Size of the file on disk.",
    "col_eta": (
        "Time left for this file. The file being indexed uses its recent article rate. "
        "After Measure, a file with a known article count keeps the ETA from its sample."
    ),
    "order": (
        "The order unfinished files are indexed in, and the order of this list. "
        "The file already in progress stays at the top and keeps going. "
        "Prioritized files come next, in this same order. Finished files stay at the bottom."
    ),
    "hide_finished": "Hide files that are already indexed. While this is off, those files stay at the bottom of the list.",
    "measure": (
        "Index each unfinished file for about five minutes, priority files first, "
        "then go on to the next. Reading a PDF or picture with OCR happens first and "
        "does not count as those five minutes. "
        "Each sample starts at the recommended batch when Auto is on, "
        "or stays on the batch you set when Auto is off. "
        "The ETAs update from those samples. "
        "If a run is already going, indexing resumes afterward. Stop ends the pass."
    ),
    "index_now": (
        "Finish the embed call already running, save that file, and index this one next. "
        "Stored entries are kept."
    ),
    "priority": (
        "Index this file before files that are not marked. "
        "Marked files follow the selected order. "
        "A red asterisk sits to the left of the name. "
        "The mark is remembered after Turbo is closed."
    ),
    "details": (
        "Saved facts for this file: size, location, dates, article and chunk counts, "
        "slowest and fastest speeds, the last measured ETA, and whether it is prioritized. "
        "These stay after Turbo is closed. A file that leaves the library is forgotten."
    ),
    "file_now": (
        "The file being indexed right now. "
        "While a PDF or picture is being read, this line names that file."
    ),
    "file_bar": (
        "Progress of the file being indexed. While OCR is reading a PDF or picture, "
        "the bar shows the page it is on. "
        "The library bar beside it is the whole library, weighed by file size."
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
        "How many chunks the next embed call sends. "
        "Type a count and the next call uses it. "
        "Recommended sets the count to the safe call detected from the memory and CPU of the computer running Turbo. "
        "One embed server stays up. A call that is already running is left alone. "
        "The line under the buttons says whether finished calls are keeping up."
    ),
    "recommend": (
        "Set the chunk count to the safe call for the computer running Turbo. "
        "That count comes from detected memory and CPU. It is a start, not a maximum."
    ),
    "tune": (
        "What the last full embed calls say about this batch size. "
        "It follows chunks per second on those calls, and it keeps a pace for each kind of content. "
        "One page length is not its own pace. It does not guess from the card. "
        "A segment rewrite is not something a larger batch will speed up."
    ),
    "tune_auto": (
        "Move the chunk count while a larger count is still faster. "
        "Step down when a larger count is clearly slower, the embed server fails the call, "
        "or free card memory is low. A brief slow patch that later calls do not repeat is left alone. "
        "Article length does not move the count, and a pace from another article length is not used. "
        "Another publication's pace is not a verdict. Auto can go above the old top. "
        "One embed server stays up. A call already running is left alone."
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
    "export": (
        "Save the current session log, or the previous session if one is saved. "
        "The file includes lines the checkboxes are hiding. Older sessions are not kept."
    ),
    "log_help": (
        "Open a window that explains the log lines Turbo writes on purpose. "
        "An unforeseen error still prints its own message."
    ),
    "delete": (
        "Remove stored knowledge points and ingest rows. You must type DELETE. "
        "Indexing does not start. ZIM files and uploads stay on disk."
    ),
    "errors": (
        "Warnings and errors. Unchecking this only hides them in the pane. The log file still records them."
    ),
    "info": "Ordinary progress lines. Unchecking this only hides them in the pane.",
    "debug": (
        "Detail lines, including one line per article. "
        "The pane starts with these hidden. Checking Debug shows them here. "
        "The log file still records them either way."
    ),
    "follow": (
        "On jumps to the newest line and stays there. Scroll up to hold your place. "
        "Off keeps the page where it is while new lines arrive."
    ),
    "log": (
        "Show or hide the log text. The bar above this row sizes the list and the log. "
        "The controls stay against the top of that bar, and this row stays against the bottom. "
        "The window edges resize the whole window. The size, the place, and that bar are remembered."
    ),
    "project": (
        "Open this project's repository."
    ),
    "nomad_repo": "Open the Project NOMAD repository.",
    "version": "This copy of Nomad TURBO Index. About names the Project NOMAD version it was tested with.",
    "host": (
        "Show or hide the host meters beside the pages. The vertical bar sets the width of that pane. "
        "Whether it is open, and that width, are remembered."
    ),
    "host_disk": (
        "Used space and size of each disk the library uses, and how fast that disk is reading and writing. "
        "A folder on its own disk, such as Qdrant, is a separate line."
    ),
    "legend": (
        "Blue names are ZIM files. Amber names are files you uploaded. "
        "Drag a column edge to resize it. The bar above the log changes the list height. "
        "The controls stay under the list."
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

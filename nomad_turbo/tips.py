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
        "Blue names are ZIM files. Amber names are uploads. "
        "kb_uploads and zim are folders: click the arrow to open or close one. "
        "A closed folder does not list its files until it is opened. "
        "The file being indexed stays in the row above the list. "
        "When a second file is being read at the same time, it stays in that row too. "
        "Finished reads wait in memory, and on disk after that text passes about a quarter of system memory. "
        "Hover a row for its full path. "
        "A red asterisk to the left of the name means that file is prioritized. "
        "A green triangle on a folder-tree row means that file is the one indexing."
    ),
    "col_type": (
        "The file type: zim, pdf, docx, epub, txt, md, rtf, or an image extension such as jpg or png. "
        "A folder row says folder."
    ),
    "col_status": "Waiting, HTML, text, OCR, OCR text, indexing, indexed, partial, failed, or needs NOMAD.",
    "col_progress": (
        "How far this file is through its articles. "
        "A ZIM fills this once for HTML, again for other text, and again for OCR. "
        "Green is HTML, blue is other text, orange is recognizing a page, and purple is storing that text. "
        "A file that is one article moves by the chunks of that article while they are embedding."
    ),
    "settings_index": (
        "Where indexing runs. CPU, or an NVIDIA card. "
        "The card is used the next time you start. "
        "AMD and Intel cards are not used for indexing."
    ),
    "settings_ocr": (
        "Where OCR runs. CPU, or a card. "
        "The program picks the reader for that card. "
        "GPU stays off until the GPU reader is installed from this page. "
        "The choice is saved and applies to the next file. "
        "A different device can read pages while indexing is running."
    ),
    "settings_install": (
        "Downloads PyTorch and EasyOCR into this user's home folder. "
        "It is a few gigabytes and needs a network connection. "
        "GPU OCR turns on when the download finishes and a small check on the card succeeds. "
        "The English models download the first time a page is read."
    ),
    "settings_engine": (
        "Which reader is in use, and whether OCR waits for indexing or runs beside it. "
        "CPU OCR gives cores back while Qdrant is writing or optimizing."
    ),
    "parallel_ocr": (
        "Experimental. Off, one file is read or indexed at a time. "
        "OCR waits until that file is stored, then the next file starts. "
        "On, CPU OCR reads the next PDF or picture while the GPU indexes. "
        "That overlap is still being worked out. "
        "The choice is saved and applies the next time you start."
    ),
    "ocr_quality": (
        "How hard a PDF page or a picture is read. "
        "Low is 200 dpi and skips a page that already has text. "
        "Medium is 300 dpi and reads every page. "
        "High is 400 dpi for small print. "
        "The choice is saved and applies to the next file."
    ),
    "col_articles": (
        "Articles stored so far, and the total when it is known. "
        "During a ZIM's text or OCR pass, this is how far that pass is. "
        "While OCR is reading a file, this is pages read and pages in the file."
    ),
    "col_art_min": (
        "Articles per minute for the file being indexed. "
        "While OCR is reading, this is pages per minute. "
        "Other rows stay blank."
    ),
    "col_chunks": (
        "Embedded pieces stored for this file. A long page is many chunks. "
        "While that article is still embedding, this counts chunks embedded so far."
    ),
    "col_chk_s": "Chunks per second for the file being indexed. This pace stays steadier than articles per minute.",
    "col_size": "Size of the file on disk.",
    "col_eta": (
        "Time left for this file. The file being indexed uses its recent article rate. "
        "A file that is one article uses its chunk pace while that article is embedding. "
        "While OCR is reading, this is the time left at the recent page rate. "
        "After Measure, a file with a known article count keeps the ETA from its sample."
    ),
    "order": (
        "The order unfinished files are indexed in. "
        "The file already in progress keeps going. "
        "Prioritized files come next, in this same order. "
        "The list is the kb_uploads and zim folders, not this order. "
        "The choice is remembered after Turbo is closed. "
        "Changing this clears a queue you arranged, so this menu applies again."
    ),
    "queue": (
        "Opens the embed queue and the OCR queue. "
        "The left side is files waiting to be embedded. "
        "The right side is files waiting for OCR. "
        "Drag a waiting file to change what comes next. "
        "The file already being read or indexed stays put and keeps its progress. "
        "Index now still comes first. "
        "A saved queue is used before priority and before the Order menu. "
        "Changing Order clears that arrangement. "
        "Files that show up later follow that queue."
    ),
    "hide_finished": (
        "Hide files that are already indexed. "
        "A folder that only holds those files is hidden too."
    ),
    "measure": (
        "Index each file that has not been started for about five minutes, "
        "priority files first, then go on to the next. "
        "A file that already has progress keeps its ETA and is left alone. "
        "Reading a PDF or picture with OCR happens first and "
        "does not count as those five minutes. "
        "Each sample starts at the recommended batch when Auto is on, "
        "or stays on the batch you set when Auto is off. "
        "The ETAs update from those samples. "
        "If a run is already going, indexing resumes afterward. "
        "Started while idle, this stops when the samples are done. "
        "Then index, while a measurement is already going, keeps indexing after those samples. "
        "Stop ends the pass."
    ),
    "then_index": (
        "While a measurement is already going, keep indexing after the samples. "
        "This does nothing until Measure has started. "
        "It does not start a measurement and it does not restart the one that is running. "
        "The sample clock and the files already sampled stay as they are. "
        "Measure by itself still stops when it was started while nothing was running. "
        "Stop ends the pass."
    ),
    "index_now": (
        "Finish the embed call already running, save that file, and index this one next. "
        "Stored entries are kept."
    ),
    "priority": (
        "Index this file before files that are not marked. "
        "Marked files follow the selected order. "
        "A red asterisk sits to the left of the name. "
        "The mark is remembered after Turbo is closed. "
        "It is removed when that file finishes embedding, and when the index is deleted."
    ),
    "details": (
        "Saved facts for this file: size, location, dates, article and chunk counts, "
        "slowest and fastest speeds, the last measured ETA, and whether it is prioritized. "
        "These stay after Turbo is closed. A file that leaves the library is forgotten."
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
        "How many embed servers are running. Turbo keeps one. "
        "The line under that is how many OCR workers are reading. "
        "The chunk count for the next call is in the field beside Recommended."
    ),
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
        "Save the log files for this knowledge index as one zip. "
        "Each day is its own file. The zip includes lines the log buttons are hiding."
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
        "Warnings and errors. Turning this off only hides them in the pane. The log file still records them."
    ),
    "info": "Ordinary progress lines. Turning this off only hides them in the pane.",
    "debug": (
        "Detail lines, including one line per article. "
        "The pane starts with these hidden. Turning Debug on shows them here. "
        "The log file still records them either way."
    ),
    "follow": (
        "On jumps to the newest line and stays there, including when several lines arrive together. "
        "Scroll away to hold your place. That place stays until you return to the end or turn this on again. "
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
        "Show or hide the host system graphs beside the pages. "
        "The vertical bar sets the width of that pane. "
        "Whether it is open, and that width, are remembered."
    ),
    "host_disk": (
        "Read and write for each disk indexing uses, and for the disk that holds the knowledge base. "
        "The blue line is read. The red line is write. "
        "A folder on its own disk is its own graph."
    ),
    "legend": (
        "Blue names are ZIM files. Amber names are files you uploaded. "
        "kb_uploads and zim open and close. The file being indexed stays above that tree. "
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

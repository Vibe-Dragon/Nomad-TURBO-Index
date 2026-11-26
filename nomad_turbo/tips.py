"""Plain-language tooltips for the Index, Repair, Qdrant, and Models pages.

The window and the tests share this table, so the wording can be checked
without opening a window. Each sentence describes what that control does.
"""

from __future__ import annotations

TIPS: dict[str, str] = {
    "qdrant": (
        "Whether Qdrant is running and the collection is answering. "
        "While a write is still moving, this says so and says not to restart Qdrant or delete the index. "
        "Ready for changes means the collection answered. "
        "Green means ready. Blue is a pause that is still working. "
        "Amber means leave Qdrant running. Red needs attention."
    ),
    "activity": (
        "What Turbo is doing when the file list is still. "
        "Green is indexing or idle. "
        "Blue is a pause that is still working, such as saving records after Stop. "
        "Amber means leave Qdrant running. Red needs attention."
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
    "col_status": (
        "Waiting, Waiting OCR, finding, documents, HTML, text, OCR, indexing, indexed, partial, failed, or needs NOMAD. "
        "Waiting OCR means the ZIM's pages are stored and PDFs from it are still in the OCR queue. "
        "Under Internal Zim Documents, finding means that ZIM's list is still growing, and documents means its pages are stored and the bar counts documents stored."
    ),
    "col_progress": (
        "While a file is being embedded, the faded bar is chunks that already have vectors "
        "and the solid bar is what is saved in the knowledge base. "
        "The label is the saved percent, then the processed percent. "
        "A ZIM fills the solid bar once for HTML, again for other text, and again for OCR. "
        "Green is HTML, blue is other text, including text stored after a page is recognized, and orange is recognizing a page. "
        "While OCR is reading, this is one bar for the pages read. "
        "The bar under Internal Zim Documents is how many of that ZIM's documents are stored."
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
    "ocr_worker_max": (
        "How many OCR workers may run at once. "
        "The slider stops with two threads left free. "
        "The number is saved and the next page uses it, including a read that is already going. "
        "Qdrant can still take workers back while it is writing."
    ),
    "zim_documents": (
        "Off, a ZIM indexes its pages. "
        "PDFs, Word documents, and EPUBs inside it stay unread. "
        "Uploaded PDFs and pictures are still read. "
        "On, those files are listed while the pages are indexed, then read after every ZIM's pages and every upload are stored. "
        "They are listed under Internal Zim Documents. That ZIM's bar there is how many of its documents are stored. "
        "A PDF is read with OCR. A page that already has text is stored from that text unless OCR every PDF is on. "
        "The choice is saved and applies the next time you start. "
        "A ZIM that is already indexed stays as it is."
    ),
    "ocr_every_pdf": (
        "Off, a PDF page that already has text is not drawn. "
        "A page with no text is still read, and a picture is still read. "
        "On, every PDF page is drawn, including a page that already has text. "
        "The text layer is kept either way. "
        "The choice is saved. The next PDF uses it. "
        "A ZIM already being read keeps going until the next time you start."
    ),
    "ocr_quality": (
        "How hard a PDF page or a picture is read. "
        "Low is 200 dpi. Medium is 300 dpi. High is 400 dpi for small print. "
        "A page that already has text is drawn when OCR every PDF is on. "
        "A PDF or picture that is not English is left out. "
        "The choice is saved and applies to the next file."
    ),
    "col_articles": (
        "While a file is being indexed, articles saved in the knowledge base, and how many are left. "
        "During a ZIM's text or OCR pass, the right number is how many that pass still has left. "
        "Other rows show articles stored and the total when it is known. "
        "While OCR is reading a file, this is pages read and pages in the file."
    ),
    "col_art_min": (
        "Articles per minute whose vectors have come back, for the file being indexed. "
        "This moves before those articles are saved. "
        "While OCR is reading, this is pages per minute. "
        "Other rows stay blank."
    ),
    "col_chunks": (
        "Embedded pieces stored for this file. A long page is many chunks. "
        "While one article is still embedding, this counts chunks embedded so far."
    ),
    "col_chk_s": (
        "Chunks per second that have received vectors, for the file being indexed. "
        "This moves while an embed call is still running."
    ),
    "col_size": "Size of the file on disk.",
    "col_eta": (
        "Time left for this file. The file being indexed uses its recent saved article rate. "
        "When nothing has been saved yet, it uses articles whose vectors have come back. "
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
        "Prioritized files come before text that has finished OCR. "
        "That finished text comes before the other files. "
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
        "Stop ends the embed that is already running and keeps the last article that was fully saved. This does not delete the index. "
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
    "qdrant_page": (
        "How the knowledge-base collection stores vectors. "
        "Distance stays cosine and each vector stays 768 numbers, which is what NOMAD search expects. "
        "NOMAD can search and embed either way. It sends ordinary floats, and Qdrant converts them on the way in."
    ),
    "qdrant_storage": (
        "Float32 stores each number in 4 bytes. Float16 stores it in 2 bytes, and search quality barely moves. "
        "A newer Qdrant also offers Turbo16, Turbo8, and Turbo4. Those keep a compressed copy as the vector itself "
        "and still accept an ordinary float on write. "
        "uint8 is not listed. That type expects numbers from 0 to 255, and this embedder does not send those. "
        "The storage type cannot be changed on a collection that already exists. "
        "Apply deletes it and creates another, after you type DELETE. Points already stored are removed. "
        "Library files stay. A later delete of the index creates the collection again with the type saved here."
    ),
    "qdrant_quant": (
        "The quantization add-on keeps the original vectors and writes a 1-byte copy beside them. "
        "The graph walk reads that smaller copy, including from disk, so that walk is faster, then scores the short result list "
        "again with the original numbers. NOMAD still sends the same search request, and Qdrant does the faster walk itself. "
        "Disk grows by about 768 bytes a point. Points stay. "
        "The copy is built when Qdrant indexes a segment. "
        "Qdrant rebuilds segments in the background, so indexing may pause. "
        "With the vectors stored on disk, the int8 copy stays on disk too and is not pinned into RAM. "
        "Search reads it from disk. In-RAM vector storage would pin that copy even when this says it is not."
    ),
    "qdrant_graph": (
        "The HNSW graph is in memory unless this is on. On disk uses less RAM and makes a search a little slower. "
        "Points stay. NOMAD does not care which place the graph lives. "
        "Apply writes this now, on ahead of time or off at any time. "
        "Qdrant rebuilds the graph in the background. "
        "The place is stored with the collection, so a newer Qdrant keeps it."
    ),
    "qdrant_vectors": (
        "Where the original vectors are stored. On disk keeps the int8 search copy on disk too, "
        "so it is not pinned into RAM. In memory pins that copy even when quantization says it is not. "
        "Apply writes this now, on ahead of time or off at any time. Points stay. "
        "Qdrant rebuilds segments in the background. Search then reads the int8 copy from disk."
    ),
    "qdrant_vectors_auto": (
        "On by default. Indexing moves the vectors to disk once spare memory cannot hold the int8 copy. "
        "It does not move them back. "
        "Turn this off to keep the vectors where you put them. "
        "Vectors on disk still applies immediately. "
        "Leaving this on and turning the vectors off lets indexing turn them on again when memory is short."
    ),
    "qdrant_graph_auto": (
        "Off by default. The graph is small next to the int8 copy. "
        "When this is on, indexing moves the graph to disk once spare memory cannot hold it. "
        "It does not move the graph back. "
        "Graph on disk still applies immediately. "
        "Leaving this on and turning the graph off lets indexing turn it on again when memory is short."
    ),
    "qdrant_apply": (
        "Write these choices. A storage-type change deletes the collection after you type DELETE, "
        "then creates an empty one. Indexing does not start. "
        "Quantization, where the vectors live, and the graph location change the live collection and keep the points."
    ),
    "models_page": (
        "The models the assistant can talk with. "
        "Pick one, change how it is named and how it answers, then Save. "
        "The line under the title is how many are installed."
    ),
    "models_list": (
        "Pick the model to look at. "
        "The list follows the Order numbers. A smaller number is higher. "
        "Models people can choose come first. "
        "Hidden models, the tasks helper, and the benchmark sit at the bottom, still in their own number order."
    ),
    "models_refresh": (
        "Read the installed models again. "
        "Use this after a model is installed or deleted somewhere else. "
        "Refresh does not change any settings."
    ),
    "models_detail": (
        "The file's own name, how many parameters it has, how tightly it was packed, and how big it is on disk. "
        "Q2_0 is a small pack and a little less exact. Q4 and Q8 are larger and closer to the original. "
        "The menu name can be friendlier. This line stays the file's own name."
    ),
    "models_in_use": (
        "Says when the assistant is set to this model right now. "
        "Deleting it moves the assistant to the next model that is still shown."
    ),
    "models_name": (
        "The words people see in the assistant's model menu. "
        "The file's real name stays the same, so updates still find it. "
        "Keep it short enough to read in the menu. At most 80 characters, on one line."
    ),
    "models_show": (
        "On, people can pick this model in the assistant. "
        "Off, it stays installed here and stays off that menu. "
        "Hiding it leaves the file on disk. "
        "The tasks helper and the benchmark stay off the assistant menu either way."
    ),
    "models_order": (
        "Where this model sits. A smaller number is higher. 1 is the first model people see. "
        "The assistant menu uses the same numbers. "
        "Hidden models, the tasks helper, and the benchmark stay out of that menu. "
        "On this page they sit at the bottom, still in their own number order."
    ),
    "models_chat": (
        "A model people talk to. It can appear in the assistant menu when Show is on. "
        "Use this for a normal chat model."
    ),
    "models_tasks": (
        "The small helper that writes chat titles and short suggestions. People never pick it. "
        "Only one model has this job, and it should stay small so it does not crowd the card. "
        "A large model is refused. "
        "Another model has to take this job before this one can return to chat."
    ),
    "models_job": (
        "Chat is a model people talk to. Tasks is the small hidden helper for titles and suggestions. "
        "Only one model is the helper. The benchmark stays on Llama 3.1 8B."
    ),
    "models_think": (
        "Thinking is the model working through the question before it answers. "
        "A careful answer takes longer. "
        "Pick one. Chat decides follows the switch in the assistant. "
        "On always thinks. Off answers right away. "
        "A model that cannot think leaves these grey."
    ),
    "models_think_follow": (
        "Chat decides means the thinking switch in the assistant controls this model. "
        "Switch that on and this model thinks. Switch it off and this model answers right away. "
        "This is the usual choice. "
        "Someone who already turned thinking on or off for this model in their own browser keeps that choice."
    ),
    "models_think_on": (
        "On means this model thinks before it answers, including when the assistant switch is off. "
        "The answer is more careful. The wait is longer. "
        "Someone who already turned thinking off for this model in their own browser keeps that until they switch it."
    ),
    "models_think_off": (
        "Off means this model answers right away and skips the working-out, including when the assistant switch is on. "
        "That is faster. The answer can be less careful. "
        "Someone who already turned thinking on for this model in their own browser keeps that until they switch it."
    ),
    "models_photos": (
        "Photos: yes means a picture sent to this model can be seen by it. "
        "Photos: no means the model only reads text. "
        "This comes with the file. A projector install is what adds pictures to a text file. "
        "This page cannot switch it by itself."
    ),
    "models_tools": (
        "Tools: yes means the model can call helpers, such as search, while it answers. "
        "Tools: no means it only answers from the question. "
        "This comes with the file. This page cannot switch it."
    ),
    "models_context": (
        "How much of the conversation this model holds at once. "
        "A bigger number remembers more of the chat and the library. It also uses more of the card, so a long chat can get slower or fail to load. "
        "Three limits apply, and the smallest one wins. "
        "The model has its own maximum. NOMAD has a cap, shown in the sentence. This page leaves that cap alone. "
        "The card has to hold the weight file and the conversation together, and it keeps a spare fifth. "
        "A bigger file leaves less room, so its list stops sooner. A small packed file can go higher. "
        "Auto lets NOMAD pick. A number is saved on the model. "
        "A number already saved above the fit can still appear in the list. NOMAD uses the smaller one. "
        "A quiet card is unused conversation room. The room is taken when the chat fills up."
    ),
    "models_temperature": (
        "How surprising the next word is. From 0 to 2. "
        "A low number, such as 0.2, stays with the obvious wording. Answers stay steady, and they can get repetitive. "
        "A higher number, such as 1.2, tries less likely words. Answers vary more, and they can wander or make things up. "
        "Many chats sit near 0.7. "
        "This is saved on the model and used when someone talks to it. Titles and suggestions ignore it. "
        "An empty box leaves the current number as it is."
    ),
    "models_top_p": (
        "How far down the likely words the model may look. From 0 to 1. "
        "It lines the words up from most likely to least, then keeps going until their chances add up to this number. "
        "A small number, such as 0.5, stays with the sure words. Answers sound confident, and they can be plain. "
        "A high number, such as 0.95, lets in more unusual words. Answers can be richer, or less reliable. "
        "1 keeps almost every word in play. "
        "This is saved on the model. An empty box leaves the current number as it is."
    ),
    "models_top_k": (
        "How many words are even allowed to be next. A whole number from 0 to 200. "
        "20 means only the 20 most likely words can be chosen, which keeps answers on topic. "
        "A larger number lets rarer words in. Answers can sound fresher, or drift off. "
        "0 turns this limit off. "
        "This is saved on the model. An empty box leaves the current number as it is."
    ),
    "models_min_p": (
        "The smallest chance a word needs before it can be picked. From 0 to 1. "
        "A higher number, such as 0.1, drops the weak words. Answers get tighter, and less playful. "
        "0 lets even unlikely words through. Type 0 to turn that floor off. "
        "An empty box does not mean 0. It leaves the current number as it is. "
        "This is saved on the model."
    ),
    "models_advanced": (
        "How the program reads this model's reply, including a thinking section when it has one. "
        "Leave this closed unless a model card gives you the names. "
        "A wrong name can scramble the answer. Empty leaves the current names in place. "
        "A new file does not copy these from the model that is selected."
    ),
    "models_renderer": (
        "How the model lays out its reply, including the thinking part when it has one. "
        "The model maker names this. A wrong name can mix the scratch work into the answer. "
        "Empty leaves the current name in place."
    ),
    "models_parser": (
        "How the program splits the reply into the thinking part and the answer people read. "
        "A wrong name can hide the answer or show the scratch work as the reply. "
        "Empty leaves the current name in place."
    ),
    "models_save": (
        "Writes the menu name, the order, Show, thinking, and the job. "
        "Context, temperature, top p, top k, and min p are written when you changed them. "
        "Those wait while indexing is running. The menu name can still be saved. "
        "A save that only changes the menu does not load the model onto the card."
    ),
    "models_test": (
        "Asks this model two short questions so you can see that it answers. "
        "One is with thinking off. When the model can think, another is with thinking on. "
        "The model is unloaded when the test ends, so the card is free again. "
        "This waits while indexing is running."
    ),
    "models_delete": (
        "Deletes this model from the disk after you confirm. The space comes back. "
        "The embedding model, the benchmark, and the current tasks helper stay until another model takes the tasks job. "
        "If people were chatting with this model, the assistant moves to the next one that is still shown. "
        "This waits while indexing is running."
    ),
    "models_install": (
        "Adds a model. Use a library tag to download one Ollama already knows, or pick a weight file from this computer. "
        "Fill in one of those. "
        "A library tag can update a model you already have. A file from disk needs a new name. "
        "The size is shown before anything is downloaded, when it can be read. "
        "This waits while indexing is running, because a download would share the card."
    ),
    "models_tag": (
        "A name from the Ollama library, such as gemma4:12b. Install downloads that model. "
        "The same name as one you already have updates that model and keeps its menu settings. "
        "Leave this empty when the model is a file on disk."
    ),
    "models_gguf": (
        "The model file on this computer. A GGUF is the packed weights Ollama runs. "
        "Pick the file, or type its path. "
        "Fill this in or a library tag, one of them. "
        "A file from disk needs a new name that is not already installed."
    ),
    "models_gguf_browse": (
        "Pick the GGUF weight file from this computer."
    ),
    "models_projector": (
        "An optional second file that lets a text model look at pictures. "
        "The GGUF is the model. The projector turns a photo into something that model can read. "
        "The file is often named mmproj or projector, and it has to be the one published for this same model. "
        "A matching projector is what turns Photos on. "
        "A projector from a different model can load and then describe the picture badly. "
        "Leave this empty for a text-only model, and when the library tag already includes pictures."
    ),
    "models_projector_browse": (
        "Pick the projector file that matches the GGUF. This is the picture add-on, not the weights."
    ),
    "models_new_name": (
        "The name Ollama will store a disk file under, such as my-model:27b. "
        "It has to be a new name. Spaces and brackets are refused. "
        "The friendly menu name is the other box, after the install. "
        "A library download keeps the library tag as its name, so this box is for a GGUF."
    ),
    "models_fork": (
        "Some packed files need a modified copy of Ollama. "
        "Names with PQ2_0, PTQ1_0, or MTP are those files. "
        "The usual Ollama will take the file and then run it wrong. "
        "Check this only when this computer is already running that modified copy. "
        "The install waits until the box is checked."
    ),
    "models_embed": (
        "Indexing uses this small model to turn pages into search data. "
        "It is not a chat model, so it is kept out of the list. "
        "Replacing it would make the knowledge base that is already stored stop matching."
    ),
}

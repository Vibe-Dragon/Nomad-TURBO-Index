# Log


The Log row shows and hides the log. **Errors**, **Info**, and **Debug** only change which lines the pane draws. The log file still records every line. **Follow** jumps to the newest line and stays there, including when several lines arrive before the page has scrolled. Scroll away to hold your place. That place stays until you return to the end or turn **Follow** on again. **Follow** off does not jump.

Log files are kept for the current copy of the knowledge index. Each day is its own file, and a new file starts at midnight. Opening the window continues today's file. Deleting the collection, from Turbo or from Project NOMAD, clears those files. If the first line of a log is older than the current collection, that log is cleared when Turbo starts. **Export log** is on the log row, at the right of **Errors**, **Info**, **Debug**, and **Follow**. **Log help** is to the right of **Export log**. **Export log** saves the kept files in one zip, including lines the log buttons are hiding. The suggested name is `nomad-turbo-logs.zip`.

**Log help** opens a window that explains the lines Turbo writes on purpose. A warning or traceback that window does not name is the program reporting what it hit.

On a long run the pane drops lines it has already shown, so the window still answers the close button. The day's log file keeps the lines.

## Common log lines

These are the lines people meet when Qdrant or the embed server is not ready to take work. The wording is the wording in the window and the log.

### Qdrant is not ready for changes

The status line under the page switch is one of these until the collection answers:

- `Qdrant is missing. Not ready for changes.` The container state replaces `missing` when Docker reports one.
- `Qdrant is running and has not passed the health check. Not ready for changes.`
- `Qdrant is running. The ready check has not passed. Not ready for changes.`
- `Qdrant is still writing. Do not restart it or delete the index.`

A large collection can be up and still on the last of those while it finishes a write. **Start indexing**, **Delete index**, **Scan**, and **Start repair** stay off. The second line names the container, how long it has been up, the health check, the ready check, and whether the collection answered. When the collection does not answer, that line also says a restart would replay the write, and it says whether the open segment is still gaining vectors. The check does not read the container log.

`Qdrant is ready for changes.` is the line that turns those buttons on. That line is green when the collection answered, blue while Turbo is in a working pause, amber while a write or an optimization should be left alone, and red when Qdrant needs attention. The line under it uses the same colors for Turbo itself. After Stop it says the knowledge-base records are updating and the file list stays still until that finishes.

### Qdrant stopped taking a write

A write that does not reach Qdrant is tried again. The log says `qdrant write timed out. Trying that batch again.` and includes the error Qdrant returned.

While indexing is running, a Qdrant problem is also logged as `Qdrant problem while indexing:`. That covers an optimizer error, a red collection, a collection that stops answering, and a status check that fails. The same problem is logged once. When the collection is answering again, the log says `Qdrant problem cleared. The collection is answering again.` A point count or stored-path read that fails uses the same `Qdrant problem while indexing:` wording. An optimization that is only still outstanding is not logged as a problem.

If it is still not taking the write, Turbo stays on that file. The status line says `Qdrant is still writing this file. Do not restart Qdrant or delete the index. Turbo is waiting here.` The log says `qdrant is not taking writes. Staying on <file> at article <n>: qdrant unreachable: ...`

`qdrant 400: ...`, `qdrant 503: ...`, and the other `qdrant <code>:` lines are answers Qdrant did return. Turbo does not treat those as "wait on this article." It tries that file once more from the same article (`will try <file> again from article <n> after an error`). A second failure at that same article is `file failed`.

### Indexing paused for a segment

When a plain segment is past the 4 GiB vector cap, the log says indexing is paused so a segment can be optimized, and that a plain segment is past the cap. When free space is under twice the segment that would be copied, the log says that instead. New points wait. Points already stored stay. The optimization bar starts over for that pause. Qdrant does not report a percent while it builds the segment. When the pause ends, the log says knowledge-base writes resumed.

While the window is open, a change in optimization state is one short line: not running, a segment build is open, the collection is yellow, waiting for an update, or status unknown. An optimizer error is a warning and keeps Qdrant's full message.

### Optimize stuck segments

Pressing the button writes `Optimize stuck segments: checking.` Then one of `refused`, `started`, `failed`, or, if the button was pressed while another job was still running, `not started`. A refusal names the reason: already in progress, segment files could not be read, nothing is stuck, not enough free space, or not enough spare memory. The collection is left unchanged. The same sentences are on the Repair status line.

### The embed server is not ready

Before Start, the count beside **Recommended** says `0 embed servers`. After Start, Turbo waits while `llama-server` loads the model. New waits are logged as `embedder not ready yet: ...`. That wait is about three minutes. The count then says `1 embed server`.

`<n> chunks per call` is the size of the next embed call. `embed batch:` is Auto, or the recommendation, saying why the count moved. `embed sequences <n>, <tokens> tokens each` means that one server added or dropped a second sequence. Each sequence keeps 2048 tokens. `long calls were slower with 2 sequences, staying at 1` means it went back to one sequence.

A call that gets HTTP 408, 429, 500, 502, 503, or 504 is logged as `embedder call failed (HTTP 503: ...). Trying again.` The code in the message is the code that came back. Turbo tries that call up to three times. The error that remains is `embedder HTTP 503: ...`. The file is then tried once more from the same article. A second failure at that article is `file failed`.

HTTP 400 is one input the embedder refused, not a dead server. The log says `cutting one chunk of <file> (<path>) in half so it fits the embedder. It was <n> tokens and the limit is <limit>.` Turbo cuts that piece and keeps the text. The article is not skipped. If that one chunk is still refused after it can no longer be cut, the log says `skipping one chunk of <file> (<path>) the embedder rejected`. That chunk counts as bad. The rest of the file continues.

A call that times out is cut in half until one chunk is left. That chunk is left unread and counts as bad. The log says `leaving one article of <file> (<path>) unread. The embedder timed out on it.` when the article had nothing else to store, or `skipping one chunk of <file> (<path>). The embedder timed out on it.` when the rest of the article continues. The file is not marked failed. Three such chunks in a row, with nothing embedded between them, means the embedder is not answering. The file stays unfinished and is tried again.

`the embed server is not running` means a call was ready and the server process was already gone.

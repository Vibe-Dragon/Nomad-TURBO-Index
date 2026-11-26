# Resource spikes while indexing


A run trades work between the embed server, Qdrant, and the disk. One of them getting busy while another goes quiet is that handoff.

### The GPU stays busy and articles per minute drop

A long page becomes many chunks. The embed server spends that time on the GPU. Articles per minute fall because the page takes a while. Chunks per minute on the stats line are the sign the run is still working. The file row's chk/s counts chunks as their vectors come back, so that column moves during a long embed call. The stats line still counts chunks after they are saved. A quiet disk during that stretch is normal.

### The GPU is quieter and the embed server uses more CPU

Short pages leave the graphics card waiting between calls. The embed server then spends more of its time on the CPU, packing those small chunks. Dictionaries and other short-article libraries look like this. Chunks per minute can stay healthy while the GPU graph sits well under full.

### CPU jumps for a few minutes and the GPU drops

Qdrant builds a new segment when the one it is writing fills. That build is CPU work: the payload indexes, then the search graph. Turbo waits until each batch is stored, so the embed server sits idle and GPU use falls until the build finishes. The collection can show yellow while those new points are not in the search graph yet. Closing a browser does not stop the build. Leave Qdrant running. A restart replays the unfinished write and can use more memory than the build did.

If a plain segment grows past 4 GiB of vectors, or free space is under twice the segment that would be copied, indexing pauses before the next upsert. Points already stored stay. The optimization bar on the Index page is that pause. Making this rewrite faster is **Speed up segment optimization**. That work is on the back burner. It has not been a large bottleneck. This version does not change how Qdrant builds the graph.

### CPU stays high after the Knowledge Base page is closed, and the article count stops

Project NOMAD's Knowledge Base page asks Qdrant to set `active` on every stored point that does not already have it. The page can time out and be closed while Qdrant is still doing that rewrite. Turbo waits on the current file, and the GPU drops because new batches are not being stored. Points Turbo stores already include `active`. The rewrite is the older points. Leave Qdrant running, and leave that page closed during a run.

### Memory climbs over a long run

Qdrant keeps the search graph in RAM. The graph grows as more of the library is stored. That climb is the index. A restart loads the graph again from disk, and a large index can use more memory on the way back up than it did while it was running.

### The disk is busy and the next batch waits

Reading a ZIM and writing the index share the disk. When the disk is saturated, Turbo waits before the next upsert. A download onto the same disk slows both.

### Start spends a short while loading the model

The embed server loads the model once when a run starts. The log says `embedder not ready yet` until that load finishes. Card memory steps up by the model and stays there. The count beside **Recommended** then says `1 embed server`.

### One page holds the article count

A very long page is read and split before its chunks are stored. The article count sits on that page while the reader and the embed server work through it. The next article shows up when that page has been stored. Chunks per minute still move if the page is producing chunks.

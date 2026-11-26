# What comes next

The open next effort is the [headless web UI](https://github.com/Vibe-Dragon/Nomad-TURBO-Index/milestone/3).

The desktop window stays. A second way to run Turbo should look and work the same: the same pages, controls, and results. A Project NOMAD install should be able to add it, as a Docker container or the same kind of extra service, so it can run on a headless server. Installing it does not start indexing and does not change stored points.

OCR in this version keeps a PDF page's text layer. A page is drawn when it has no text, or when **OCR every PDF** is on. A picture is still read. Words the text layer does not have are added when a page is drawn. The row says OCR, the bar shows the page, and the file is not handed to NOMAD. Pages are read together, and a page that is drawn is drawn once. That work is [Optimize OCR](https://github.com/Vibe-Dragon/Nomad-TURBO-Index/milestone/2). It is not the open next effort.

**Speed up segment optimization** stays open and is on the back burner. The rewrite has not been a large bottleneck. After the write hold, Qdrant still builds the search graph for a large segment in memory, the indexed-point count stays flat until the segment is committed, and Qdrant does not publish a percent. This version shows that wait and refuses an unsafe rewrite. It does not make the build itself faster. That milestone is [Speed up segment optimization](https://github.com/Vibe-Dragon/Nomad-TURBO-Index/milestone/1).

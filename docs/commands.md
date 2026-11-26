# Commands


The desktop icon runs `nomad-turbo toggle`. It opens or raises the window. It does not start or stop indexing.

- `nomad-turbo plan` prints the call size for the computer it is running on. It does not pause NOMAD or start containers.
- `nomad-turbo status` prints whether indexing is running, or only the window is open.
- `nomad-turbo stop` ends the embed that is already running and keeps the last article that was fully saved.
- `nomad-turbo recover` puts the ingest policy back and resumes NOMAD embedding after a crash that left the queue paused. A repair delete that may still be running leaves embedding paused.

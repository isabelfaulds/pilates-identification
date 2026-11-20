# Lagree video tools

Run from the project root:

```sh
npm install --prefix tools/lagree
node tools/lagree/download.cjs --list
node tools/lagree/download.cjs
```

Edit `output/lagree/micro-exercises.csv`: set `training_data` to `1` for videos
to download and `0` for the rest. The downloader reads selected names and URLs
from the CSV. It saves to `data/raw/` with lowercase filenames and whitespace
replaced by underscores: `Elevator lunge` becomes `elevator_lunge.mp4`.
An optional CSV `folder` column can override the destination relative to the
project root. Duplicate destinations cause an error rather than overwriting
another selected video.

The script logs in once, downloads sequentially, skips existing nonempty files,
and uses a `.part` file until a download finishes. Use `--force` to replace
selected files or `--list` to preview without opening the browser. Failures are
reported per exercise while the rest of the batch continues.

Requires Node.js and Google Chrome. Reads `lagree_od_username` and
`lagree_od_password` from the root `.env`. Credentials and videos are ignored by
Git; browser cookies and profiles are not saved.

## Refresh catalog

```sh
node tools/lagree/catalog.cjs
```

Writes an alphabetical Markdown list, CSV, and JSON to `output/lagree/`.
Follows the logged-in library's pagination through its final response.
Catalog refreshes preserve `training_data` selections by exercise ID; new rows
default to 0.

## Extract landmarks

See [the pose pipeline](../../docs/pose-pipeline.md). It processes all local
videos under `data/raw/`, independently of the CSV download selections, and saves
results under `data/processed/landmarks/`.

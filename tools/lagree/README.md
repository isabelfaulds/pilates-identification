Run from the project root folder:

```sh
npm install --prefix tools/lagree
node tools/lagree/download.cjs
```

Edit the `videos` array in `download.cjs` to select exercises:

```js
{
  url: 'https://www.lagreeod.com/exercises/jMjQb1wQ',
  name: 'Elevator lunge',
  folder: 'training_data',
}
```

Each entry saves to its folder, relative to project root. Names become lowercase
filenames with whitespace replaced by underscores: `elevator_lunge.mp4`.
You can use a subfolder such as `training_data/elevator_lunge` to organize groups.
Duplicate destination paths are rejected to avoid overwriting different entries.

The script logs in once, downloads sequentially, skips existing nonempty files,
and uses a `.part` file until each download finishes. Use `--force` to replace
existing files, or `--list` to preview destinations without opening the browser.
Failures are reported per exercise; other entries continue.

Edit `output/lagree/micro-exercises.csv`: set `training_data` to `1` for
exercises to download and `0` for the rest. Only Elevator lunge is selected
initially. The downloader reads names and URLs from selected rows, saves to
`training_data/`, and changes lowercase names' whitespace to underscores
(for example, `Elevator lunge` becomes `elevator_lunge.mp4`).

```sh
node tools/lagree/download.cjs --list   # Preview selected names and destinations
node tools/lagree/download.cjs          # Download selected videos; skip existing files
node tools/lagree/download.cjs --force  # Replace selected files
```

Signs in once and downloads sequentially. Different selected videos with the
same filename cause an error rather than silently overwriting one another.
Requires Node.js and Google Chrome. Reads `lagree_od_username` and
`lagree_od_password` from the root `.env`. Credentials and training_data are
gitignored. It does not save login cookies or browser profiles.

To collect the Micro exercise catalog without downloading videos:

```sh
node tools/lagree/catalog.cjs
```

Writes an alphabetical Markdown list, CSV, and JSON to `output/lagree/`.
Uses the logged-in library's pagination and checks for its final response.

Catalog refreshes preserve the training_data selections by exercise ID. New rows default to 0.

# Pilates Identification

Exercise video selection and download tooling for a Pilates identification project.
The catalog currently contains 1,160 Lagree Micro exercise videos, with named
movement groups and a `training_data` selection column.

## Setup

Requires Node.js and Google Chrome.

```sh
npm install --prefix tools/lagree
cp .env.example .env
```

Fill in your Lagree credentials in `.env`. Credentials, downloaded videos, and
browser sessions are excluded from Git.

## Choose and download training videos

Edit `output/lagree/micro-exercises.csv`. Set `training_data` to `1` to select a
video, or `0` to leave it out. The `group` column can contain multiple groups
separated by semicolons.

```sh
node tools/lagree/download.cjs --list
node tools/lagree/download.cjs
```

Videos save to `training_data/` with lowercase names and underscores. Existing
files are skipped; use `--force` to replace them. Download only videos you have
permission to save.

## Refresh the catalog

```sh
node tools/lagree/catalog.cjs
```

Refreshes the catalog while preserving training selections by exercise ID.
See `tools/lagree/README.md` for details.

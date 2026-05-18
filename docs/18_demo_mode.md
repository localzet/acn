# Demo Mode

ACN demo mode is a deterministic presentation workflow for showing adaptive continual learning
without requiring a live backend.

## Command

```bash
make demo-mode
```

The command generates reproducible demo assets and starts the web dashboard with
`VITE_DEMO_MODE=true`.

## Features

- guided experiment playback;
- animated commit graph;
- live metric updates;
- rollback visualization;
- branch evolution visualization;
- controller decision timeline;
- human override simulation;
- forgetting visualization;
- dark presentation mode.

## Screenshot Export

Generate presentation assets without starting the web server:

```bash
make demo-assets
```

Default outputs are written to `experiments/acn-demo-mode`:

- `demo_summary.json`
- `demo_presentation.svg`

## Preset

The reproducible demo preset lives in `configs/demo/acn_demo_mode.json`.

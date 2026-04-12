# AST speech summarization

Shortens long speech using the **Audio Spectrogram Transformer (AST)** attention: one contiguous excerpt (default **60 seconds**) with the highest saliency.

**Full technical documentation** (file map, pipeline flow, algorithms, limitations): [docs/PIPELINE_AND_IMPLEMENTATION.md](docs/PIPELINE_AND_IMPLEMENTATION.md).

## Layout

| Path | Purpose |
|------|---------|
| `ast/` | Upstream AST model (`src/models/`) |
| `ast_summarize/` | Summarization pipeline and CLI |
| `examples/` | Put your input audio here; write outputs to `examples/output/` |

## Setup

1. Python 3.10+ recommended.
2. Install [ffmpeg](https://ffmpeg.org) and add it to your `PATH`. **Required for MP3/M4A** (WAV/FLAC use soundfile).
3. Create a venv and install:

```bash
pip install -r requirements.txt
```

On first run, timm may download ImageNet DeiT weights into `ast/pretrained_models/`.

## Run

From the **project root** (this folder):

```bash
python ast_summarize/run_ast_summarize.py --audio examples/audio/your.mp3 --out-wav examples/output/summary_griffinlim.wav --out-wav-cuts examples/output/summary_cuts.wav
```

Use the real filename (e.g. `your.mp3` or `your.wav`). If you see *file not found*, the CLI lists other files in that folder.

- **`--out-wav-cuts`**: use this file for listening (real audio from the original recording).
- **`--summary-duration-sec 60`**: length of the excerpt (default 60). Use `0` for a short-window mode; see `--help`.
- **`--debug-dir path`**: optional numpy/text dumps for debugging.

### Optional: merge many `.flac` / `.wav` chunks

```python
from pathlib import Path
from ast_summarize.chunk_audio import concat_audio_files, discover_flac_chunks
p = Path("examples/audio/my_chapters")
concat_audio_files(discover_flac_chunks(p), Path("examples/output/combined.wav"))
```

Then run the CLI on `combined.wav`.

## License

`ast/` follows the upstream AST repository license. Application code in `ast_summarize/` is provided as-is for this project.

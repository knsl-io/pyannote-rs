# pyannote-rs — ScribeFlow Patched Fork

This repository is a patched fork of [`thewh1teagle/pyannote-rs`](https://github.com/thewh1teagle/pyannote-rs) used by the `pyannote3-onnx` diarization backend in [ScribeFlow](https://github.com/knsl-io/scribeflow). The upstream `0.3.4` crate (latest on crates.io as of 2026-05) does not produce any speech segments on standard PCM WAV input. This fork combines two unmerged upstream PRs and adds a few additional patches so the backend works against the v0.1.0 release ONNX models.

## Required models (download once)

The patches in this fork target **a specific snapshot of the ONNX models** — the one bundled with the [pyannote-rs v0.1.0 GitHub release](https://github.com/thewh1teagle/pyannote-rs/releases/tag/v0.1.0). Other snapshots (e.g. [`onnx-community/pyannote-segmentation-3.0`](https://huggingface.co/onnx-community/pyannote-segmentation-3.0) on HuggingFace) name their tensors differently and **will fail with `Invalid input name` errors** against this fork — see the patch list below for the schema differences.

```bash
# Segmentation (pyannote 3.0; "input" / "output" tensor names)
curl -L -o segmentation-3.0.onnx \
  https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/segmentation-3.0.onnx

# Embedding (WeSpeaker CAM++; "feats" input, single positional output)
curl -L -o wespeaker_en_voxceleb_CAM++.onnx \
  https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/wespeaker_en_voxceleb_CAM++.onnx
```

Sizes: segmentation ~6 MB, embedding ~28 MB.

The original models are `pyannote/segmentation-3.0` and `pyannote/wespeaker-voxceleb-resnet34-LM` (CAM++ variant) from the HuggingFace Hub; the v0.1.0 release just bundles ONNX exports of those.

## How this fork was assembled

1. **Base**: `tmoroney/pyannote-rs`, branch `main`, commit `2348012549` — open upstream **PR #22** ("Improve segmentation logic and fix no speech segments being output for some audio"). Rewrites the segmentation iterator with overlapping windows, adaptive frame tracking, end-of-audio flushing, and introduces the `knf-rs` Kaldi fbank submodule for feature extraction.
2. **PR #28 applied on top**: `gregoire22enpc/pyannote-rs`, branch `fix/segmentation-normalization`, commit `0035386b4d` — open upstream **PR #28** ("fix: normalize i16 to f32 in segmentation"). Adds the missing `/ 32768.0` normalization that the segmentation ONNX expects.
3. **ScribeFlow model-schema patches** (see "Patches" below) — adjust hardcoded ONNX input/output names to match the model snapshots in `scribeflow-transcriber-rs/models/`, which come from the [pyannote-rs v0.1.0 release](https://github.com/thewh1teagle/pyannote-rs/releases/tag/v0.1.0). PR #22's authors targeted the `onnx-community/pyannote-segmentation-3.0` HuggingFace export instead, which has different input/output tensor names.

## Patches

### 1. `src/segment.rs` — i16 → f32 normalization (from PR #28)

The segmentation ONNX expects samples normalized to `[-1.0, 1.0]`. PR #22 alone casts `i16` to `f32` without dividing, so values land in `[-32768, 32767]` and the model classifies everything as non-speech.

```diff
- let window_f32 = window.iter().map(|&x| x as f32).collect::<Vec<_>>();
+ let window_f32 = window
+     .iter()
+     .map(|&x| x as f32 / 32768.0)
+     .collect::<Vec<_>>();
```

### 2. `src/segment.rs` — segmentation input/output names

PR #22 hardcodes `"input_values"` / `"logits"`. The `segmentation-3.0.onnx` we use (from the pyannote-rs v0.1.0 release) names them `"input"` / `"output"`.

```diff
- let inputs = ort::inputs!["input_values" => tensor];
+ let inputs = ort::inputs!["input" => tensor];
  ...
- let ort_out = match ort_outs.get("logits") ...
+ let ort_out = match ort_outs.get("output") ...
```

### 3. `src/embedding.rs` — embedding input name

PR #22 hardcodes `"fbank_features"`. The WeSpeaker ONNX exports we use (`wespeaker_en_voxceleb_resnet34.onnx`, `wespeaker_en_voxceleb_CAM++.onnx`, `wespeaker-resnet34.onnx`) all name the input `"feats"`.

```diff
- let inputs = ort::inputs![
-     "fbank_features" => Tensor::from_array(...)
- ];
+ let inputs = ort::inputs![
+     "feats" => Tensor::from_array(...)
+ ];
```

### 4. `src/embedding.rs` — embedding output (read by position)

PR #22 looks up the output by name `"embeddings"`. Our WeSpeaker exports name it `"embs"`, but the model only has one output, so reading by position (`outputs[0]`) is robust to either naming.

```diff
- let ort_out = ort_outs
-     .get("embeddings")
-     .context("Output tensor 'embeddings' not found")?
-     .try_extract_tensor::<f32>()?;
+ let ort_out = ort_outs[0]
+     .try_extract_tensor::<f32>()?;
```

## Caller-side patch (in scribeflow-transcriber-rs)

PR #22 also changed `EmbeddingExtractor::compute()` to return `Vec<f32>` directly (in `0.3.4` it returned an iterator). The wrapper in `scribeflow-transcriber-rs/src/pipeline/pyannote3_onnx_diarizer.rs` removed its trailing `.collect()` to match.

## How to verify the patches

You can probe the input/output names of any local ONNX model against our patched fork:

```bash
cd crates/pyannote-rs
# (no `probe` example committed — write a 6-line one if you need it)
```

Or directly via Python (in a venv with `onnxruntime`):

```python
import onnxruntime as ort
s = ort.InferenceSession("../../scribeflow-transcriber-rs/models/segmentation-3.0.onnx")
print("inputs:", [(i.name, i.shape) for i in s.get_inputs()])
print("outputs:", [(o.name, o.shape) for o in s.get_outputs()])
```

Expected for `segmentation-3.0.onnx`: input `input`, output `output`.
Expected for `wespeaker_en_voxceleb_*.onnx`: input `feats`, output `embs`.

## When to revert

Drop this vendored fork and go back to a published `crates.io` version once **all four** of the following are upstreamed:

- [ ] PR #22 (iterator rewrite + flush) — `tmoroney/pyannote-rs`
- [ ] PR #28 (`/ 32768.0` normalization) — `gregoire22enpc/pyannote-rs`
- [ ] Configurable / discovered ONNX input names (today PR #22 hardcodes the `onnx-community` schema)
- [ ] Configurable / positional ONNX output reading

In the meantime, **do not bump** the `pyannote-rs` version in `scribeflow-transcriber-rs/Cargo.toml` without re-applying these patches against the newer upstream tree.

## Submodule

This fork carries a git submodule under `crates/knf-rs/sys/knf` pointing at `csukuangfj/kaldi-native-fbank` (used to compute mel-filterbank features for WeSpeaker embeddings). It is initialized on first use:

```bash
cd crates/pyannote-rs
git submodule update --init --recursive
```

If you ever pull this whole monorepo fresh, the submodule init step must happen once before any `cargo build` of `scribeflow-transcriber-rs` with `--features pyannote3-onnx`.

## Build prerequisites

The `knf-rs-sys` build script requires `cmake` and a C++ toolchain. On macOS:

```bash
brew install cmake
xcode-select --install   # if not already installed
```

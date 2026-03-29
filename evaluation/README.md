# Evaluation Workflow

Use this directory to store verified reference transcripts for objective speech-to-text evaluation.

## Manifest format

Create a JSON file shaped like this:

```json
[
  {
    "name": "sermon-sample",
    "audio_path": "../sermon.ogg",
    "reference_path": "references/sermon.txt"
  }
]
```

`reference_path` is resolved relative to the manifest file.
You can also use `reference_text` instead of `reference_path` for short samples.

## Run evaluation

```bash
python evaluate_transcriptions.py evaluation/manifest.example.json --profiles none offline live
```

The evaluator now prints progress before and after each clip so long CPU runs do not appear stuck.
By default it also uses the `tiny` model and a fast decoding preset for quicker benchmarking.

For the fastest comparison run from the repo root:

```bash
python evaluate_transcriptions.py manifest.example.json --profiles none offline live
```

If you need slower but stricter decoding, add `--accurate-decode` and optionally choose a larger model:

```bash
python evaluate_transcriptions.py manifest.example.json --model base --profiles none offline live --accurate-decode
```

The evaluator prints per-clip WER and real-time factor, then an aggregate summary per preprocessing profile.

## Recommended process

1. Create reference transcripts from manually verified text, not from the model output.
2. Keep the same manifest and model when comparing preprocessing changes.
3. Compare `none`, `offline`, and `live` profiles to see whether cleanup improves WER or only changes latency.
4. Long clips scale linearly with runtime. The bundled `sermon.ogg` sample is about 13.5 minutes, so evaluating all three profiles means transcribing more than 40 minutes of audio on CPU.

# Full labelled MATH and MMLU upload to Cloudflare R2

## Scope and layout

The three instruct models are Llama-3.2-1B-Instruct, Qwen2-7B-Instruct and Meta-Llama-3-8B-Instruct.
Each has MATH 5,000 and MMLU 14,042: **57,126 responses in 573 completed shards**.
The original receipt inventory contains 2,205,181,553,446 bytes in 3,855,628 files, excluding the copy receipts themselves.

Destination: `s3://autoact-data/openact/math-mmlu-full-20260916/`.

- `archives/<run>/<model>/<shard>.tar`: deterministic uncompressed USTAR archives, preserving every original file byte.
- `metadata/<run>/<model>/<shard>/`: standalone questions/responses, correctness labels, source manifests and verification receipts.
- `indices/<run>/<model>/<shard>.json`: each archive member's byte offset, size and SHA256; enables HTTP Range access without downloading a whole archive.
- `receipts/`: verified upload receipts.
- `manifest.json`: coverage and source inventory.
- `_SUCCESS.json`: written only after all 573 archives and metadata are uploaded and verified.

Extract each archive into its `<run>/<model>/` folder to restore the original OpenAct layout. Activations are not normalized, recomputed or requantized. Both prompt-last and complete generated-token activations, means, and final RMSNorm pre/post are retained. Original NFS data is retained.

## Integrity and resume

Before upload, validate the 573 source completion/copy receipts, continuous coverage, model IDs, label/sample alignment, missing-label/error status and required full activation arrays. While packing, verify every original file against its existing SHA256 receipt and reject extra/missing/symlink files or changed source data.

The uploader uses 64 MiB multipart parts, initially four concurrent shard streams and eight concurrent parts per stream. It streams from Lambda NFS directly to R2, with bounded RAM and no local archive staging. It has no bandwidth limit and uses no GPU.

Live run: the first 2,487,848,960-byte archive passed source checks, multipart checks and remote Range checks in 69 seconds. Eight concurrent streams sustained approximately 150 MB/s initially. The run was then resumed with **16 shard streams × 8 multipart slots** to test higher aggregate throughput; completed objects and accepted parts were preserved. `parallelism_maintenance.json` records this controlled transition. NFS checkpoint reads retry transient `ESTALE` errors caused by atomic status replacement.

Each part sends Content-MD5 and verifies the returned ETag. Final verification checks composite ETag and object length, then downloads labels, manifest and representative tensor chunks via Range GET and checks their original SHA256. This does not claim a second full-object download verification. Full archive SHA256 and member hashes remain recorded.

Accepted parts and upload IDs persist on NFS. After interruption, deterministic replay skips matching accepted parts. A previously completed archive is not overwritten; it requires matching source and completion records. Failed source integrity checks stop that shard. Upload errors keep checkpoint state and original data.

## Execution

Configuration: `configs/r2_math_mmlu_upload.toml`. The private credential file is outside the repository, owner-readable only, and never logged or uploaded.

```bash
# Dedicated upload environment; does not alter the collection environment.
/home/ubuntu/.venvs/openact-r2/bin/python scripts/upload_r2_collection.py --prepare-only
/home/ubuntu/.venvs/openact-r2/bin/python -u scripts/upload_r2_collection.py --limit 1 --workers 1
/home/ubuntu/.venvs/openact-r2/bin/python -u scripts/upload_r2_collection.py --workers 16
```

State/logs: `/lambda/nfs/dami/openact/runs/r2_math_mmlu_20260922/`.
`status.json` distinguishes accepted in-progress parts from completely verified shards; only the latter count as complete.
The existing half-hour monitoring automation now checks both Safety collection and this transfer, staying quiet during normal progress and notifying on actionable failure/recovery or completion.
Source changes are deployed through local commit → GitHub push → Lambda fetch/fast-forward.

References: [R2 multipart uploads](https://developers.cloudflare.com/r2/objects/upload-objects/), [S3 API compatibility](https://developers.cloudflare.com/r2/api/s3/api/).

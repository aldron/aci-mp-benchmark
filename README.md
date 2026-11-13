# ACI-MP benchmark (experimental)

C++17 AVX2 approximate coordinate-group multi-probe ANN index, with a Python/ctypes benchmark against Faiss FlatL2, HNSWFlat, IVFFlat and IVFPQ.

## Setup (Linux x86-64 CPU supporting AVX2)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make
pytest -q
python bench.py --n 100000 --nq 100 --k 10 --posting 20000 --rerank 1000
```

Output: `results/benchmark.csv` and per-configuration JSON lines on stdout. For more stable p99 use `--nq 1000`. Runs are single-query sequential and exclude index building from latency. FlatL2 is exact ground truth. Reported `index_bytes` for ACI is an estimate of C++ allocations, and Faiss index size uses serialized bytes (not identical to live RSS). `peak_rss_kib` includes the whole Python process and all indexes, so it is **not** per-index memory. Memory figures are not directly comparable without an isolated-process RSS experiment.

## Sweeps

```bash
python bench.py --n 100000 --nq 1000 --groups 2 4 8 --bins 8 16 32 --postings 5000 20000 50000 --reranks 200 1000 5000 --out results/sweep.csv
python bench.py --n 100000 --nq 1000 --mode normalized --out results/normalized.csv
python bench.py --n 100000 --nq 1000 --mode correlated --out results/correlated.csv
```

For N scaling, repeat with `--n 10000`, `--n 100000`, and `--n 1000000` (requires substantial RAM/time). `--threads 1` keeps Faiss CPU parallelism controlled. Repeat each configuration across multiple seeds and report medians/confidence intervals before interpreting performance.

## Method

Each group contains adjacent coordinate dimensions. Values are quantized to equal-width bins; nonempty group-cell signatures map to ID posting lists. Query probes group cells using a global min-heap ordered by squared query-to-cell-box lower bound. It accumulates votes per ID (number of encountered group postings) and the sum of cell bounds. Candidates are ranked by vote count descending, then cell bound sum ascending, and up to `rerank_budget` are reranked with an AVX2 L2 implementation. Partial L2 sums permit safe early rejection against the current top-K distance.

`posting_budget` limits the total ID entries visited. When a posting is oversized, the implementation reads its first remaining-budget entries, creating **ID-order bias**; this is an explicit experimental limitation. Query exploration is capped at 100000 cells and can terminate early, potentially yielding fewer than K results. Group-cell traversal may be expensive in sparse high-dimensional regions. Priority-queue duplicates are ignored on pop, but still consume heap space. Float32 squared L2 results are subject to small floating-point differences. The method is approximate and **does not guarantee 95% recall**.

## Evaluation caveats

- The Gaussian/correlated/normalized datasets are synthetic, not SIFT1M. Real data should be loaded separately to test embedding distributions.
- HNSW uses M=32, efConstruction=200, efSearch sweep; IVF uses nlist≈sqrt(N), nprobe sweep; IVF-PQ uses m=16, 8-bit subquantizers. These are starting settings, not globally optimal tuning.
- ACI's coordinate index and the baselines use different memory footprints. Compare at matched recall and (ideally) matched memory budgets.
- The benchmark reports `recall`, latency p50/p95/p99, QPS, build seconds, index bytes, and ACI posting/candidate/rerank/probe means.
- The C++ code uses AVX2 but not FMA, making it usable on AVX2-only CPUs.
- No performance results are included or claimed until the full benchmark is executed.

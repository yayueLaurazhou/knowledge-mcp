# Knowledge Base

Drop your CUDA/Triton documentation, papers, and code files here.

## Supported formats
- `.md` / `.txt` / `.rst` — documentation, notes, paper summaries
- `.py` — Triton kernels, PyTorch CUDA extensions
- `.cu` / `.cuh` / `.c` / `.cpp` — CUDA C++ kernels
- `.triton` — raw Triton IR

## Suggested structure
```
knowledge/
  cuda/          # CUDA programming guide excerpts, PTX notes
  triton/        # Triton tutorial snippets, API docs
  papers/        # arXiv summaries or full-text markdown exports
  kernels/       # Reference kernel implementations
  tricks/        # Optimization recipes (tiling, vectorized loads, etc.)
```

Files are indexed with BM25 on every query — no rebuild step needed.

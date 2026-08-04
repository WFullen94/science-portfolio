"""Stage 3 — Distributed training (DDP → FSDP) of the time-series Transformer.

Distributed data-parallel is a multi-*process* pattern, not a multi-GPU-only one:
each rank holds a replica (DDP) or a shard (FSDP) of the model, trains on its slice
of the data (a `DistributedSampler`), and the collective all-reduces gradients so
every rank stays in sync. That machinery runs correctly on CPU via the `gloo`
backend — which is what we do here (no CUDA on this box), self-launched with
`mp.spawn`. The *identical code* scales to real multi-GPU with
`torchrun --nproc_per_node=N` and `backend=nccl` on CUDA (see the README).

  * **DDP** — every rank keeps a full model replica; gradients are all-reduced each
    step. Simple, fast, the default. Memory scales with model size per GPU.
  * **FSDP** — parameters, gradients, and optimizer state are *sharded* across ranks
    and all-gathered just-in-time for each layer's forward/backward. Trades
    communication for memory, so models too big for one GPU still fit. We wrap the
    same model and pull a FULL_STATE_DICT to rank 0 for evaluation — the standard
    FSDP checkpoint pattern.

Run:  python -m structdl.train_distributed --strategy ddp   (or fsdp, or `make ddp`)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from structdl.config import load_config
from structdl.metrics import binary_metrics
from structdl.sequence_data import build_sequence_bundle
from structdl.ts_transformer import TSTransformer


def _setup(rank: int, world_size: int, dcfg):
    os.environ["MASTER_ADDR"] = dcfg["master_addr"]
    os.environ["MASTER_PORT"] = str(dcfg["master_port"])
    dist.init_process_group(dcfg["backend"], rank=rank, world_size=world_size)
    torch.manual_seed(dcfg["random_state"])  # deterministic, same across ranks


def _wrap(model, strategy: str):
    if strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        # use_orig_params keeps parameters individually addressable (cleaner optimizer
        # + checkpoint story). FSDP auto-detects the CUDA device on a GPU host.
        return FSDP(model, use_orig_params=True)
    return DDP(model)  # CPU DDP needs no device_ids


def _evaluate(model, strategy: str, b, features, tscfg, rank: int):
    """Rank-0 evaluation. DDP: use the local replica. FSDP: gather a full state dict."""
    if rank != 0 and strategy != "fsdp":
        return None

    if strategy == "fsdp":
        # FULL_STATE_DICT is a COLLECTIVE — all ranks must enter it — but it
        # materializes the unsharded weights only on rank 0.
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            full_sd = model.state_dict()
        if rank != 0:
            return None
        eval_model = TSTransformer(len(features), tscfg)
        eval_model.load_state_dict(full_sd)
    else:
        eval_model = model.module  # DDP replica is complete on every rank

    eval_model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(eval_model(torch.from_numpy(b.Xtest).float())).numpy()
    return binary_metrics(b.ytest, scores)


def _worker(rank: int, world_size: int, args, cfg):
    dcfg = cfg["distributed"]
    _setup(rank, world_size, dcfg)
    tscfg = cfg["ts_transformer"]

    b = build_sequence_bundle(cfg)  # deterministic → identical on every rank
    ds = TensorDataset(torch.from_numpy(b.Xtrain).float(),
                       torch.from_numpy(b.ytrain).float())
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=dcfg["batch_size"], sampler=sampler)

    model = _wrap(TSTransformer(len(b.features), tscfg), args.strategy)
    pos = float((b.ytrain == 1).sum()); neg = float((b.ytrain == 0).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos if pos else 1.0]))
    opt = torch.optim.AdamW(model.parameters(), lr=dcfg["lr"],
                            weight_decay=dcfg["weight_decay"])

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[dist] strategy={args.strategy} world_size={world_size} "
              f"backend={dcfg['backend']} | {len(ds)} train windows "
              f"→ {len(sampler)}/rank | model params (rank view)={n_params:,}")

    for epoch in range(1, dcfg["epochs"] + 1):
        sampler.set_epoch(epoch)  # reshuffle differently each epoch, consistently
        model.train()
        running = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item()
        if rank == 0:
            print(f"[dist] epoch {epoch:02d}  rank0 mean loss={running / len(loader):.4f}")

    dist.barrier()
    metrics = _evaluate(model, args.strategy, b, b.features, tscfg, rank)
    if rank == 0 and metrics is not None:
        print(f"[dist] DONE ({args.strategy}, {world_size} ranks) — "
              f"test ROC-AUC={metrics['roc_auc']:.4f} F1={metrics['f1']:.4f}")
    dist.destroy_process_group()


def main() -> int:
    cfg = load_config()
    dcfg = cfg["distributed"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["ddp", "fsdp"], default=dcfg["strategy"])
    ap.add_argument("--nproc", type=int, default=dcfg["nproc"])
    args = ap.parse_args()

    # FSDP needs a real accelerator: torch's FSDP can't initialize on a CPU/MPS-only
    # host (it probes MPS for a device API it lacks, and the CPU flat-param path
    # segfaults). The wrapping + FULL_STATE_DICT checkpoint below are the same on a
    # CUDA box — run `--strategy fsdp` there with backend=nccl. DDP demonstrates the
    # distributed mechanics end-to-end on this machine.
    if args.strategy == "fsdp" and not torch.cuda.is_available():
        print("[dist] FSDP requires CUDA — torch FSDP does not run on a CPU/MPS host.\n"
              "       The FSDP wiring (shard wrap + FULL_STATE_DICT eval) is in _wrap/\n"
              "       _evaluate; launch it on a GPU box: backend=nccl, one rank per GPU.\n"
              "       Use `--strategy ddp` here to run the distributed path on CPU (gloo).")
        return 0

    print(f"[dist] spawning {args.nproc} '{args.strategy}' workers on {dcfg['backend']} "
          f"(CPU). Same code → multi-GPU via torchrun + nccl.")
    mp.spawn(_worker, args=(args.nproc, args, cfg), nprocs=args.nproc, join=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

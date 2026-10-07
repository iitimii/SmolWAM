"""End-to-end check of the training code on a tiny synthetic latent bank: train (with the curriculum switch and
KD targets), then export the checkpoint to a release directory and load it back."""

import json

import numpy as np
import torch
from safetensors.torch import save_file

from smolwam.config import SmolWAMConfig
from smolwam.export import export
from smolwam.hub import load_release
from smolwam.layout import ACTION_DIM, CAMERAS, CHUNK, LATENT_CH, LATENT_HW, PROPRIO_DIM
from smolwam.model import SmolWAMModel
from smolwam.policy import load_model
from smolwam.text import TextCache
from smolwam.train.train import parser, train


def make_bank(root, cfg, n_episodes=4, steps=40, stride=4):
    rng = np.random.default_rng(0)
    bank = root / "bank"
    bank.mkdir()
    ep_len = [steps] * n_episodes
    rows = [(e, t) for e in range(n_episodes) for t in range(e % stride, steps, stride)]
    row_start = [next(i for i, (e2, _) in enumerate(rows) if e2 == e) for e in range(n_episodes)]
    meta = {
        "num_rows": len(rows),
        "stride": stride,
        "instructions": ["pick it up", "put it down"],
        "row_start": row_start,
        "ep_start_full": [e * steps for e in range(n_episodes)],
        "ep_len": ep_len,
        "ep_instr": [e % 2 for e in range(n_episodes)],
    }
    (bank / "meta.json").write_text(json.dumps(meta))
    for c in CAMERAS:
        np.save(
            bank / f"lat_{c}.npy", rng.standard_normal((len(rows), LATENT_CH, LATENT_HW, LATENT_HW)).astype(np.float16)
        )
    np.save(bank / "proprio_full.npy", rng.uniform(-1, 1, (sum(ep_len), PROPRIO_DIM)).astype(np.float32))
    np.save(bank / "actions_full.npy", rng.uniform(-1, 1, (sum(ep_len), ACTION_DIM)).astype(np.float32))
    np.save(bank / "row_step.npy", np.array([t for _, t in rows], np.int32))
    np.save(bank / "row_ep.npy", np.array([e for e, _ in rows], np.int32))
    np.save(bank / "blank_latent.npy", rng.standard_normal((LATENT_CH, LATENT_HW, LATENT_HW)).astype(np.float32))
    kd = root / "kd"
    kd.mkdir()
    np.save(kd / "rows.npy", np.arange(len(rows)))
    np.save(kd / "kd_act.npy", rng.uniform(-1, 1, (len(rows), CHUNK, ACTION_DIM)).astype(np.float32))
    np.save(
        kd / "kd_fut.npy",
        rng.standard_normal((len(rows), len(CAMERAS), LATENT_CH, LATENT_HW, LATENT_HW)).astype(np.float16),
    )
    stats = {
        f"{k}_{b}": [float(v)] * d
        for k, d in (("actions", ACTION_DIM), ("proprio", PROPRIO_DIM))
        for b, v in (("min", -1.0), ("max", 1.0))
    }
    (root / "stats.json").write_text(json.dumps(stats))
    feats = torch.randn(2, cfg.max_text_tokens, cfg.text_dim)
    TextCache(meta["instructions"], feats, torch.ones(2, cfg.max_text_tokens, dtype=torch.bool), "test").save(
        root / "text.safetensors"
    )
    adapter = {k: v for k, v in SmolWAMModel(cfg).state_dict().items() if k.startswith("adapter")}
    save_file(adapter, str(root / "adapter.safetensors"))
    return bank


def test_train_export_load(tmp_path, tiny_config):
    torch.manual_seed(0)
    bank = make_bank(tmp_path, tiny_config)
    args = parser().parse_args(
        [
            "--run",
            str(tmp_path / "run"),
            "--bank",
            str(bank),
            "--kd",
            str(tmp_path / "kd"),
            "--text_cache",
            str(tmp_path / "text.safetensors"),
            "--adapter",
            str(tmp_path / "adapter.safetensors"),
            "--stats",
            str(tmp_path / "stats.json"),
            "--init",
            "none",
            "--steps",
            "4",
            "--curriculum_frac",
            "0.5",
            "--bs",
            "2",
            "--warmup",
            "1",
            "--val_every",
            "2",
            "--save_every",
            "2",
            "--workers",
            "0",
            "--device",
            "cpu",
        ]
    )
    best = train(args, tiny_config)
    assert np.isfinite(best)
    log = [json.loads(line) for line in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    assert {r.get("phase") for r in log} == {1, 2}  # curriculum: 2 steps in the teacher layout, then compact

    release = tmp_path / "release"
    export(tmp_path / "run" / "best.pt", tmp_path / "stats.json", tmp_path / "text.safetensors", release)
    config, state_dict, cache = load_release(release)
    model = load_model(SmolWAMConfig.from_dict(config["model"]), state_dict, "cpu", torch.float32)
    ck = torch.load(tmp_path / "run" / "best.pt", weights_only=True)
    assert all(torch.equal(v, ck["model"][k]) for k, v in model.state_dict().items())
    assert cache.instructions == ["pick it up", "put it down"]

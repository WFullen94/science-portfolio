"""Data-logic tests — pure (no download, no torch model). Uses tmp_path fixtures with
tiny fake image directories to validate list_items/stratified_split, which is the
leakage-relevant logic (fair split across real/fake, no accidental overlap).
"""

from __future__ import annotations

from PIL import Image

from dfdetect.data import list_items, stratified_split


def _make_class_dirs(tmp_path, n_real=10, n_fake=6):
    real_dir = tmp_path / "real"
    fake_dir = tmp_path / "fake"
    real_dir.mkdir()
    fake_dir.mkdir()
    for i in range(n_real):
        Image.new("RGB", (8, 8)).save(real_dir / f"r{i}.jpg")
    for i in range(n_fake):
        Image.new("RGB", (8, 8)).save(fake_dir / f"f{i}.jpg")
    return {"real": real_dir, "fake": fake_dir}


def test_list_items_labels_match_class_order(tmp_path):
    dirs = _make_class_dirs(tmp_path, n_real=3, n_fake=2)
    items = list_items(dirs)
    labels = sorted(y for _, y in items)
    assert labels == [0, 0, 0, 1, 1]  # real=0 (3), fake=1 (2)


def test_list_items_respects_max_per_class(tmp_path):
    dirs = _make_class_dirs(tmp_path, n_real=10, n_fake=10)
    items = list_items(dirs, max_per_class=4)
    assert len(items) == 8
    assert sum(1 for _, y in items if y == 0) == 4
    assert sum(1 for _, y in items if y == 1) == 4


def test_stratified_split_no_overlap_and_covers_all(tmp_path):
    dirs = _make_class_dirs(tmp_path, n_real=20, n_fake=20)
    items = list_items(dirs)
    train, val = stratified_split(items, val_fraction=0.2, seed=42)

    train_paths = {p for p, _ in train}
    val_paths = {p for p, _ in val}
    assert train_paths.isdisjoint(val_paths)                 # no leakage
    assert len(train) + len(val) == len(items)                # nothing dropped
    assert len(val) == 8                                       # 20% of 40, stratified


def test_stratified_split_is_stratified_per_class(tmp_path):
    dirs = _make_class_dirs(tmp_path, n_real=20, n_fake=20)
    items = list_items(dirs)
    _, val = stratified_split(items, val_fraction=0.5, seed=1)
    val_real = sum(1 for _, y in val if y == 0)
    val_fake = sum(1 for _, y in val if y == 1)
    assert val_real == val_fake == 10  # balanced classes -> balanced val split


def test_stratified_split_is_deterministic(tmp_path):
    dirs = _make_class_dirs(tmp_path, n_real=10, n_fake=10)
    items = list_items(dirs)
    t1, v1 = stratified_split(items, 0.3, seed=7)
    t2, v2 = stratified_split(items, 0.3, seed=7)
    assert [p for p, _ in t1] == [p for p, _ in t2]
    assert [p for p, _ in v1] == [p for p, _ in v2]

"""N06: scripts/train_detector.py's fine-tune flags, with a fake YOLO (no ultralytics, torch or GPU).

--optimizer, --lr0, --cls-pw, --freeze and --warmup-bias-lr reach model.train() as optimizer, lr0,
cls_pw, freeze and warmup_bias_lr (0.0 with an explicit optimizer, as ultralytics' auto has it);
left out, train() gets exactly the keywords it got before. --resume refuses the flags ultralytics
would drop. A run that would replace weights or move the pin unasked (a best.pt start with the
default --name, a run folder that already holds weights, --pin with trained weights) prints a
WARNING and still trains. Every run trains on the CPU by request (--device cpu) and pins into
tmp_path.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vision import detector as D  # noqa: E402


def _load_train_detector():
    """scripts/train_detector.py as the module `train_detector`, as test_detector.py loads it."""
    if "train_detector" in sys.modules:
        return sys.modules["train_detector"]
    spec = importlib.util.spec_from_file_location("train_detector", ROOT / "scripts" / "train_detector.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_detector"] = module
    spec.loader.exec_module(module)
    return module


td = _load_train_detector()

#: The keywords train_detector passed to train() before N06 (with --epochs 2, --device cpu).
OLD_TRAIN_KEYS = {"data", "epochs", "imgsz", "batch", "device", "workers", "project", "name", "exist_ok",
                  "resume", "patience", "hsv_h", "hsv_s", "hsv_v", "degrees", "shear", "perspective",
                  "flipud", "fliplr", "mosaic", "close_mosaic", "cache", "plots", "val"}


@pytest.fixture
def yolo(monkeypatch):
    """A fake ultralytics.YOLO: train(**kw) keeps kw and writes <project>/<name>/weights/best.pt
    ("best <name>") and last.pt into the folder, existing or not (exist_ok=True reuses it)."""
    class FakeYOLO:
        instances: list = []

        def __init__(self, model):
            self.model, self.train_kwargs, self.trainer = model, None, None
            FakeYOLO.instances.append(self)

        def train(self, **kw):
            self.train_kwargs = kw
            wdir = Path(kw["project"]) / kw["name"] / "weights"
            wdir.mkdir(parents=True, exist_ok=True)
            (wdir / "best.pt").write_bytes(b"best " + kw["name"].encode())
            (wdir / "last.pt").write_bytes(b"last")
            self.trainer = SimpleNamespace(best=str(wdir / "best.pt"))

    monkeypatch.setattr(td, "_import_torch", lambda: None)
    monkeypatch.setattr(td, "_import_yolo", lambda: FakeYOLO)
    return FakeYOLO


def run(tmp_path, *flags, name: str | None = "t") -> int:
    data = tmp_path / "cr.yaml"
    data.write_text("names:\n  0: a@ally\n  1: a@enemy\n", encoding="utf-8")
    argv = ["--data", str(data), "--project", str(tmp_path / "runs"), "--device", "cpu", "--epochs", "2",
            "--pin-file", str(tmp_path / "pin.json")]
    if name is not None:
        argv += ["--name", name]
    return td.main(argv + list(flags))


def manifest(tmp_path, name: str = "t") -> dict:
    return json.loads((tmp_path / "runs" / name / "detector_manifest.json").read_text(encoding="utf-8"))


def test_without_the_flags_train_gets_the_old_keywords(tmp_path, yolo, capsys):
    assert run(tmp_path) == 0
    kw = yolo.instances[-1].train_kwargs
    assert set(kw) == OLD_TRAIN_KEYS
    assert (kw["data"], kw["project"], kw["name"], kw["device"]) == (
        str(tmp_path / "cr.yaml"), str(tmp_path / "runs"), "t", "cpu")
    assert (kw["epochs"], kw["imgsz"], kw["batch"], kw["workers"], kw["patience"]) == (2, 640, 32, 4, 30)
    assert (kw["exist_ok"], kw["resume"], kw["cache"], kw["plots"], kw["val"]) == (True, False, False, True, True)
    assert (kw["hsv_h"], kw["hsv_s"], kw["hsv_v"], kw["degrees"], kw["shear"], kw["perspective"]) == (
        0.015, 0.7, 0.4, 0.0, 0.0, 0.0)
    assert (kw["flipud"], kw["fliplr"], kw["mosaic"], kw["close_mosaic"]) == (0.0, 0.5, 1.0, 10)
    assert "train_overrides" not in manifest(tmp_path)
    out = capsys.readouterr().out
    assert "train settings" not in out and "note:" not in out and "WARNING" not in out


def test_fine_tune_flags_reach_model_train(tmp_path, yolo, capsys):
    assert run(tmp_path, "--optimizer", "adamw", "--lr0", "0.001", "--cls-pw", "0.5", "--freeze", "10") == 0
    kw = yolo.instances[-1].train_kwargs
    assert set(kw) == OLD_TRAIN_KEYS | {"optimizer", "lr0", "cls_pw", "freeze", "warmup_bias_lr"}
    assert (kw["optimizer"], kw["lr0"], kw["cls_pw"], kw["freeze"]) == ("AdamW", 0.001, 0.5, 10)
    assert kw["warmup_bias_lr"] == 0.0                # as auto: ultralytics' 0.1 is 100x this lr0
    assert (kw["epochs"], kw["batch"], kw["fliplr"]) == (2, 32, 0.5)          # the rest as before
    assert manifest(tmp_path)["train_overrides"] == {"optimizer": "AdamW", "lr0": 0.001, "cls_pw": 0.5,
                                                     "freeze": 10, "warmup_bias_lr": 0.0}
    out = capsys.readouterr().out
    assert "train settings: optimizer AdamW, lr0 0.001, cls_pw 0.5, freeze 10, warmup_bias_lr 0.0" in out
    assert "note: warmup_bias_lr 0, as optimizer auto starts it" in out
    assert "ignores --lr0" not in out and "WARNING" not in out

    assert run(tmp_path, "--optimizer", "musgd", "--freeze", "0,1,22.cv2", name="t2") == 0
    kw = yolo.instances[-1].train_kwargs
    assert (kw["optimizer"], kw["freeze"], kw["warmup_bias_lr"]) == ("MuSGD", [0, 1, "22.cv2"], 0.0)
    assert "lr0" not in kw and "cls_pw" not in kw
    assert manifest(tmp_path, "t2")["train_overrides"] == {"optimizer": "MuSGD", "freeze": [0, 1, "22.cv2"],
                                                           "warmup_bias_lr": 0.0}
    # PowerShell passes an unquoted 0,1,22.cv2 as three arguments: the same freeze list
    assert run(tmp_path, "--freeze", "0", "1", "22.cv2", "--epochs", "2", name="t3") == 0
    kw = yolo.instances[-1].train_kwargs
    assert kw["freeze"] == [0, 1, "22.cv2"] and "warmup_bias_lr" not in kw and "optimizer" not in kw


def test_warmup_bias_lr_is_passed_as_given(tmp_path, yolo, capsys):
    assert run(tmp_path, "--optimizer", "SGD", "--lr0", "0.01", "--warmup-bias-lr", "0.1") == 0
    kw = yolo.instances[-1].train_kwargs
    assert (kw["optimizer"], kw["lr0"], kw["warmup_bias_lr"]) == ("SGD", 0.01, 0.1)    # ultralytics' own
    out = capsys.readouterr().out
    assert "warmup_bias_lr 0.1" in out and "note:" not in out
    assert run(tmp_path, "--warmup-bias-lr", "0.05", name="t2") == 0                 # no optimizer: auto
    kw = yolo.instances[-1].train_kwargs
    assert kw["warmup_bias_lr"] == 0.05 and "optimizer" not in kw
    assert "ultralytics resets warmup_bias_lr to 0 with optimizer auto" in capsys.readouterr().out
    assert run(tmp_path, "--optimizer", "auto", name="t3") == 0                       # auto: nothing added
    assert "warmup_bias_lr" not in yolo.instances[-1].train_kwargs


@pytest.mark.parametrize("flags", [["--optimizer", "AdamW"], ["--lr0", "0.001"], ["--cls-pw", "0.5"],
                                   ["--warmup-bias-lr", "0"]])
def test_resume_refuses_what_ultralytics_would_drop(tmp_path, yolo, capsys, flags):
    with pytest.raises(SystemExit) as exc:
        run(tmp_path, "--resume", *flags)
    assert exc.value.code == 2 and yolo.instances == []
    assert "--resume continues the checkpoint's own settings" in capsys.readouterr().err


def test_resume_keeps_freeze(tmp_path, yolo):
    assert run(tmp_path, "--resume", "--freeze", "10") == 0                 # check_resume re-applies it
    kw = yolo.instances[-1].train_kwargs
    assert (kw["resume"], kw["freeze"]) == (True, 10) and "warmup_bias_lr" not in kw


def test_lr0_without_an_optimizer_says_auto_ignores_it(tmp_path, yolo, capsys):
    assert run(tmp_path, "--lr0", "0.002") == 0
    kw = yolo.instances[-1].train_kwargs
    assert kw["lr0"] == 0.002 and "optimizer" not in kw
    assert "ultralytics ignores --lr0 with optimizer auto" in capsys.readouterr().out
    assert run(tmp_path, "--optimizer", "AUTO", "--lr0", "0.002", name="t2") == 0
    assert yolo.instances[-1].train_kwargs["optimizer"] == "auto"
    assert "ultralytics ignores --lr0 with optimizer auto" in capsys.readouterr().out


@pytest.mark.parametrize("flags", [["--optimizer", "lion"], ["--lr0", "0"], ["--lr0", "1.5"], ["--lr0", "fast"],
                                   ["--lr0", "nan"], ["--cls-pw", "-0.1"], ["--cls-pw", "2"],
                                   ["--freeze", "-1"], ["--freeze", "1,,2"], ["--freeze", ""],
                                   ["--freeze", "1", "-2"], ["--warmup-bias-lr", "1.5"]])
def test_values_ultralytics_would_refuse_are_refused_first(tmp_path, yolo, flags):
    with pytest.raises(SystemExit) as exc:
        run(tmp_path, *flags)
    assert exc.value.code == 2 and yolo.instances == []


def test_argument_parsers():
    names = [td.optimizer_name(x) for x in ("rmsprop", " SGD ", "Adamax", "auto")]
    assert names == ["RMSprop", "SGD", "Adamax", "auto"]
    assert td.freeze_spec("3") == 3 and td.freeze_spec("22.cv2") == ["22.cv2"] and td.freeze_spec("0, 1") == [0, 1]
    assert td.learning_rate("1") == 1.0 and td.class_weight_power("0") == 0.0
    assert td.bias_warmup_rate("0") == 0.0 and td.bias_warmup_rate("0.1") == 0.1
    stock = ("yolov8s.pt", "yolo11m.pt", "yolov8s.yaml", "weights/yolov8n.pt", "YOLOv8S.PT")
    trained = ("best.pt", "tmp/yolo/runs/cr_detector/weights/last.pt", "yolov8s-m0.pt", "m0_yolov8s.pt")
    assert all(td.is_stock(m) for m in stock) and not any(td.is_stock(m) for m in trained)


def test_fine_tune_from_best_pt_with_the_default_name_warns(tmp_path, yolo, capsys):
    old = tmp_path / "runs" / td.DEFAULT_NAME / "weights" / "best.pt"      # tonight's model, pinned
    old.parent.mkdir(parents=True)
    old.write_bytes(b"M0")
    D.write_pin(old, path=tmp_path / "pin.json")
    assert run(tmp_path, "--model", str(old), name=None) == 0
    out = capsys.readouterr().out
    assert f"WARNING: fine-tuning from {old} with the default --name {td.DEFAULT_NAME}" in out
    assert "The starting weights are in that folder" in out
    assert "The pinned detector" in out and "the pin moves to the new best.pt" in out
    assert out.index("WARNING") < out.index("weights and curves under")      # said before training
    assert yolo.instances[-1].train_kwargs["name"] == td.DEFAULT_NAME        # a warning, not a refusal
    assert old.read_bytes() == b"best cr_detector"                           # ... and why it warns
    assert D.read_pin(tmp_path / "pin.json")["sha256"] == hashlib.sha256(b"best cr_detector").hexdigest()


def test_no_warning_for_a_new_name_a_resume_or_stock_weights(tmp_path, yolo, capsys):
    old = tmp_path / "elsewhere" / "best.pt"
    old.parent.mkdir()
    old.write_bytes(b"M0")
    for i, (flags, name) in enumerate(((["--model", str(old), "--no-pin"], "ft_20261101"),   # a fine-tune
                                       (["--model", str(old), "--resume"], None),     # continues its own run
                                       ([], None),                                    # yolov8s.pt, tonight's run
                                       (["--model", "yolo11m.pt"], None))):
        root = tmp_path / f"case{i}"                                       # a fresh --project each
        root.mkdir()
        assert run(root, *flags, name=name) == 0
        assert "WARNING" not in capsys.readouterr().out, flags


def test_a_rebase_over_the_pinned_baseline_warns(tmp_path, yolo, capsys):
    m0 = tmp_path / "runs" / td.DEFAULT_NAME / "weights" / "best.pt"      # tonight's M0, the gate baseline
    m0.parent.mkdir(parents=True)
    m0.write_bytes(b"M0")
    (m0.parent / "last.pt").write_bytes(b"M0 last")
    D.write_pin(m0, path=tmp_path / "pin.json")
    assert run(tmp_path, "--no-pin", name=None) == 0                      # from yolov8s.pt: stock weights
    out = capsys.readouterr().out
    assert f"WARNING: {tmp_path / 'runs' / td.DEFAULT_NAME} already holds weights/best.pt and last.pt" in out
    assert "The pinned detector" in out and "(--no-pin: the pin stays)" in out
    assert "The starting weights" not in out and "pin moves" not in out and "Pass a new --name" in out
    assert out.index("WARNING") < out.index("weights and curves under")
    assert m0.read_bytes() == b"best cr_detector"                         # replaced, as it warned


def test_a_fine_tune_rerun_under_its_own_pinned_name_warns(tmp_path, yolo, capsys):
    ft = tmp_path / "runs" / "ft_1" / "weights" / "best.pt"               # pinned after its gate PASS
    ft.parent.mkdir(parents=True)
    ft.write_bytes(b"FT1")
    D.write_pin(ft, path=tmp_path / "pin.json")
    old = tmp_path / "elsewhere" / "best.pt"
    old.parent.mkdir()
    old.write_bytes(b"M0")
    assert run(tmp_path, "--model", str(old), "--no-pin", name="ft_1") == 0
    out = capsys.readouterr().out
    assert "already holds weights/best.pt:" in out and "The pinned detector" in out
    assert "(--no-pin: the pin stays)" in out and ft.read_bytes() == b"best ft_1"


def test_a_fine_tune_that_keeps_the_pin_on_warns(tmp_path, yolo, capsys):
    old = tmp_path / "elsewhere" / "best.pt"
    old.parent.mkdir()
    old.write_bytes(b"M0")
    assert run(tmp_path, "--model", str(old), name="ft_2") == 0           # forgot --no-pin
    out = capsys.readouterr().out
    assert f"WARNING: training from {old}, not stock weights, with the pin on." in out
    assert "When training ends the pin moves to the new best.pt" in out and "Pass a new --name" not in out
    assert D.read_pin(tmp_path / "pin.json")["sha256"] == hashlib.sha256(b"best ft_2").hexdigest()


def test_fine_tune_warnings_reads_the_pin_only_when_it_warns(tmp_path):
    base = dict(model="elsewhere/best.pt", name=td.DEFAULT_NAME, project=tmp_path / "runs", resume=False,
                pin=False, pin_file=None)
    lines = td.fine_tune_warnings(SimpleNamespace(**base), read_pin=lambda pin_file: None)
    assert len(lines) == 2 and lines[0].startswith("WARNING") and "--name" in lines[1]
    assert not any("pin moves" in line or "starting weights" in line for line in lines)   # --no-pin, elsewhere
    never = lambda pin_file: pytest.fail("read the pin")                    # noqa: E731
    for change in ({"name": "ft1"}, {"resume": True}, {"model": "yolov8n.pt"}, {"model": "yolov8s.yaml"}):
        assert td.fine_tune_warnings(SimpleNamespace(**{**base, **change}), read_pin=never) == [], change

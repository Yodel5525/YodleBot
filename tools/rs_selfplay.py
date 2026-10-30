"""Self-play reinforcement learning in RoyaleSim, starting from imitation.

Everything up to now copied pros. This lets the Hog model learn from its own
results: it plays full RoyaleSim battles under the live configuration (21-tick
command delay, decisions on a board rebuilt from public information and run
21 ticks ahead) against a frozen reactive opponent playing the real
opponents' decks from IL_Replay's train split, and is pushed toward the
decisions of the battles it won.

Two halves, because the engine and the GPU live in different environments:

  collect  (.venvs/royalegym, CPU, engine)  - N battles -> one .npz of decisions
  train    (.venvs/buildabot, CUDA)         - one PPO update on that .npz

and a driver that alternates them and evaluates every few iterations:

  .venvs\\royalegym\\Scripts\\python.exe tools\\rs_selfplay.py loop --iters 40

The policy being trained is exactly the one the live bot runs: the gate's
wait / play probabilities with the evaluator's wait correction and its
p_wait ** (DECIDE / 20) cadence, a slot among the affordable cards, and a tile
among the legal ones. Here the slot and tile are SAMPLED (the evaluator takes
the argmax), so the policy can find out what else works; that is what the
log-probabilities below describe.

Update: PPO (clipped ratio) on advantage = result - value, where result is
1 / 0.5 / 0 for win / draw / loss and value is the model's own win head at that
moment, plus a KL penalty to the imitation model so it cannot wander off into
play no pro would recognise. The value head is trained on the results too.

Engine (tools/rs_engine.py): every battle runs under data/engine.json and the HASTYCR_*
variables, or under the --card-table / --engine-config given to collect or loop; loop hands
them on to collect and to the evaluation. collect resolves the config once and hands it to
every worker (the pool initializer rs_engine.set_active, and the engine slot of each match
task, as rs_eval does), then writes it with the engine's fingerprint into its .npz (ENGINE_KEY).
train copies that record into the checkpoint it writes (STAMP_KEY), and loop into the run's
args.json ("engine", "engine_fingerprint", "engine_id"). A run continued under another engine
is refused: start a new --name.

Visibility (E12): the sampling agent decides on, and records, the rows its checkpoint's rows
showed (meta "visibility": legacy before E12, else public), as RSAgent does in rs_eval. Only
checkpoints of the legacy row schema (meta "schema" rs_rows/legacy, or none) are played and
trained here: KEYS and the update below are that layout's.

Nothing here touches the emulator or the game.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "src")]

RUNS = ROOT / "runs" / "selfplay"
KEYS = ("unit_ids", "unit_feats", "towers", "scalars", "hand", "opp_seen", "opp_last", "opp_last_age")
#: The live configuration every self-play and evaluation match runs under: the client's command
#: delay, and decisions on a board rebuilt from public information 21 ticks ahead (rs_eval --lead).
DELAY, LEAD = 21, -21
#: The .npz entry collect writes: the engine record of its rollouts, as JSON text (engine_record).
ENGINE_KEY = "engine_json"
#: The checkpoint key train writes that record under (next to "model" and "meta").
STAMP_KEY = "selfplay"
#: What the record holds, and what args.json carries of it.
STAMP_FIELDS = ("engine", "engine_fingerprint", "engine_id")
#: The row layout KEYS and the update read (rs_rows.SCHEMA before E14's v2).
LEGACY_SCHEMA = "rs_rows/legacy"
DECIDE = 10
TOP_TILES = 8          # tiles sampled among the model's best legal ones
EXTRA_TILES = 8        # plus this many random legal tiles from anywhere
EXPLORE = 0.15         # share of the tile choice spread evenly over all candidates
TILE_TEMP = 1.0
# Why the extra tiles: sampling only the model's 8 best legal tiles never once
# tried a pocket Hog after a princess tower fell - the bridge row alone holds
# 16 legal tiles, so all 8 were bridge tiles - and what is never tried cannot
# be learned (2026-10-10).
CANDS = TOP_TILES + EXTRA_TILES


# ----------------------------------------------------------------------------
# collect (royalegym env)
# ----------------------------------------------------------------------------

def _sampling_agent():
    import rs_eval
    import rs_rows as RR

    class SPAgent(rs_eval.RSAgent):
        """RSAgent.act with a sampled slot and tile, recording every decision. The rows see
        what the checkpoint's rows saw (RSAgent reads `visibility` from its meta; rs_eval.play
        sets a --visibility over it)."""

        def __init__(self, ckpt, seed=0, wait_keep=0.35):
            super().__init__(ckpt, seed=seed, wait_keep=wait_keep)
            check_schema(self.meta, ckpt)
            self.records = []

        def act(self, battle, state, team, mem, tick, sub, royalesim):
            torch = self.torch
            self._team = team
            view = self.ahead(battle, state)
            row = RR.obs_row(view, team, sub, mem, tick + self.lead, live=self.live,
                             visibility=self.visibility)
            b = self.batch(row)
            with torch.no_grad():
                gate, _, value = self.model(b, card_ids=b["hand"][:, 0].clamp(min=0))
            g = gate[0].float().numpy().astype(np.float64)
            g[0] += self.wait_bias
            p = state["players"][team]
            pending_cost = int(p.get("pending_cost", 0) or 0)
            mask = np.ones(5, bool)
            for slot in range(4):
                cost = p["hand_costs"][slot] if slot < len(p["hand_costs"]) else 99
                if (slot >= len(p["hand"]) or p["elixir_milli"] - pending_cost * 1000 < cost * 1000
                        or p["hand"][slot] < 0):
                    mask[1 + slot] = False
            if not mask[1:].any():
                return None
            q = gate_probs(g, mask)
            a = int(self.rng.choice(5, p=q))
            rec = {k: np.asarray(row[k]) for k in KEYS}
            rec.update(next=int(row["next"]), mask=mask, gate_a=a, logq_gate=float(math.log(q[a])),
                       v=float(1 / (1 + math.exp(-float(value[0, 1])))), tile_a=-1, logq_tile=0.0,
                       tile_cands=np.full(CANDS, -1, np.int16), tick=tick)
            if a == 0:
                self.records.append(rec)
                return None
            slot = a - 1
            with torch.no_grad():
                _, tile, _ = self.model(b, card_ids=b["hand"][:, slot])
            logits = tile[0].float().numpy().astype(np.float64)

            def legal_at(t):
                tx, ty = int(t % 18), int(t // 18)
                wx, wy = rs_eval.to_world(tx * 1000 + 500, ty * 1000 + 500, team)
                ok = battle.check_deploy(team, slot, wx * sub, wy * sub) == 0
                return ok, wx, wy

            legal, seen = [], set()
            for t in np.argsort(-logits)[:60]:
                ok, wx, wy = legal_at(t)
                seen.add(int(t))
                if ok:
                    legal.append((int(t), wx, wy))
                    if len(legal) == TOP_TILES:
                        break
            if not legal:
                return None
            for t in self.rng.permutation(len(logits))[:120]:
                if len(legal) == CANDS:
                    break
                if int(t) in seen:
                    continue
                seen.add(int(t))
                ok, wx, wy = legal_at(t)
                if ok:
                    legal.append((int(t), wx, wy))
            pt = tile_probs(np.array([logits[t] for t, _, _ in legal]))
            k = int(self.rng.choice(len(legal), p=pt))
            t, wx, wy = legal[k]
            cands = np.full(CANDS, -1, np.int16)
            cands[:len(legal)] = [c for c, _, _ in legal]
            rec.update(tile_a=k, logq_tile=float(math.log(pt[k])), tile_cands=cands)
            self.records.append(rec)
            return (team, slot, wx * sub, wy * sub)

    return SPAgent


def check_schema(meta, source="checkpoint"):
    """ValueError unless `meta` (a checkpoint's) is of the legacy row schema, LEGACY_SCHEMA (or
    says none, as every checkpoint before E14): the rows recorded here (KEYS) and the update in
    train are that layout's, and another one would be misread silently."""
    schema = (meta or {}).get("schema") or LEGACY_SCHEMA
    if schema not in (LEGACY_SCHEMA, "legacy"):
        raise ValueError(f"{source}: a checkpoint of row schema {schema!r}; tools/rs_selfplay.py "
                         f"plays and trains {LEGACY_SCHEMA!r} checkpoints only")


# ----------------------------------------------------------------------------
# the engine (tools/rs_engine.py)
# ----------------------------------------------------------------------------

def engine_config(args):
    """The rs_engine.EngineConfig of --card-table / --engine-config (rs_engine.config_from_args;
    data/engine.json and the HASTYCR_* variables when neither is given). A bad setting or a
    missing file exits with status 2."""
    import rs_engine
    try:
        return rs_engine.config_from_args(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"engine config: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from None


def engine_argv(args):
    """--engine-config / --card-table as this command was given them, for its subprocesses."""
    out = []
    if getattr(args, "engine_config", None) is not None:
        out += ["--engine-config", str(args.engine_config)]
    if getattr(args, "engine_card_table", None) is not None:
        out += ["--card-table", str(args.engine_card_table)]
    return out


def engine_record(cfg, rs=None):
    """What the rollouts run on: {"engine": `cfg` with the match delay (to_json),
    "engine_fingerprint": rs_engine.fingerprint of an undelayed Battle under `cfg` (as rs_eval
    takes it), "engine_id": its id}. `rs`: the royalesim module (None: import it)."""
    import rs_engine
    rs = rs_engine.royalesim_module(rs)
    fp = rs_engine.fingerprint(rs_engine.make_battle(cfg.undelayed(), rs), rs, cfg)
    return {"engine": cfg.delayed(DELAY).to_json(), "engine_fingerprint": fp, "engine_id": fp["id"]}


def read_engine_record(data):
    """The engine record a collect wrote (ENGINE_KEY) into its .npz: `data` is the np.load of it
    or its path. None for an .npz without one (written before it was recorded) or a missing file."""
    if isinstance(data, (str, os.PathLike)):
        try:
            with np.load(data) as d:
                return read_engine_record(d)
        except OSError:
            return None
    if ENGINE_KEY not in data.files:
        return None
    return json.loads(str(data[ENGINE_KEY][()]))


def run_engine(args_json):
    """The engine record (STAMP_FIELDS) a run's args.json holds once a collect reported one; {}
    for a new run, or one from before the record."""
    try:
        known = json.loads(Path(args_json).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(known, dict) or not known.get("engine_id"):
        return {}
    return {k: known.get(k) for k in STAMP_FIELDS}


def tile_probs(lg):
    """Tile choice among the candidates: (1 - EXPLORE) x the model's softmax
    + EXPLORE spread evenly, so every candidate keeps a real chance."""
    p = np.exp((lg - lg.max()) / TILE_TEMP)
    p /= p.sum()
    return (1 - EXPLORE) * p + EXPLORE / len(lg)


def gate_probs(g, mask):
    """The live / evaluator gate as a distribution over wait + 4 slots.

    Evaluator: wait with probability p_wait ** (DECIDE / 20), where p_wait is
    the softmax wait share; otherwise play. Here the slot played is drawn in
    proportion to the slot softmax, among affordable slots only.
    """
    g = np.where(mask, g, -1e9)
    pr = np.exp(g - g.max())
    pr /= pr.sum()
    qw = float(pr[0]) ** (DECIDE / 20)
    s = pr[1:].sum()
    q = np.zeros(5)
    q[0] = qw
    q[1:] = (1 - qw) * pr[1:] / max(s, 1e-12)
    q = np.where(mask, q, 0.0)
    return q / q.sum()


def _collect_one(task):
    import rs_eval
    rs_eval.AGENT_CLASS = _sampling_agent()
    out = rs_eval.play(task)
    if not out or not out.get("records"):
        return None
    r = {"win": 1.0, "draw": 0.5, "loss": 0.0}[out["result"]]
    return out["result"], out["crowns"], r, out["records"]


def pick_matches(n, seed, split="train", engine=None):
    """Hog 2.6 matches from IL_Replay (train split by default), shuffled by seed. Card names
    from a Battle under `engine` (an rs_engine config; None: rs_engine.active())."""
    import random
    import pyarrow.parquet as pq
    import rs_rows as RR
    from royalesim_redrive import CORPUS, base_key
    _, names, _, _ = RR._setup(engine)
    parts = sorted((CORPUS / "replays").glob("part-*.parquet"))
    rnd = random.Random(seed)
    rnd.shuffle(parts)
    picked = []
    for part in parts:
        rows = pq.read_table(part, columns=["replay_tag", "game_mode", "payload_json"]).to_pylist()
        rnd.shuffle(rows)
        for r in rows:
            if r["game_mode"] not in RR.MODES or RR.split_of(r["replay_tag"]) != split:
                continue
            p = json.loads(r["payload_json"])
            try:
                sides = [frozenset(names.get(base_key(c["card_key"])[0])
                                   for c in p["battle"][s]["players"][0]["deck"]) for s in ("team", "opponent")]
            except (KeyError, IndexError):
                continue
            if any(None in s for s in sides) or RR.HOG26 not in sides or sides[0] == sides[1]:
                continue
            p.setdefault("source", {})["replay_tag"] = r["replay_tag"]
            picked.append(json.dumps(p))
            if len(picked) >= n:
                return picked
    return picked


def collect(args):
    """--matches battles -> one .npz of decisions, with the engine record (ENGINE_KEY). The
    engine config is resolved here once and handed to every worker: as the pool initializer
    (rs_engine.set_active, as rs_eval does) and in each task's engine slot (rs_eval.play
    args[10]). The tasks are otherwise rs_eval.play's 10-tuple as before: memory lag, decision
    rule and visibility are play's defaults (the checkpoint's visibility)."""
    import rs_engine
    cfg = engine_config(args)
    rs_engine.set_active(cfg)
    engine = engine_record(cfg)
    picked = pick_matches(args.matches, args.seed, engine=cfg)
    tasks = [(t, args.ckpt, "agent", args.seed * 1000 + k, args.opponent, DELAY, LEAD, False, False,
              False, cfg.to_json())
             for k, t in enumerate(picked)]
    t0 = time.time()
    recs, results = [], []
    with ProcessPoolExecutor(args.workers, initializer=rs_engine.set_active,
                             initargs=(cfg.to_json(),)) as pool:
        for res in pool.map(_collect_one, tasks):
            if res is None:
                continue
            outcome, crowns, r, rows = res
            results.append(outcome)
            for row in rows:
                row["ret"] = r
                recs.append(row)
    out = {k: np.stack([x[k] for x in recs]) for k in recs[0]}
    out[ENGINE_KEY] = np.array(json.dumps(engine, sort_keys=True))
    np.savez_compressed(args.out, **out)
    w = results.count("win") / max(1, len(results))
    print(json.dumps({"matches": len(results), "decisions": len(recs), "win_rate": round(w, 3),
                      "seconds": round(time.time() - t0), "engine_id": engine["engine_id"]}),
          flush=True)


# ----------------------------------------------------------------------------
# train (buildabot env, CUDA)
# ----------------------------------------------------------------------------

def train(args):
    import torch
    import torch.nn.functional as F
    from sim.model_tokens import TokenPolicy
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def load(path):
        blob = torch.load(path, map_location="cpu")
        m = blob["meta"]
        check_schema(m, path)
        net = TokenPolicy(m["n_cards"], m["n_units"], d=m.get("d", 192),
                          layers=m.get("layers", 4), heads=m.get("heads", 6))
        net.load_state_dict(blob["model"])
        return net.to(dev), blob

    model, blob = load(args.ckpt)
    # No dropout: the probabilities trained on must be the ones the agent
    # sampled from (with dropout on, ratios were off by 4% at learning rate 0).
    model.eval()
    ref, _ = load(args.ref)
    ref.eval()
    wait_bias = math.log(1 / float(args.wait_keep))
    d = np.load(args.data)
    engine = read_engine_record(d)          # what the rollouts ran on; stamped below
    n = len(d["gate_a"])
    adv_all = d["ret"] - d["v"]
    adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-6)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    stats = []
    for epoch in range(args.epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            b = {k: torch.as_tensor(d[k][idx]).to(dev) for k in KEYS}
            for k in ("unit_ids", "hand", "opp_seen", "opp_last"):
                b[k] = b[k].long()
            b["unit_feats"] = b["unit_feats"].float()
            b["opp_last_age"] = b["opp_last_age"].float()
            b["next"] = torch.as_tensor(d["next"][idx]).long().to(dev)
            mask = torch.as_tensor(d["mask"][idx]).to(dev)
            a = torch.as_tensor(d["gate_a"][idx]).long().to(dev)
            old = torch.as_tensor(d["logq_gate"][idx] + d["logq_tile"][idx]).float().to(dev)
            adv = torch.as_tensor(adv_all[idx]).float().to(dev)
            ret = torch.as_tensor(d["ret"][idx]).float().to(dev)
            hand = b["hand"]
            slot = (a - 1).clamp(min=0)
            card = hand.gather(1, slot[:, None])[:, 0].clamp(min=0)
            gate, tile, value = model(b, card_ids=card)
            with torch.no_grad():
                rgate, rtile, _ = ref(b, card_ids=card)

            def gate_logq(gl):
                gl = gl.float().clone()
                gl[:, 0] = gl[:, 0] + wait_bias
                gl = gl.masked_fill(~mask, -1e9)
                lp = F.log_softmax(gl, 1)
                lw = lp[:, 0] * (DECIDE / 20)                      # log q_wait = log p_wait ** 0.5
                q_play = (1 - lw.exp()).clamp(min=1e-6)
                ls = F.log_softmax(gl[:, 1:], 1)                   # slot share among playable
                lq = torch.cat([lw[:, None], q_play.log()[:, None] + ls], 1)
                return lq

            lq = gate_logq(gate)
            lq_ref = gate_logq(rgate)
            new = lq.gather(1, a[:, None])[:, 0]
            # tile: log-softmax over the recorded legal candidates
            cands = torch.as_tensor(d["tile_cands"][idx]).long().to(dev)
            played = a > 0
            cmask = cands >= 0
            tl = tile.float().gather(1, cands.clamp(min=0)) / TILE_TEMP
            tl = tl.masked_fill(~cmask, -1e9)
            ncand = cmask.float().sum(1, keepdim=True).clamp(min=1)
            ptile = (1 - EXPLORE) * F.softmax(tl, 1) + EXPLORE / ncand
            ltile = ptile.clamp(min=1e-12).log().masked_fill(~cmask, -1e9)
            ta = torch.as_tensor(d["tile_a"][idx]).long().to(dev).clamp(min=0)
            new = new + torch.where(played, ltile.gather(1, ta[:, None])[:, 0], torch.zeros_like(new))
            ratio = (new - old).exp()
            pg = -torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv).mean()
            # keep near the imitation policy (gate distribution, and tile on plays)
            kl_gate = (lq.exp() * (lq - lq_ref)).sum(1).mean()
            rtl = rtile.float().gather(1, cands.clamp(min=0)).masked_fill(~cmask, -1e9)
            # the reference under the same exploration mixture, or the brake
            # would push against the exploration itself
            pref = (1 - EXPLORE) * F.softmax(rtl / TILE_TEMP, 1) + EXPLORE / ncand
            lref_tile = pref.clamp(min=1e-12).log().masked_fill(~cmask, -1e9)
            kl_tile = ((ltile.exp() * (ltile - lref_tile)).masked_fill(~cmask, 0).sum(1)
                       * played.float()).sum() / played.float().sum().clamp(min=1)
            v_loss = F.binary_cross_entropy_with_logits(value[:, 1].float(), ret)
            loss = pg + args.kl * (kl_gate + kl_tile) + 0.5 * v_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            stats.append((pg.item(), kl_gate.item(), kl_tile.item(), v_loss.item(),
                          (ratio - 1).abs().mean().item()))
    s = np.mean(stats, 0)
    blob["model"] = {k: v.cpu() for k, v in model.state_dict().items()}
    # The engine these weights last learned in (builtins only, so weights_only loads keep
    # working); an older .npz has none, and the stamp --ckpt carried is not this update's.
    blob.pop(STAMP_KEY, None)
    if engine is not None:
        blob[STAMP_KEY] = engine
    torch.save(blob, args.out)
    print(json.dumps({"rows": n, "pg": round(s[0], 4), "kl_gate": round(s[1], 4),
                      "kl_tile": round(s[2], 4), "value_bce": round(s[3], 4),
                      "ratio_dev": round(s[4], 4),
                      "engine_id": None if engine is None else engine.get("engine_id")}), flush=True)


# ----------------------------------------------------------------------------
# loop (royalegym env drives both)
# ----------------------------------------------------------------------------

def loop(args):
    """Collect, update, evaluate; resumes from the run's newest it*.pt. The engine flags go to
    collect and to the evaluation; args.json records the engine (STAMP_FIELDS) once the first
    collect reports it, and a launch or a collect under another engine is refused."""
    run = RUNS / args.name
    run.mkdir(parents=True, exist_ok=True)
    args_json = run / "args.json"
    cfg = engine_config(args)
    flags = engine_argv(args)
    want = cfg.delayed(DELAY).to_json()
    prior = run_engine(args_json)           # what an earlier launch of this run collected under
    if prior and prior["engine"] != want:
        print(f"error: run {args.name!r} was collected under engine {json.dumps(prior['engine'])} "
              f"(id {prior['engine_id']}); this launch resolves {json.dumps(want)}. Give it the "
              "same --card-table / --engine-config (and HASTYCR_* settings), or start a new "
              "--name.", file=sys.stderr, flush=True)
        return 1
    record = {**vars(args), "engine": want, **prior}
    args_json.write_text(json.dumps(record, default=str))
    print(f"engine: {cfg.delayed(DELAY).describe()}", flush=True)
    # Interpreters: HASTYCR_PY_ENGINE / HASTYCR_PY_TRAIN, machine.json, or .venvs/ (hastycr.machine).
    from hastycr import machine
    rg, bb = machine.py_engine(), machine.py_train()
    me = str(Path(__file__).resolve())
    # args.json next to each checkpoint so RSAgent reads the right wait_keep
    init_args = Path(args.ref or args.init).with_name("args.json")
    wait_keep = json.loads(init_args.read_text())["wait_keep"] if init_args.exists() else "0.25"
    ckpt = run / "it000.pt"
    if not ckpt.exists():
        import shutil
        shutil.copy(args.init, ckpt)
    record["wait_keep"] = wait_keep
    args_json.write_text(json.dumps(record, default=str))
    log = run / "log.jsonl"
    start = max([int(p.stem[2:]) for p in run.glob("it*.pt")])
    for it in range(start, args.iters):
        ckpt = run / f"it{it:03d}.pt"
        data = run / f"it{it:03d}.npz"
        t0 = time.time()
        c = subprocess.run([str(rg), me, "collect", "--ckpt", str(ckpt), "--opponent", args.opponent,
                            "--matches", str(args.matches), "--workers", str(args.workers),
                            "--seed", str(it + 1), "--out", str(data), *flags],
                           capture_output=True, text=True)
        if c.returncode != 0:
            print(c.stderr[-2000:], flush=True)
            return 1
        col = json.loads(c.stdout.strip().splitlines()[-1])
        engine = read_engine_record(data)
        if engine is not None:
            # The collect's config must be the one this launch resolved (record["engine"]), also
            # on the run's first collect; once stamped, its engine id too.
            if engine.get("engine") != record["engine"] or (
                    record.get("engine_id") and engine.get("engine_id") != record["engine_id"]):
                print(f"error: iteration {it} was collected under engine id "
                      f"{engine.get('engine_id')} {json.dumps(engine.get('engine'))}, run "
                      f"{args.name!r} under {record.get('engine_id') or '(not stamped yet)'} "
                      f"{json.dumps(record['engine'])}; start a new --name to change engines.",
                      file=sys.stderr, flush=True)
                return 1
            if not record.get("engine_id"):
                record.update((k, engine.get(k)) for k in STAMP_FIELDS)
                args_json.write_text(json.dumps(record, default=str))
        nxt = run / f"it{it + 1:03d}.pt"
        t = subprocess.run([str(bb), me, "train", "--ckpt", str(ckpt), "--ref", args.ref or args.init,
                            "--data", str(data), "--out", str(nxt), "--lr", str(args.lr),
                            "--kl", str(args.kl), "--wait-keep", str(wait_keep)],
                           capture_output=True, text=True)
        if t.returncode != 0:
            print(t.stderr[-2000:], flush=True)
            return 1
        tr = json.loads(t.stdout.strip().splitlines()[-1])
        data.unlink(missing_ok=True)                    # rollouts are cheap to regenerate
        rec = {"iter": it, **{f"collect_{k}": v for k, v in col.items()}, **tr,
               "seconds": round(time.time() - t0)}
        if args.eval_every and (it + 1) % args.eval_every == 0:
            e = subprocess.run([str(rg), str(ROOT / "tools" / "rs_eval.py"), "--ckpt", str(nxt),
                                "--opponent-ckpt", args.eval_opponent, "--matches", str(args.eval_matches),
                                "--workers", str(args.workers), "--delay", str(DELAY),
                                "--lead", str(LEAD), *flags],
                               capture_output=True, text=True)
            line = [l for l in e.stdout.splitlines() if " wins (" in l]
            rec["eval"] = line[-1] if line else e.stderr[-300:]
        with log.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("collect")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--opponent", required=True)
    c.add_argument("--matches", type=int, default=48)
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--out", required=True)
    import rs_engine
    rs_engine.add_engine_args(c)
    t = sp.add_parser("train")
    t.add_argument("--ckpt", required=True)
    t.add_argument("--ref", required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--kl", type=float, default=0.05)
    t.add_argument("--clip", type=float, default=0.2)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--wait-keep", default="0.25")
    lp = sp.add_parser("loop")
    lp.add_argument("--name", default="sp_v1")
    lp.add_argument("--init", default=str(ROOT / "runs" / "tokens" / "rs_big_hog" / "best.pt"))
    lp.add_argument("--ref", default="",
                    help="model the KL brake pulls toward (default: --init). Set it to the "
                         "imitation model when --init is already a self-play checkpoint")
    lp.add_argument("--opponent", default=str(ROOT / "runs" / "tokens" / "rs_big" / "best.pt"),
                    help="frozen reactive opponent playing the real opponent's deck")
    lp.add_argument("--iters", type=int, default=40)
    lp.add_argument("--matches", type=int, default=48)
    lp.add_argument("--workers", type=int, default=4)
    lp.add_argument("--lr", type=float, default=1e-5)
    lp.add_argument("--kl", type=float, default=0.05)
    lp.add_argument("--eval-every", type=int, default=5)
    lp.add_argument("--eval-matches", type=int, default=300)
    lp.add_argument("--eval-opponent", default=str(ROOT / "runs" / "tokens" / "rs_bc" / "ep01.pt"),
                    help="the reference every earlier number was measured against (the model "
                         "after rs_bc's first epoch; sim/train_tokens writes it as ep01.pt)")
    rs_engine.add_engine_args(lp)
    args = ap.parse_args()
    return {"collect": collect, "train": train, "loop": loop}[args.cmd](args) or 0


if __name__ == "__main__":
    sys.exit(main())

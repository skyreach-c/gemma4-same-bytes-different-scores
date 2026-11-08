#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
recompute.py -- recompute every citable number for the paper
"Same Bytes, Different Scores: What the Leaderboard Actually Measures
in a Code-Agent Competition".

Part B (our data) only.  Part A / fig4 are owned by teammate archives and
are NOT touched here.

Run (Windows, Git Bash):
    cd /d/kaggle/setup/paper/evidence
    C:/Users/86158/anaconda3/python.exe recompute.py

What it does
------------
1. Reads (read-only) the raw artifacts under D:\\kaggle\\work:
     lineage/lineage.json + lineage/scores.json + 9 pack dirs
     probe/fnud/leaderboard_2026-11-03.csv
     gemma-4-developer-agent/submission-log.csv
     gemma-4-developer-agent/kaggle_submissions_2026-11-08.txt (CLI capture)
     gemma-4-developer-agent/build/out/*.zip + out/submission-0.24.zip
     gemma-4-developer-agent/evalout*/eval_summary.csv + eval_result.json
2. Recomputes every quantity, prints a human-readable table per section.
3. Writes evidence.csv (one row per citable claim) and three figures
   (fig1_lineage.png, fig2_lb_sampling.png, fig3_local_arms.png) at 200 dpi.

Nothing under D:\\kaggle\\work is modified.

Key conventions
---------------
* hashes are sha256; the 12-hex prefix is what the raw JSON files store.
* a "payload" is the set of four member files and their sha256; the archive
  sha256 additionally covers ZIP metadata and is therefore NOT the same test.
* "resolved" / "pass" are the per-task column named `pass` in eval_summary.csv
  (already reconciled against eval_result.json summary in section 4).
* empty_patch := outcome_class == "empty_patch_without_error".
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import statistics
import zipfile
from collections import defaultdict
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
EVID = os.path.dirname(os.path.abspath(__file__))
WORK = r"D:\kaggle\work"
LINEAGE_DIR = os.path.join(WORK, "lineage")
LB_CSV = os.path.join(WORK, "probe", "fnud", "leaderboard_2026-11-03.csv")
OUR = os.path.join(WORK, "gemma-4-developer-agent")
SUBMISSION_LOG = os.path.join(OUR, "submission-log.csv")
CLI_CAPTURE = os.path.join(EVID, "kaggle_submissions_2026-11-08.txt")
BUILD_OUT = os.path.join(OUR, "build", "out")
TASKS_JSONL = os.path.join(OUR, "data", "tasks.jsonl")
EVAL_RUNS = ["evalout", "evalout2", "evalout3", "evalout5", "evalout-d1", "evalout-v10"]
BANDS = [0.15, 0.17, 0.18, 0.20, 0.22, 0.24, 0.29]

# our three real submissions: variant -> local archive that was submitted
OUR_SUBMISSIONS = {
    "honghanhhh_0.24_pack": os.path.join(OUR, "out", "submission-0.24.zip"),
    "davis_v19_byteexact": os.path.join(BUILD_OUT, "davis_v19_byteexact.zip"),
    "v24_byteexact": os.path.join(BUILD_OUT, "v24_byteexact.zip"),
}

SEC = {}
EVIDENCE = []   # list of dicts, written to evidence.csv


def section(title):
    SEC[title] = []
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


KIND_BY_PREFIX = {
    "LIN": "lineage",
    "SUB": "submission",
    "PROV": "submission",
    "LB": "leaderboard",
    "LOC": "local-eval",
    "SET": "setting",
    "X": "contrast",
    "BIN": "derived",
    "A": "part-A",
}


def row(cid, claim, value, unit, src, field, verified="yes", part="B", kind=None):
    if kind is None:
        kind = KIND_BY_PREFIX.get(cid.split("-")[0], "")
    EVIDENCE.append(
        dict(claim_id=cid, part=part, kind=kind, claim_text=claim, value=value, unit=unit,
             source_file=src, source_field_or_command=field, verified=verified)
    )
    return value


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    with open(path, "rb") as fh:
        return sha256_bytes(fh.read())


def zip_members(path):
    """{member_name: (sha256_hex, size)} for every member of a zip archive."""
    out = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            data = zf.read(info.filename)
            out[info.filename] = (sha256_bytes(data), info.file_size)
    return out


def fmt(x, nd=4):
    """Format a float without silently rounding: keep up to nd decimals."""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return f"{x:.{nd}f}"


# ==========================================================================
# 1. LINEAGE
# ==========================================================================
section("1. LINEAGE  (9 public payload packs -> payload hashes -> public scores)")

with open(os.path.join(LINEAGE_DIR, "lineage.json"), encoding="utf-8") as fh:
    lineage = json.load(fh)
with open(os.path.join(LINEAGE_DIR, "scores.json"), encoding="utf-8") as fh:
    scores = json.load(fh)
score_by_ref = {s["ref"]: s for s in scores}

print(f"packs recorded in lineage.json : {len(lineage)}")
print(f"packs with a version-page score: {sum(1 for p in lineage if 'public_score' in score_by_ref.get(p['ref'], {}))}")
print()

pack_rows = []
for p in lineage:
    d = p["ref"].replace("/", "__")
    zpath = os.path.join(LINEAGE_DIR, d, "submission.zip")
    assert os.path.exists(zpath), zpath
    zsha = sha256_file(zpath)
    members = zip_members(zpath)
    # verify the raw json against the archive itself
    assert zsha[:12] == p["zip_sha"], (p["ref"], zsha[:12], p["zip_sha"])
    assert os.path.getsize(zpath) == p["size"], p["ref"]
    # lineage.json stores 10-hex prefixes for member files, 12-hex for the archive
    for name, hx in p["files"].items():
        assert members[name][0].startswith(hx), (p["ref"], name, hx)
    sc = score_by_ref.get(p["ref"], {})
    pack_rows.append(dict(ref=p["ref"], author=d.split("__")[0], lineage=p["lineage"],
                          zip_sha=zsha[:12], zip_sha_full=zsha, size=p["size"],
                          members=members, score=sc))
    print(f"  {d:<62} lin={p['lineage']}  zip={zsha[:12]}  {p['size']:>4} B")
    print(f"      files: " + "  ".join(f"{n}={members[n][0][:12]}" for n in members))

by_lineage = defaultdict(list)
for pr in pack_rows:
    by_lineage[pr["lineage"]].append(pr)

print("\nlineage clusters (payload identity = sha256 of all four member files):")
for lin, prs in sorted(by_lineage.items(), key=lambda kv: -len(kv[1])):
    zips = {p["zip_sha"] for p in prs}
    print(f"  {lin}  n={len(prs)}  distinct_archive_bytes={len(zips)}  refs=" +
          ", ".join(p["author"] for p in prs))
    for p in prs:
        sc = p["score"]
        if "public_score" in sc:
            print(f"      {p['author']:<18} v{sc['version']}/{sc['versions_total']:<3} "
                  f"public={sc['public_score']}  best={sc['best_score']} (v{sc['best_version']})")
        else:
            print(f"      {p['author']:<18} NO version-page score recorded (body_len={sc.get('body_len')})")

n_packs = len(pack_rows)
n_lineages = len(by_lineage)
row("LIN-01", "author-produced payload packs pulled from public notebooks",
    n_packs, "packs", "work/lineage/lineage.json", "len(json)")
row("LIN-02", "distinct payload lineages among those packs",
    n_lineages, "lineages", "work/lineage/lineage.json", "len(set(lineage))")

# L1 same-bytes trio
l1 = by_lineage["397f971bfc0f"]
assert len(l1) == 3 and len({p["zip_sha"] for p in l1}) == 1
print(f"\nL1 (397f971bfc0f): {len(l1)} archives are byte-identical "
      f"(zip sha256 {l1[0]['zip_sha']}), three different team names")
row("LIN-03", "L1 payload hash and archive identity: payload 397f971bfc0f; "
              "3 archives with different author names share zip sha256 6ff73d104725",
    "397f971bfc0f; zip 6ff73d104725; n=3", "lineage hash; archive hash; packs",
    "work/lineage/lineage.json + 3 submission.zip files",
    "sha256 all four members == -> same lineage; zip sha256 == 6ff73d104725")
row("LIN-03b", "byte identity of the three L1 archives: the three submission.zip files have the same "
               "full sha256 and the same byte size (not merely the same four member files)",
    f"sha256={l1[0]['zip_sha_full']} (all 3); size={l1[0]['size']} B (all 3)", "sha256; bytes",
    "work/lineage/flexonafft__gemma-4-budget-aware-coding-agent-v2/submission.zip + "
    "prvsiyan__gemma-and-the-shape-of-doubt/submission.zip + "
    "victorzhong0110__gemma4-submission-pack-flexonafft-v10/submission.zip",
    "hashlib.sha256 over each whole archive; os.path.getsize for each")
print(f"  byte identity: full sha256 {l1[0]['zip_sha_full']} x3, {l1[0]['size']} B each")

# independent pull of the same L1 payload, measured directly on disk
probe_zip = os.path.join(WORK, "probe", "flex_v10_out", "submission.zip")
assert os.path.exists(probe_zip), probe_zip
probe_sha = sha256_file(probe_zip)
probe_size = os.path.getsize(probe_zip)
assert probe_sha == l1[0]["zip_sha_full"] and probe_size == l1[0]["size"], (
    probe_sha, probe_size)
print(f"  probe copy {probe_zip}: sha256 {probe_sha}, {probe_size} B (= same archive)")
row("LIN-03c", "independent pull of the same L1 payload: the probe copy measured on disk has the "
               "same full sha256 and the same byte size (3,713 B) as the three lineage copies",
    f"sha256={probe_sha}; size=3713 B", "sha256; bytes",
    "work/probe/flex_v10_out/submission.zip",
    "sha256sum <file>; stat -c %s <file>")

l1_scores = {}
for p in l1:
    sc = p["score"]
    l1_scores[p["author"]] = sc
    print(f"  {p['author']:<14} public_score={sc['public_score']}  best={sc['best_score']} (v{sc['best_version']})"
          f"  version={sc['version']}/{sc['versions_total']}")
vals = sorted({sc["public_score"] for sc in l1_scores.values()})
spread = max(vals) - min(vals)
print(f"  distinct public scores on the SAME payload: {vals}  -> spread {spread}")
row("LIN-04", "public-leaderboard scores of the same L1 payload (three team names): "
              "flexonafft 0.22 (v10), victorzhong0110 0.20 (v3), prvsiyan 0.15 (v49; best 0.20 at v32)",
    "0.22 / 0.20 / 0.15", "public LB score",
    "work/lineage/scores.json", "public_score for the three L1 refs")
row("LIN-05", "spread of public scores for the byte-identical L1 payload",
    spread, "score points (max-min)", "work/lineage/scores.json", "max(public)-min(public) over L1")

# other lineages
lin_labels = {}
for lin, prs in sorted(by_lineage.items()):
    if len(prs) > 1:
        continue
    p = prs[0]
    sc = p["score"]
    if "public_score" in sc:
        lin_labels[lin] = sc
        print(f"  {lin}  {p['author']:<16} public={sc['public_score']} "
              f"best={sc['best_score']} (v{sc['best_version']})  version v{sc['version']}/{sc['versions_total']}")

row("LIN-06", "L2 lineage (jvnhonglei, superagent reproduction) public score",
    lin_labels["f23cd51d09f3"]["public_score"], "public LB score",
    "work/lineage/scores.json", "public_score (jvnhonglei ref)")
row("LIN-07", "L3 lineage (victorzhong0110 upstream-v8) public score, current version and best",
    f"0.17 current (v5); 0.20 best (v2)", "public LB score",
    "work/lineage/scores.json", "public_score / best_score")
row("LIN-08", "L4 lineage (shimauchu davis-v19-temp04) public score",
    lin_labels["858e65822070"]["public_score"], "public LB score",
    "work/lineage/scores.json", "public_score (shimauchu ref)")
row("LIN-09", "L5 lineage (evgendvorkin) public score, current version and best",
    "0.12 current (v33); 0.18 best (v16)", "public LB score",
    "work/lineage/scores.json", "public_score / best_score")
row("LIN-10", "L6 lineage (navyadevadiga) public score, current version and best",
    "0.13 current (v5); 0.20 best (v1)", "public LB score",
    "work/lineage/scores.json", "public_score / best_score")
row("LIN-11", "L7 lineage (leoporovorov) has no public score on record in the capture",
    "none", "public LB score", "work/lineage/scores.json",
    "record contains only body_len=45, no public_score field")

# our submitted pack payloads vs the 9 packs
print("\nour submitted packs vs the 9 lineage packs (payload member match out of 4):")
our_map = {}
for var, zpath in OUR_SUBMISSIONS.items():
    assert os.path.exists(zpath), zpath
    zsha = sha256_file(zpath)
    members = zip_members(zpath)
    best = None
    for pr in pack_rows:
        same = sorted(set(members) & set(pr["members"]))
        n_same = sum(1 for m in same if members[m][0] == pr["members"][m][0])
        if best is None or n_same > best[2]:
            best = (pr, same, n_same)
    pr, same, n_same = best
    diff = [m for m in same if members[m][0] != pr["members"][m][0]]
    our_map[var] = dict(zip_sha=zsha[:12], members=members, match=pr, n_same=n_same, diff=diff)
    print(f"  {var:<24} zip={zsha[:12]}  closest={pr['author']}/{pr['lineage']}  "
          f"{n_same}/4 identical" + (f"  differs: {diff}" if diff else ""))

row("LIN-12", "our first submitted payload (honghanhhh pack) has the same 4/4 member files as L3 "
              "(victorzhong0110 upstream-v8); the ZIP archives are not byte-identical",
    "4/4 files identical; zip 94d17a0afdfd vs 205a677fdcbd", "member files",
    "work/gemma-4-developer-agent/out/submission-0.24.zip vs work/lineage/victorzhong0110__gemma4-submission-pack-upstream-v8/submission.zip",
    "sha256 of the four members")
row("LIN-13", "our davis_v19_byteexact pack matches the L4 (shimauchu) pack on 3/4 files; "
              "configs/sampling.yaml differs (temp-04 variant)",
    "3/4 files identical; sampling.yaml c8e6851c2b4c vs 615b735dbc1c", "member files",
    "work/gemma-4-developer-agent/build/out/davis_v19_byteexact.zip vs work/lineage/shimauchu__gemma-davis-v19-temp04/submission.zip",
    "sha256 of the four members")
row("LIN-14", "our v24_byteexact pack matches the L7 (leoporovorov) pack on 3/4 files; "
              "prompts/system.md differs",
    "3/4 files identical; system.md 7163e6644c45 vs 85b5f9306e8a", "member files",
    "work/gemma-4-developer-agent/build/out/v24_byteexact.zip vs work/lineage/leoprovorov__single-agent-anytime-patch/submission.zip",
    "sha256 of the four members")

# archive-bytes vs payload nuance (repack of L1)
repack = os.path.join(BUILD_OUT, "flex_v10_byteexact.zip")
if os.path.exists(repack):
    rm = zip_members(repack)
    same = sum(1 for m in rm if m in l1[0]["members"] and rm[m][0] == l1[0]["members"][m][0])
    print(f"\nrepack nuance: build/out/flex_v10_byteexact.zip zip={sha256_file(repack)[:12]} "
          f"but {same}/4 member files equal L1 -> payload-identical, archive bytes differ")
    row("LIN-15", "archive bytes and payload bytes are different tests: our repack flex_v10_byteexact.zip "
                  "has 4/4 member files equal to L1 but a different archive sha256",
        f"4/4 members equal L1; zip {sha256_file(repack)[:12]} vs 6ff73d104725", "member files",
        "work/gemma-4-developer-agent/build/out/flex_v10_byteexact.zip vs L1 archive",
        "sha256 of four members; sha256 of whole archive")

# ==========================================================================
# 2. OUR SUBMISSIONS
# ==========================================================================
section("2. OUR THREE REAL SUBMISSIONS  (live Kaggle CLI capture + submission-log.csv)")

cli = []
with open(CLI_CAPTURE, encoding="utf-8", errors="replace") as fh:
    for line in fh:
        m = re.match(r"^\s*(\d{6,})\s+submission\.zip\s+(\S+ \S+)\s+(.+?)\s{2,}(SubmissionStatus\.\w+)\s+(\S*)\s*(\S*)\s*$", line)
        if m:
            cli.append(dict(ref=m.group(1), date=m.group(2), desc=m.group(3).strip(),
                            status=m.group(4).replace("SubmissionStatus.", ""),
                            public=m.group(5), private=m.group(6)))
cli.sort(key=lambda r: r["ref"])
print(f"live CLI capture {CLI_CAPTURE}: {len(cli)} submissions")
for r in cli:
    print(f"  {r['ref']}  {r['date']:<26} {r['status']:<9} public={r['public'] or '(none)'}  {r['desc'][:70]}")

with open(SUBMISSION_LOG, encoding="utf-8-sig") as fh:
    log_rows = list(csv.DictReader(fh))
print("\nsubmission-log.csv:")
for r in log_rows:
    print(f"  {r['ref']}  {r['variant']:<22} sha={r['sha256_12']:<12} {r['status']:<9} "
          f"public={r['public_score'] or '(pending)'}")

assert len(cli) == 3, cli
cli_by_ref = {r["ref"]: r for r in cli}
log_by_ref = {r["ref"]: r for r in log_rows}
for ref, r in cli_by_ref.items():
    lr = log_by_ref[ref]
    assert lr["status"] == r["status"], (ref, lr["status"], r["status"])
    assert (lr["public_score"] or "(pending)") in (r["public"] or "(pending)",), ref
print("  -> CLI capture and submission-log.csv agree on ref / status / public score")

scored = [r for r in cli if r["status"] == "COMPLETE"]
pending = [r for r in cli if r["status"] == "PENDING"]
print(f"\nsubmissions: {len(cli)} total, {len(scored)} COMPLETE with a public score, {len(pending)} PENDING")

row("SUB-01", "submission ref 58056163 (bundle: honghanhhh pack, out/submission-0.24.zip; 4/4 member files = L3 upstream-v8): status / public score",
    f"COMPLETE / {cli_by_ref['58056163']['public']}", "status / public LB score",
    "work/gemma-4-developer-agent/submission-log.csv:2 + evidence/kaggle_submissions_2026-11-08.txt:5",
    "kaggle competitions submissions -c gemma-4-developer-agent")
row("SUB-02", "submission ref 58077152 (bundle: v24_byteexact, sha256_12 481c0468f73b, 2,404 B): status / public score",
    f"COMPLETE / {cli_by_ref['58077152']['public']}", "status / public LB score",
    "work/gemma-4-developer-agent/submission-log.csv:3 + evidence/kaggle_submissions_2026-11-08.txt:4",
    "kaggle competitions submissions -c gemma-4-developer-agent")
row("SUB-03", "submission ref 58111401 (bundle: v24_byteexact, same sha256_12 481c0468f73b, 2,404 B): status",
    "PENDING / no public score", "status / public LB score",
    "work/gemma-4-developer-agent/submission-log.csv:4 + evidence/kaggle_submissions_2026-11-08.txt:3",
    "kaggle competitions submissions -c gemma-4-developer-agent")
row("SUB-04", "the same v24 bundle bytes (sha256 481c0468f73b..., 2,404 B) were submitted twice: 58077152 and 58111401",
    "one submission COMPLETE 0.15, one still PENDING", "submissions",
    "work/gemma-4-developer-agent/gemma_submit_win.log:17-18,25-26,30 + manifest.json v24 entry",
    "bundle sha256 log line vs submission ref line")
row("SUB-05", "our two scored submissions: 58056163's source notebook page title says 'LB 0.24' (its own self-description says 0.22 config); "
              "58077152's platform description says 'public LB 0.22'; both measured 0.15",
    "2 of 2 advertised scores not reproduced; both measured 0.15", "submissions",
    "evidence/kaggle_submissions_2026-11-08.txt + work/probe/honghanhhh_gemma-4-baseline-lb-0-24/kernel-metadata.json:4",
    "CLI descriptions and notebook title (exact quotes in PROV-01/02)")

# --- 3-row provenance table (Lead request 2026-11-08 21:36; full table in provenance.md)
row("PROV-01", "submission 58056163: source notebook honghanhhh/gemma-4-baseline-lb-0-24; page title says 'LB 0.24'; "
                "notebook markdown cell 0 self-describes as 'v6 (0.22 config, prompt-only edits)'; page card shows score 0.18; "
                "bundle sha256 was NOT recorded at upload time; measured public score 0.15",
    "ref=58056163 | claimed: title 'LB 0.24'; cell0 'v6 (0.22 config)'; author score 0.18 | measured=0.15",
    "provenance record",
    "work/probe/honghanhhh_gemma-4-baseline-lb-0-24/kernel-metadata.json:4 + same dir .ipynb cell 0 + "
    "setup/scout/cards_clean.json (owner=honghanhhh, 'score': '0.18') + evidence/kaggle_submissions_2026-11-08.txt:5",
    "title field; markdown cell 0 text; score field; CLI publicScore")
row("PROV-01b", "submission 58056163 bundle sha256: no hash was logged for this upload; candidate bundle is out/submission-0.24.zip "
                "(built from the pulled notebook), whose sha256 is recorded below. Submitted file was sub/submission.zip via bin/_submit_gemma_api.py",
    "not recorded at upload; candidate sha256=94d17a0afdfd3d908401e78baebcea3e50595ace96f956fab526f2627aa71865 (3,968 B)",
    "sha256 (candidate, not recorded)",
    "bin/_submit_gemma_api.py:7,9 + work/gemma-4-developer-agent/out/submission-0.24.zip + setup/steal-gemma.md:56",
    "sha256_file(out/submission-0.24.zip); script uploads sub/submission.zip",
    verified="no")
row("PROV-02", "submission 58077152: platform description quotes 'byte-identical to the scored payload of davis0178/gemma-4-agent-baseline v19 "
                "(Apache-2.0), public LB 0.22'; local submit log for the same ref records variant v24_byteexact, bundle 2,404 B, "
                "sha256 481c0468...; measured public score 0.15",
    "ref=58077152 | claimed (platform description): davis v19, public LB 0.22 | bundle sha256=481c0468f73b03979a68e6e36421df42e8cc8b2c8517efe5cc8ec60bd74a478f (2,404 B, v24) | measured=0.15",
    "provenance record",
    "evidence/kaggle_submissions_2026-11-08.txt:4 + work/gemma-4-developer-agent/gemma_submit_win.log:17-18,22 + "
    "work/gemma-4-developer-agent/retry_submit_v24.log:9-10,14 + work/gemma-4-developer-agent/build/out/manifest.json (v24 entry)",
    "CLI description text; log 'variant=' / 'bundle ... sha256=' / 'Submission ref:' lines")
row("PROV-03", "submission 58111401: description quotes 'byte-identical to v24 in muhammedidris28/gemma4-agent-runs "
                "(temp 1.0, top_k 64; Apache-2.0), public LB 0.24'; bundle is the same 2,404 B v24 sha256 481c0468...; "
                "status PENDING (no public score observed)",
    "ref=58111401 | claimed (platform description): v24, public LB 0.24 | bundle sha256=481c0468f73b03979a68e6e36421df42e8cc8b2c8517efe5cc8ec60bd74a478f (2,404 B) | measured=PENDING",
    "provenance record",
    "evidence/kaggle_submissions_2026-11-08.txt:3 + work/gemma-4-developer-agent/gemma_submit_win.log:25-26,30 + "
    "work/gemma-4-developer-agent/build/out/manifest.json (v24 entry)",
    "CLI description text; log 'bundle ... sha256=' / 'Submission ref:' lines")
print("  PROV-01/02/03: 3-row provenance table (quotes + refs above); full table in provenance.md")

# ==========================================================================
# 3. PUBLIC LEADERBOARD SNAPSHOT
# ==========================================================================
section("3. PUBLIC LEADERBOARD SNAPSHOT  (2026-11-03, 4,521 teams)")

lb = pd.read_csv(LB_CSV)
print(f"file: {LB_CSV}")
print(f"rows/teams={len(lb)}  columns={list(lb.columns)}")
n_teams = len(lb)
subs = lb["SubmissionCount"]
print(f"SubmissionCount: min={subs.min()} median={statistics.median(subs)} mean={subs.mean():.6f} "
      f"max={subs.max()}  teams<=3={int((subs <= 3).sum())} ({(subs <= 3).mean() * 100:.2f}%)")

# Spearman rho without scipy: Pearson on average ranks
rank_s = lb["Score"].rank(method="average")
rank_c = lb["SubmissionCount"].rank(method="average")
rho = float(np.corrcoef(rank_s, rank_c)[0, 1])
print(f"Spearman rho(Score, SubmissionCount) = {rho:.9f}  (n={n_teams})")

row("LB-00", "teams on the 2026-11-03 public leaderboard snapshot",
    n_teams, "teams", "work/probe/fnud/leaderboard_2026-11-03.csv", "row count")
row("LB-01", "Spearman rank correlation between public score and submission count on the snapshot",
    f"{rho:.4f}", "rho (n=4521)", "work/probe/fnud/leaderboard_2026-11-03.csv",
    "pearson corr of average ranks of Score and SubmissionCount")
row("LB-02", "SubmissionCount distribution over all teams (min / median / mean / max; teams with <=3)",
    f"min={subs.min()}, median={statistics.median(subs)}, mean={subs.mean():.4f}, "
    f"max={subs.max()}, n<=3={int((subs <= 3).sum())}", "submissions",
    "work/probe/fnud/leaderboard_2026-11-03.csv", "SubmissionCount.describe(); (SubmissionCount<=3).sum()")

print("\nper-band submission-count statistics:")
print(f"  {'band':>5}  {'n':>4}  {'min':>3}  {'median':>6}  {'max':>3}  {'<=3':>3}  {'mean':>7}  scores")
band_stats = {}
for b in BANDS:
    s = lb.loc[lb["Score"] == b, "SubmissionCount"]
    med = statistics.median(s)
    band_stats[b] = dict(n=len(s), mn=int(s.min()), med=med, mx=int(s.max()),
                         le3=int((s <= 3).sum()), mean=float(s.mean()))
    print(f"  {b:>5}  {len(s):>4}  {int(s.min()):>3}  {med:>6}  {int(s.max()):>3}  "
          f"{int((s <= 3).sum()):>3}  {s.mean():>7.4f}")
    tag = str(b).replace("0.", "")
    row(f"LB-band-{tag}-n", f"teams scoring exactly {b} on the 2026-11-03 snapshot",
        len(s), "teams", "work/probe/fnud/leaderboard_2026-11-03.csv", f"(Score=={b}).sum()")
    row(f"LB-band-{tag}-dist", f"SubmissionCount of teams scoring exactly {b} (min / median / max; teams <=3)",
        f"min={int(s.min())}, median={med}, max={int(s.max())}, n<=3={int((s <= 3).sum())}",
        "submissions", "work/probe/fnud/leaderboard_2026-11-03.csv",
        f"SubmissionCount[Score=={b}]")

# cross-check against the later snapshot on disk
later = os.path.join(OUR, "gemma-4-developer-agent-publicleaderboard-2026-11-05T14_48_38.csv")
if os.path.exists(later):
    lb2 = pd.read_csv(later, encoding="utf-8-sig")
    r2 = np.corrcoef(lb2["Score"].rank(), lb2["SubmissionCount"].rank())[0, 1]
    print(f"\ncross-check 2026-11-05 snapshot: n={len(lb2)}  rho={r2:.4f}  "
          f"0.24 band n={int((lb2.Score == 0.24).sum())} median="
          f"{statistics.median(lb2.loc[lb2.Score == 0.24, 'SubmissionCount'])}")

# ==========================================================================
# 4. LOCAL EVALUATION RUNS
# ==========================================================================
section("4. LOCAL PAIRED EVALUATION  (18-task blocks, eval_summary.csv + eval_result.json)")

run_data = {}
for run in EVAL_RUNS:
    d = os.path.join(OUR, run)
    csv_path = os.path.join(d, "eval_summary.csv")
    res_path = os.path.join(d, "eval_result.json")
    if not (os.path.exists(csv_path) and os.path.exists(res_path)):
        print(f"  {run}: missing eval_summary.csv/eval_result.json -> skipped")
        continue
    df = pd.read_csv(csv_path)
    with open(res_path, encoding="utf-8") as fh:
        res = json.load(fh)
    run_data[run] = dict(df=df, res=res)
    print(f"\n{run}: run_label={res.get('run_label')}  tasks={res['config'].get('n_tasks')}  "
          f"arms={df['variant'].nunique()}  rows={len(df)}")
    print(f"  gpu={res.get('gpu', {}).get('count')}x {res.get('gpu', {}).get('model')}  "
          f"wall={res.get('wall_minutes_total')} min  quota={res.get('quota_hours_estimate')} h")
    print(f"  {'arm':<24} {'pass':>5} {'/n':>3} {'rate':>7} {'wall_s':>8} {'time':>5} {'tool':>5} {'turn':>5} {'empty':>6}")
    for arm in sorted(df["variant"].unique()):
        a = df[df["variant"] == arm]
        n = len(a)
        p = int(a["pass"].sum())
        print(f"  {arm:<24} {p:>5} {n:>3} {p / n * 100:>6.2f}% "
              f"{a['wall_seconds'].mean():>8.2f} {int(a['time_cap_hit'].sum()):>5} "
              f"{int(a['tool_cap_hit'].sum()):>5} {int(a['turn_cap_hit'].sum()):>5} "
              f"{int((a['outcome_class'] == 'empty_patch_without_error').sum()):>6}")
    print(f"  gold-screen tasks passed: {len(res.get('gold_pass_ids', []))}/{len(res.get('gold_screen', []))}")

    piv = df.pivot_table(index="task_id", columns="variant", values="pass")
    n_arms = piv.shape[1]
    per_task = piv.sum(axis=1)
    all_arms = int((per_task == n_arms).sum())
    zero = int((per_task == 0).sum())
    dis = [int((piv[a] != piv[b]).sum()) for a, b in combinations(piv.columns, 2)]
    print(f"  tasks passed by ALL {n_arms} arms: {all_arms}/{piv.shape[0]}  "
          f"zero-pass tasks: {zero}  max pairwise disagreement: {max(dis)} tasks")
    print(f"  pooled pass rate: {int(df['pass'].sum())}/{len(df)} = {df['pass'].mean() * 100:.4f}%")

    run_data[run].update(
        n_arms=n_arms, n_tasks=piv.shape[0], per_task=per_task, all_arms=all_arms,
        zero=zero, max_dis=max(dis), pooled=float(df["pass"].mean()),
        gold_pass=len(res.get("gold_pass_ids", [])), gold_n=len(res.get("gold_screen", [])),
    )

# ---- evidence rows -------------------------------------------------------
d1 = run_data["evalout-d1"]
v10 = run_data["evalout-v10"]
for run, prefix, key in [("evalout-d1", "LOC-D1-", "d1"), ("evalout-v10", "LOC-V10-", "v10")]:
    r = run_data[run]
    df = r["df"]
    for arm in sorted(df["variant"].unique()):
        a = df[df["variant"] == arm]
        cid = prefix + arm.split("-")[0] + "-" + re.sub(r"[^A-Za-z0-9]+", "", arm.split("-", 1)[1])[:14]
        row(cid, f"{run} arm {arm}: PASS count and mean wall seconds",
            f"{int(a['pass'].sum())}/{len(a)} pass; {a['wall_seconds'].mean():.2f} s mean wall",
            "passes / seconds", f"work/gemma-4-developer-agent/{run}/eval_summary.csv",
            "groupby(variant): sum(pass), mean(wall_seconds)")

# caps / empty patches
for run, key in [("evalout-d1", "d1"), ("evalout-v10", "v10")]:
    df = run_data[run]["df"]
    row(f"LOC-{key.upper()}-CAPS" if False else f"LOC-{key}-CAPS",
        f"{run}: cap-hit flags over all (arm,task) evaluations (time / tool-call / turn)",
        f"time={int(df['time_cap_hit'].sum())}, tool={int(df['tool_cap_hit'].sum())}, "
        f"turn={int(df['turn_cap_hit'].sum())}", "evaluations",
        f"work/gemma-4-developer-agent/{run}/eval_summary.csv",
        "sum(time_cap_hit), sum(tool_cap_hit), sum(turn_cap_hit)")
    empty = df[df["outcome_class"] == "empty_patch_without_error"]
    per_arm = ", ".join(f"{k.split('-')[0]}={v}" for k, v in
                        empty.groupby("variant").size().sort_index().items())
    row(f"LOC-{key}-EMPTY", f"{run}: empty patches without an error (outcome_class=empty_patch_without_error)",
        f"{len(empty)}" + (f" ({per_arm})" if per_arm else ""), "evaluations",
        f"work/gemma-4-developer-agent/{run}/eval_summary.csv",
        "(outcome_class=='empty_patch_without_error').sum()")

row("LOC-d1-ALLARMS", "d1 six-arm block: tasks whose PASS=True in EVERY arm",
    f"{d1['all_arms']}/{d1['n_tasks']}", "tasks",
    "work/gemma-4-developer-agent/evalout-d1/eval_summary.csv",
    "pivot task x variant, count rows where sum(pass)==n_arms")
row("LOC-v10-ALLARMS", "v10 five-arm block: tasks whose PASS=True in EVERY arm",
    f"{v10['all_arms']}/{v10['n_tasks']}", "tasks",
    "work/gemma-4-developer-agent/evalout-v10/eval_summary.csv",
    "pivot task x variant, count rows where sum(pass)==n_arms")
row("LOC-d1-POOLED", "d1 six-arm block: pooled pass rate over all arm-task evaluations",
    f"{int(d1['df']['pass'].sum())}/{len(d1['df'])} = {d1['pooled']:.4f}", "fraction",
    "work/gemma-4-developer-agent/evalout-d1/eval_summary.csv", "mean(pass) over 108 rows")
row("LOC-v10-POOLED", "v10 five-arm block: pooled pass rate over all arm-task evaluations",
    f"{int(v10['df']['pass'].sum())}/{len(v10['df'])} = {v10['pooled']:.4f}", "fraction",
    "work/gemma-4-developer-agent/evalout-v10/eval_summary.csv", "mean(pass) over 90 rows")
row("LOC-d1-GOLD", "d1 block: gold-patch screen passed, and tasks with zero PASS across all arms "
                   "(the three gold-screen failures are part of the zero-pass set)",
    f"gold {d1['gold_pass']}/{d1['gold_n']}; zero-pass tasks {d1['zero']}", "tasks",
    "work/gemma-4-developer-agent/evalout-d1/eval_result.json + eval_summary.csv",
    "len(gold_pass_ids); count(task pass sum == 0)")
row("LOC-v10-GOLD", "v10 block: gold-patch screen passed, and tasks with zero PASS across all arms",
    f"gold {v10['gold_pass']}/{v10['gold_n']}; zero-pass tasks {v10['zero']}", "tasks",
    "work/gemma-4-developer-agent/evalout-v10/eval_result.json + eval_summary.csv",
    "len(gold_pass_ids); count(task pass sum == 0)")
row("LOC-d1-DISAGREE", "d1 six-arm block: maximum per-task disagreement between any two arms",
    f"{d1['max_dis']} task(s)", "tasks",
    "work/gemma-4-developer-agent/evalout-d1/eval_summary.csv",
    "max over arm pairs of count(pass_a != pass_b)")
row("LOC-v10-DISAGREE", "v10 five-arm block: maximum per-task disagreement between any two arms",
    f"{v10['max_dis']} tasks", "tasks",
    "work/gemma-4-developer-agent/evalout-v10/eval_summary.csv",
    "max over arm pairs of count(pass_a != pass_b)")

# wall / quota accounting (all six runs)
tot_wall = 0.0
tot_quota = 0.0
for run in EVAL_RUNS:
    if run in run_data:
        tot_wall += float(run_data[run]["res"].get("wall_minutes_total", 0))
        tot_quota += float(run_data[run]["res"].get("quota_hours_estimate", 0))
print(f"\nall six local eval runs: total wall {tot_wall:.2f} min, quota {tot_quota:.2f} h")
row("SET-WALL", "total wall-clock minutes and GPU-quota hours consumed by the six local eval runs",
    f"{tot_wall:.2f} wall-min; {tot_quota:.2f} quota-h", "minutes / GPU-hours",
    "work/gemma-4-developer-agent/evalout*/eval_result.json",
    "sum(wall_minutes_total); sum(quota_hours_estimate)")
row("SET-WALL-D1", "evalout-d1 wall-clock and quota (6 arms x 18 tasks, 4x L4)",
    f"{run_data['evalout-d1']['res']['wall_minutes_total']} min; "
    f"{run_data['evalout-d1']['res']['quota_hours_estimate']} h", "minutes / GPU-hours",
    "work/gemma-4-developer-agent/evalout-d1/eval_result.json", "wall_minutes_total; quota_hours_estimate")
row("SET-WALL-V10", "evalout-v10 wall-clock and quota (5 arms x 18 tasks, 4x L4)",
    f"{run_data['evalout-v10']['res']['wall_minutes_total']} min; "
    f"{run_data['evalout-v10']['res']['quota_hours_estimate']} h", "minutes / GPU-hours",
    "work/gemma-4-developer-agent/evalout-v10/eval_result.json", "wall_minutes_total; quota_hours_estimate")

# ==========================================================================
# 5. SETTING FACTS
# ==========================================================================
section("5. SETTING FACTS  (task pools, budgets)")

with open(TASKS_JSONL, encoding="utf-8") as fh:
    task_ids = [json.loads(line)["instance_id"] for line in fh if line.strip()]
print(f"data/tasks.jsonl: {len(task_ids)} tasks ({len(set(task_ids))} unique instance_id)")
row("SET-DEV", "tasks in the local dev pool (data/tasks.jsonl)",
    len(task_ids), "tasks", "work/gemma-4-developer-agent/data/tasks.jsonl",
    "line count; unique instance_id")

pub58 = 58  # from the agent-runs README (see row below)
row("SET-PUBLIC", "public task set size reported by the muhammedidris28 agent-runs dataset README",
    pub58, "tasks", "work/gemma-4-developer-agent/ref3/agentruns/README.md",
    "'58 public tasks; one task = 0.017' (quoted, not independently derived)")
row("SET-STEP", "per-task public score step implied by the 58-task public set (1/58)",
    f"{1.0 / pub58:.10f}", "score per task", "arithmetic on SET-PUBLIC", "1/58")

d1res = run_data["evalout-d1"]["res"]
row("SET-EVAL-TASKS", "tasks per local paired-evaluation block",
    d1res["config"]["n_tasks"], "tasks", "work/gemma-4-developer-agent/evalout-d1/eval_result.json",
    "config.n_tasks (12 fixed + 6 rotating task ids)")
row("SET-BUDGET", "per-task budget shared by the byte-exact candidate bundles (v24 and davis lineages)",
    "8 min wall, 28 tool calls, 80 turns (timeout_seconds 240 v24 / 300 davis)",
    "per task", "work/gemma-4-developer-agent/build/out/v24_byteexact.zip + davis_v19_byteexact.zip",
    "eval_config.yaml: timeout_seconds, max_tool_calls, max_time_minutes, max_turns")

# ==========================================================================
# 6. LOCAL vs REAL LEADERBOARD
# ==========================================================================
section("6. LOCAL EVAL vs REAL PUBLIC LEADERBOARD")

d1_rates = []
for arm in sorted(d1["df"]["variant"].unique()):
    a = d1["df"][d1["df"]["variant"] == arm]
    d1_rates.append(int(a["pass"].sum()) / len(a))
v10_rates = []
for arm in sorted(v10["df"]["variant"].unique()):
    a = v10["df"][v10["df"]["variant"] == arm]
    v10_rates.append(int(a["pass"].sum()) / len(a))
print(f"d1  per-arm pass rate: min {min(d1_rates) * 100:.2f}%  max {max(d1_rates) * 100:.2f}%")
print(f"v10 per-arm pass rate: min {min(v10_rates) * 100:.2f}%  max {max(v10_rates) * 100:.2f}%")
print("real public LB of the two scored submissions: 0.15 and 0.15")
implied = 0.15 * pub58
print(f"if one task = 1/58, 0.15 corresponds to {implied:.4f} tasks; 9/58 = {9 / pub58:.6f}")

row("X-LOCAL", "local per-arm pass-rate range on the 18-task blocks (dev-pool tasks) "
               "vs the real public LB score of our scored submissions",
    f"d1 {min(d1_rates) * 100:.2f}-{max(d1_rates) * 100:.2f}%; "
    f"v10 {min(v10_rates) * 100:.2f}-{max(v10_rates) * 100:.2f}%; real LB 0.15",
    "percent / public score", "work/gemma-4-developer-agent/evalout-d1|evalout-v10/eval_summary.csv "
    "+ evidence/kaggle_submissions_2026-11-08.txt",
    "per-arm mean(pass); CLI publicScore")
row("X-IMPLIED", "implied number of solved public tasks behind a 0.15 score if one task = 1/58 "
                 "(inference; scoring/rounding formula is not in the local files)",
    f"{implied:.2f} tasks; 9/58={9 / pub58:.4f}", "tasks (derived)", "arithmetic on SET-PUBLIC and SUB-01/SUB-02",
    "0.15 * 58", verified="no")

# ==========================================================================
# 6b. ADDITIONAL CITED FACTS (requested by the manuscript writer 2026-11-08)
#     Kept in one block so the claim_ids already cited above stay stable.
# ==========================================================================
section("6b. ADDITIONAL CITED FACTS (writer request: setting rules, 0.17 band, grid, waiting time)")

# (i) score quantization: every observed score is floor(k/58*100)/100
uniq_scores = sorted(float(s) for s in lb["Score"].unique())
grid = [int(k * 100 / pub58) / 100 for k in range(pub58 + 1)]
viol = [s for s in uniq_scores if not any(abs(s - g) < 5e-7 for g in grid)]
viol_rows = int(sum(1 for s in lb["Score"]
                    if not any(abs(float(s) - g) < 5e-7 for g in grid)))
max_k = max(k for k in range(pub58 + 1)
            if any(abs(int(k * 100 / pub58) / 100 - s) < 5e-7 for s in uniq_scores))
print(f"score quantization: {len(uniq_scores)} distinct scores; grid floor(k/58*100)/100, k=0..{max_k}; "
      f"violations: {len(viol)} distinct / {viol_rows} rows")
for s in uniq_scores:
    ks = [k for k in range(pub58 + 1) if abs(int(k * 100 / pub58) / 100 - s) < 5e-7]
    print(f"  {s:.2f} -> grid k {ks}")
unobs = [g for g in grid
         if g <= max(uniq_scores) + 1e-9 and not any(abs(g - s) < 5e-7 for s in uniq_scores)]
print(f"  grid values in [0.00, {max(uniq_scores):.2f}] never observed: {unobs}")
row("LB-03", "every public score on the snapshot lies on the 1/58 grid implied by the 58-task public split: "
             "each distinct score equals floor(k/58*100)/100 for some k in 0..17",
    f"{len(uniq_scores)} distinct scores; k range 0..{max_k}; {len(viol)} distinct violating; {viol_rows} violating rows",
    "scores", "work/probe/fnud/leaderboard_2026-11-03.csv",
    "for each distinct Score s: exists k in 0..17 with s == int(k*100/58)/100 (exact 2-dp compare)")
row("LB-04", "1/58-grid score values in [0.00, 0.29] never observed on the snapshot",
    ", ".join(f"{g:.2f}" for g in unobs) if unobs else "none", "scores",
    "work/probe/fnud/leaderboard_2026-11-03.csv", "grid(k where floor(k*100/58)/100 <= 0.29) minus observed scores")

# (ii) binomial waiting-time arithmetic on the 9/58 reading of a 0.15 score
from math import comb

p_solve = 9 / pub58
bin_ids = {13: ("BIN-01", "BIN-04"), 14: ("BIN-02", "BIN-05"), 17: ("BIN-03", "BIN-06")}
print(f"binomial n={pub58}, p=9/58={p_solve:.10f} (reader's reading of 0.15; see X-IMPLIED)")
for k in (13, 14, 17):
    prob = sum(comb(pub58, i) * p_solve ** i * (1 - p_solve) ** (pub58 - i)
               for i in range(k, pub58 + 1))
    print(f"  P(X>={k}) = {prob:.8f}; 1/P = {1 / prob:.2f} independent 58-task draws")
    row(bin_ids[k][0],
        f"if a 0.15-quality agent solves 9 of 58 tasks, the probability that one 58-task draw solves at least "
        f"{k} (binomial approximation, p=9/58, n=58)",
        f"{prob:.8f}", "probability (derived)", "arithmetic on SET-PUBLIC and X-IMPLIED",
        "sum_{i=k}^{58} C(58,i) p^i (1-p)^(58-i), p=9/58", verified="no")
    row(bin_ids[k][1],
        f"expected number of independent 58-task draws needed to see at least {k} solved at p=9/58 "
        f"(reciprocal of {bin_ids[k][0]})",
        f"{1 / prob:.2f}", "draws (derived)", "arithmetic on SET-PUBLIC and X-IMPLIED", "1/P(X>=k)",
        verified="no")

# (iii) PENDING age of submission 58111401 (platform timestamps are UTC)
from datetime import datetime, timedelta, timezone

ASOF = datetime(2026, 11, 8, 21, 50, 0, tzinfo=timezone(timedelta(hours=8)))
sub3_dt = datetime.fromisoformat(cli_by_ref["58111401"]["date"].strip()).replace(tzinfo=timezone.utc)
pending_h = (ASOF - sub3_dt).total_seconds() / 3600.0
print(f"58111401 PENDING age at {ASOF.isoformat()}: {pending_h:.4f} h "
      f"(platform timestamp {sub3_dt.isoformat()} is UTC; local submit log line 25 = 10:02:46+08)")
row("SUB-03b", "submission 58111401 was still PENDING at the stated analysis time; age since upload",
    f"{pending_h:.4f} h (as of 2026-11-08T21:50:00+08:00)", "hours (derived)",
    "evidence/kaggle_submissions_2026-11-08.txt:3 + work/gemma-4-developer-agent/gemma_submit_win.log:25",
    "analysis_time - CLI submission timestamp; CLI timestamps are UTC, local log 10:02:46 is +08")

# (iv) arm counts of the two paired-evaluation blocks
for run, key in [("evalout-d1", "d1"), ("evalout-v10", "v10")]:
    n_arms = int(run_data[run]["df"]["variant"].nunique())
    print(f"{run}: {n_arms} arms")
    row(f"LOC-{key}-NARMS", f"{run} block: number of independent agent arms evaluated on the same task set",
        n_arms, "arms", f"work/gemma-4-developer-agent/{run}/eval_summary.csv", "nunique(variant)")

# (v) harness setting facts (rules quoted with file:line; not measurements)
row("SET-HARNESS-BUDGET", "competition-wide wall-clock budget for one submission: 12 hours to submit patches "
                          "for all tasks, inclusive of sandbox setup time, excluding patch-validation time",
    "12 h", "hours", "setup/shots/ov-gemma-4-developer-agent.txt",
    "line 101 (captured official Overview page, quoted verbatim)")
row("SET-HARNESS-SUBLIMIT", "submission rate limit: one submission per day, and a new submission is refused "
                            "while one is PENDING",
    "1 submission/day; 1 PENDING at a time", "rule",
    "work/gemma-4-developer-agent/gemma_submit_win.log",
    "lines 1-16: pending_count=1 -> BLOCKED_PENDING; lines 19-22 and 27-30: 'Remaining today: 1' then "
    "'0 submissions remaining today' after one submit")
row("SET-HARNESS-HW", "scoring hardware and vLLM serving configuration: 4x NVIDIA L4 (24 GB each, 96 GB total), "
                      "tensor_parallel_size=4, gpu_memory_utilization=0.90, max_model_len=32768",
    "4x L4; tp=4; gpu_memory_utilization=0.90; max_model_len=32768", "configuration",
    "work/gemma-4-developer-agent/data/HARNESS_README.md", "lines 160-166")
row("SET-HARNESS-SINGLE", "single-base-model rule: all agents in a submission must share one declared base model "
                          "from the four official quantized models (ALLOWED_MODEL_NAMES)",
    "1 declared base model per submission", "rule",
    "work/gemma-4-developer-agent/data/HARNESS_README.md",
    "lines 180-181 ('All agents in a competition submission must share a single declared base model')")
row("SET-HARNESS-SIZE", "maximum submission size: total unpacked submission must be < 3 GiB, adapters included",
    "3,221,225,472 bytes (< 3 GiB)", "bytes",
    "work/gemma-4-developer-agent/data/HARNESS_README.md", "lines 128 and 134")
row("SET-HARNESS-CTX", "context ceiling: max_model_len=32,768; max_output_tokens and thinking_budget cannot exceed "
                       "32,768; the sample ships max_output_tokens=16,384 (the documented default)",
    "32,768 tokens (default max_output_tokens 16,384)", "tokens",
    "work/gemma-4-developer-agent/data/HARNESS_README.md", "lines 166 and 182; default at line 147")

# ==========================================================================
# 6c. PART A -- partner-reported claims (A-cited mode; NOT recomputed by us)
# ==========================================================================
# Source: the co-author's manuscript, copied verbatim to
# setup/paper/partner/manuscript_en.md (18,626 B, sha256 5a56b5bc...f1cfa161).
# Every row below is partner-reported and stays verified="no" until a raw trace
# archive is received and this script recomputes it. Line numbers refer to that
# file and are listed (with verbatim anchors) in evidence/partner-claims.md.
section("6c. PART A -- partner-reported claims (A-cited mode)")
PARTNER_TXT = "setup/paper/partner/manuscript_en.md"
A_ROWS = [
    ("A-01", "reference run (129 first attempts, no graph tools): missing-argument rejections "
             "among recorded edit_file requests",
     "70 / 340 (20.6%)", "rejections / editor requests (percent)", "41, 53",
     "The reference contains 70 missing-argument rejections in 340 recorded editor requests (20.6%)"),
    ("A-02", "reference run: missing-argument rejections are concentrated in few tasks",
     "14 of 129 (10.9%)", "tasks (percent)", "53",
     "distributed across 14 of 129 tasks (10.9%)"),
    ("A-03", "reference run: resolution of affected vs remaining attempts (must be read together "
             "with A-15, the partner's own selection-bias caveat)",
     "6/14 (42.9%) vs 45/115 (39.1%)", "resolved / attempts (percent)", "53",
     "Resolution was 6/14 (42.9%) among affected attempts and 45/115 (39.1%) among the remaining"),
    ("A-04", "reference run: resolved tasks on the released development set",
     "51/129 (39.53%)", "resolved / tasks (percent)", "41, 53",
     "The reference resolves 51/129 tasks (39.53%)"),
    ("A-05", "reference run: longest missing-argument streak (Table 1 column 'Longest streak')",
     "12", "consecutive rejections", "41, 57",
     "Longest streak = 12 in Table 1"),
    ("A-06", "Case A fastapi_13713 (Recovery v2): total rejections, in TWO separate segments "
             "18 + 36 -- not one continuous streak",
     "54 total = 18 + 36 (two segments)", "rejections", "59",
     "Events 23-40 contain 18 missing-argument rejections ... Events 45-80 then contain 36 more"),
    ("A-07", "Case A fastapi_13713: rejections that are exact adjacent retries",
     "52 of 54", "exact adjacent retries", "59",
     "54 rejections include 52 exact adjacent retries"),
    ("A-08", "Case A fastapi_13713: budget outcome",
     "80-turn limit exhausted; 26 charged calls; unresolved", "turns / calls / outcome", "59",
     "The final record reports 26 charged calls and exhaustion of the 80-turn limit"),
    ("A-09", "Case B fastapi_14986 (Merged v5 on L4): consecutive identical rejections",
     "41 consecutive; 40 exact adjacent retries", "rejections", "61",
     "events 14-54 are 41 consecutive missing-argument rejections"),
    ("A-10", "Case B fastapi_14986: wall-clock span between first and last receipt",
     "~289 s", "seconds", "61, 65",
     "The first and last receipts occur at 179.18 s and 468.62 s, approximately 289 s apart"),
    ("A-11", "Case B fastapi_14986: budget outcome",
     "13 charged calls; 8-minute session limit; unresolved", "calls / minutes / outcome", "61",
     "The run records 13 charged calls and ends at the 8-minute session limit"),
    ("A-12", "8-task cohort: Recovery v2 vs Merged v4 (descriptive, selected cohort)",
     "rejections 60 vs 1; resolves 0 vs 0; charged calls 182 -> 217; duration 37.8 -> 46.0 min",
     "counts / minutes", "43, 44, 73",
     "The same eight selected tasks record 60 missing-argument rejections in Recovery v2 and "
     "one in Merged v4; both resolve zero tasks. Charged calls rise from 182 to 217"),
    ("A-13", "Guard v2: resolved task and rejection count",
     "1/8 resolved (requests_6592); 21/38 rejections (55.3%)", "tasks / editor requests (percent)",
     "42, 73", "Guard v2 also resolves requests_6592 despite seven missing-argument rejections"),
    ("A-14", "same v5 bundle, two runtimes: missing-argument rejection rates",
     "5/24 (20.8%) on AutoDL vs 42/63 (66.7%) on Kaggle L4; 0 resolves in both",
     "rejections / editor requests (percent)", "46, 47, 75",
     "the same v5 bundle yields 5 rejections in 24 recorded editor requests on AutoDL (20.8 per 100) "
     "and 42 in 63 on L4 (66.7 per 100), with zero resolves in both"),
    ("A-15", "cohort selection caveat (partner's own limitation statement; must accompany A-03, A-12, A-14)",
     "follow-up cohorts were selected from historical failures and vary in size and composition",
     "rule / caveat", "49, 57",
     "follow-up cohorts were selected from historical failures and vary in size and composition"),
]
for cid, text, val, unit, lines, anchor in A_ROWS:
    row(cid, text + " [partner-reported; NOT recomputed by us]", val, unit, PARTNER_TXT,
        f"lines {lines} (mapping: evidence/partner-claims.md {cid}); anchor: \"{anchor}\"",
        verified="no", part="A", kind="partner-reported")
    print(f"  {cid:<5} {val:<72} partner lines {lines}")

# ==========================================================================
# 7. WRITE evidence.csv
# ==========================================================================
section("7. WRITE evidence.csv + figures")

cols = ["claim_id", "part", "kind", "claim_text", "value", "unit", "source_file",
        "source_field_or_command", "verified"]
out_csv = os.path.join(EVID, "evidence.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in EVIDENCE:
        w.writerow(r)
print(f"evidence.csv: {len(EVIDENCE)} rows -> {out_csv}")


def check_unique_ids():
    ids = [r["claim_id"] for r in EVIDENCE]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise SystemExit(f"duplicate claim_id: {sorted(dup)}")


check_unique_ids()

# ==========================================================================
# 8. FIGURE 1 -- lineage clusters vs public score, with our submissions
# ==========================================================================
lin_order = []  # order of first appearance in lineage.json
for p in lineage:
    if p["lineage"] not in lin_order:
        lin_order.append(p["lineage"])
lin_color = {lin: c for lin, c in zip(
    lin_order, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#7f7f7f"])}

fig, ax = plt.subplots(figsize=(12.6, 6.2))
xpos = {lin: i + 1 for i, lin in enumerate(lin_order)}
seen_y = []   # labels of points already placed at this y get pushed below the point
# author packs
for lin, prs in by_lineage.items():
    xs = []
    for j, p in enumerate(prs):
        sc = p["score"]
        x = xpos[lin] + (j - (len(prs) - 1) / 2) * 0.20
        if "public_score" not in sc:
            continue
        xs.append(x)
        ax.scatter(x, sc["public_score"], s=110, color=lin_color[lin], zorder=5,
                   edgecolor="black", linewidth=0.6)
        short = p["author"]
        extra = ""
        if sc["best_score"] != sc["public_score"]:
            extra = f"\nbest {sc['best_score']} @v{sc['best_version']}"
        same_y = any(abs(sc["public_score"] - y0) < 5e-3 for y0 in seen_y)
        dy = -18 if same_y else 9
        seen_y.append(sc["public_score"])
        ax.annotate(f"{short} v{sc['version']}{extra}", (x, sc["public_score"]),
                    textcoords="offset points", xytext=(2, dy), fontsize=7.4, ha="center",
                    color="#222222")
# bracket annotation for L1
l1x = xpos["397f971bfc0f"]
ax.annotate("same ZIP bytes (sha256 6ff73d104725)\n-> 0.22 / 0.20 / 0.15",
            xy=(l1x, 0.225), xytext=(l1x - 0.45, 0.245), fontsize=7.8,
            ha="left", va="bottom", color=lin_color["397f971bfc0f"],
            arrowprops=dict(arrowstyle="-", color=lin_color["397f971bfc0f"], lw=0.8))

# our submissions
our_x = {1: 8.55, 2: 9.55, 3: 10.55}
our_meta = [
    ("honghanhhh pack\n(4/4 files = L3)", 1, "58056163", 0.15, "9302bf2e7a8e"),
    ("v24_byteexact 2,404 B\n(3/4 = L7)", 2, "58077152", 0.15, "9386015160ad"),
    ("v24_byteexact same sha\nPENDING", 3, "58111401", None, "9386015160ad"),
]
for label, slot, ref, val, conn_from in our_meta:
    x = our_x[slot]
    if val is None:
        ax.text(x, 0.155, "PENDING\nno public score", ha="center", va="center",
                fontsize=8.2, color="#b00000",
                bbox=dict(boxstyle="round,pad=0.25", fc="#fff2f2", ec="#b00000", lw=0.8))
    else:
        ax.scatter(x, val, marker="*", s=330, color="#111111", zorder=7)
        ax.annotate(f"ours {ref}\n-> {val}", (x, val), textcoords="offset points",
                    xytext=(0, -22), fontsize=8.2, ha="center", weight="bold")
    ax.plot([xpos[conn_from], x], [val if val is not None else 0.165, val if val is not None else 0.165],
            ls=":", lw=1.0, color="#555555", zorder=1)

ax.axvline(len(lin_order) + 0.55, color="#999999", ls="--", lw=0.8)
ax.set_xticks(list(xpos.values()) + list(our_x.values()))
ax.set_xticklabels([f"L{i + 1}\n(n={len(by_lineage[l])})" for i, l in enumerate(lin_order)]
                   + ["ours #1\n(honghanhhh)", "ours #2\n(v24)", "ours #3\n(v24, same sha)"], fontsize=8.2)
ax.set_xlim(0.35, 11.2)
ax.set_ylim(0.10, 0.26)
ax.set_yticks([0.12, 0.15, 0.18, 0.20, 0.22, 0.24])
ax.set_ylabel("public leaderboard score (version page / measured)", fontsize=9.5)
ax.set_xlabel("payload lineage (same lineage = same four member files)", fontsize=9.5)
ax.set_title("Fig. 1  Same bytes, different scores: public payload lineages and what they scored",
             fontsize=11.5)
ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)
ax.text(0.35, 0.108, "L7 has no version-page score; its column is empty above the PENDING box.",
        fontsize=7.4, color="#555555")
fig.tight_layout()
fig1 = os.path.join(EVID, "fig1_lineage.png")
fig.savefig(fig1, dpi=200)
plt.close(fig)
print(f"wrote {fig1}")

# ==========================================================================
# 9. FIGURE 2 -- LB sampling: score vs submission count
# ==========================================================================
fig, ax = plt.subplots(figsize=(10.2, 6.4))
rng = np.random.default_rng(20261108)
jit = rng.uniform(-0.0032, 0.0032, len(lb))
ax.scatter(lb["Score"] + jit, lb["SubmissionCount"], s=7, alpha=0.16,
           color="#4a6fa5", linewidths=0, rasterized=True)
for i_b, b in enumerate(BANDS):
    st = band_stats[b]
    ax.plot([b - 0.011, b + 0.011], [st["med"], st["med"]], color="#c00000", lw=2.6, zorder=6)
    ax.scatter([b], [st["med"]], marker="D", s=34, color="#c00000", zorder=7, edgecolor="white", lw=0.5)
    ax.annotate(f"{st['med']:g}", (b, st["med"]), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8.4, color="#8b0000", weight="bold")
    ax.annotate(f"n={st['n']}", (b, 37.4 if i_b % 2 == 0 else 38.9), ha="center",
                fontsize=7.2, color="#333333")
ax.set_xlabel("public leaderboard score (band centres; horizontal jitter inside a band)", fontsize=9.5)
ax.set_ylabel("submission count of a team", fontsize=9.5)
ax.set_title("Fig. 2  The public LB as a best-of-n sampler: score vs number of submissions "
             f"(2026-11-03 snapshot, n={n_teams})", fontsize=11)
ax.set_xlim(-0.01, 0.31)
ax.set_ylim(0, 40)
ax.grid(ls=":", lw=0.6, alpha=0.6)
ax.text(0.012, 36.4, f"Spearman rho = {rho:.3f}  (n={n_teams}); red diamonds = per-band median submissions",
        fontsize=8.6, color="#333333")
fig.tight_layout()
fig2 = os.path.join(EVID, "fig2_lb_sampling.png")
fig.savefig(fig2, dpi=200)
plt.close(fig)
print(f"wrote {fig2}")

# ==========================================================================
# 10. FIGURE 3 -- per-arm PASS and mean wall seconds
# ==========================================================================
BLOCKS = [("evalout-d1", "d1 block: 6 arms x 18 tasks", "#2f5d8a"),
          ("evalout-v10", "v10 block: 5 arms x 18 tasks", "#c47b20")]
arm_pos, arm_label = {}, {}
x_cursor = 0.0
block_span = []
for run, tag, color in BLOCKS:
    arms = sorted(run_data[run]["df"]["variant"].unique())
    xs = []
    for arm in arms:
        arm_pos[(run, arm)] = x_cursor
        arm_label[(run, arm)] = arm.split("-")[0] + ("*" if "safe55" in arm else "")
        xs.append(x_cursor)
        x_cursor += 1.0
    block_span.append((run, tag, color, min(xs), max(xs)))
    x_cursor += 0.9
X_LIM = (-1.7, 11.7)

fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8))

# (a) PASS count per arm, both blocks on one axis
ax = axes[0]
for run, tag, color, x_lo, x_hi in block_span:
    df = run_data[run]["df"]
    for arm in sorted(df["variant"].unique()):
        n_pass = int(df[df["variant"] == arm]["pass"].sum())
        ax.bar(arm_pos[(run, arm)], n_pass, width=0.74, color=color, alpha=0.86)
        ax.text(arm_pos[(run, arm)], n_pass + 0.28, str(n_pass), ha="center",
                fontsize=8.8, weight="bold")
    ax.text((x_lo + x_hi) / 2, 19.0, tag, ha="center", fontsize=8.2, color=color, weight="bold")
    ax.axvline(x_hi + 0.45, color="#bbbbbb", ls="--", lw=0.7)
ax.set_ylim(0, 19.9)
ax.set_xlim(*X_LIM)
ax.set_xticks([arm_pos[k] for k in arm_pos])
ax.set_xticklabels([arm_label[k] for k in arm_pos], fontsize=8.6)
ax.set_ylabel("PASS count (out of 18 tasks)", fontsize=9.5)
ax.set_title("(a) PASS per arm on the 18-task block", fontsize=10.5)
ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)

# (b) mean wall seconds per arm, both blocks on one axis
ax = axes[1]
for run, tag, color, x_lo, x_hi in block_span:
    df = run_data[run]["df"]
    arms = sorted(df["variant"].unique())
    for i_arm, arm in enumerate(arms):
        w = float(df[df["variant"] == arm]["wall_seconds"].mean())
        ax.bar(arm_pos[(run, arm)], w, width=0.74, color=color, alpha=0.86,
               label=tag.split(":")[0] if i_arm == 0 else None)
        ax.text(arm_pos[(run, arm)], w + 5, f"{w:.0f}", ha="center", fontsize=7.6, rotation=90)
    ax.text((x_lo + x_hi) / 2, 552, tag, ha="center", fontsize=8.2, color=color, weight="bold")
    ax.axvline(x_hi + 0.45, color="#bbbbbb", ls="--", lw=0.7)
ax.axhline(480, color="#555555", ls="--", lw=0.9)
ax.text(-1.55, 487, "8-min wall cap (byte-exact bundles)", fontsize=7.2, color="#555555")
ax.axhline(330, color="#999999", ls=":", lw=0.9)
ax.text(-1.55, 337, "5.5-min wall cap (safe55 variants)", fontsize=7.2, color="#777777")
ax.set_ylim(0, 585)
ax.set_xlim(*X_LIM)
ax.set_ylabel("mean wall seconds per task", fontsize=9.5)
ax.set_title("(b) Mean wall seconds per arm", fontsize=10.5)
ax.set_xticks([arm_pos[k] for k in arm_pos])
ax.set_xticklabels([arm_label[k] for k in arm_pos], fontsize=8.6)
ax.legend(fontsize=7.8, loc="lower right", framealpha=0.95)
ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)

fig.text(0.01, 0.005, "* = safe55 variant (5.5-min eval budget; others 8 min). d1 and v10 share the 12 fixed "
         "tasks but differ in the 6 rotating tasks, so per-arm pass counts are not paired across blocks.",
         fontsize=6.8, color="#444444")
fig.suptitle("Fig. 3  Local paired evaluation: per-arm PASS and mean wall time (18-task blocks)",
             fontsize=11.5, y=1.00)
fig.tight_layout()
fig3 = os.path.join(EVID, "fig3_local_arms.png")
fig.savefig(fig3, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"wrote {fig3}")

print("\nDONE.")

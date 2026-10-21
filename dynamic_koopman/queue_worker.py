#!/usr/bin/env python3
"""Pull tasks from a shared task file and run them one by one.

Each line of the task file is the argument string for train.py. Workers in any
number of SLURM jobs can share one file: a task is claimed by atomically creating
runs/.claims/<task-id>, and tasks whose result JSON already exists are skipped,
so a cancelled job can simply be resubmitted (use --reclaim to retry tasks whose
claim was left behind by a killed job). The task file is re-read before every task.
A claim containing HOLD is parked on purpose; one containing FAILED needs a look at
runs/logs/<task>.log and is retried only after its claim directory is deleted.
"""
import argparse
import os
import shlex
import subprocess
import sys
import time

_DIR = os.path.dirname(os.path.abspath(__file__))


def _arm(a):
    return f"{a.get('--koopman', 'eig')}_{a.get('--mode', 'static')}" + \
        (f"_{a['--head']}" if a.get("--head", "kepin") != "kepin" else "") + \
        ("_nokooploss" if "--no_koopman_loss" in a else "") + \
        (f"_future{float(a['--future_loss']):g}" if float(a.get("--future_loss", 0) or 0) > 0 else "") + \
        (f"_sub{a['--future_subspace']}" if float(a.get("--future_loss", 0) or 0) > 0
         and int(a.get("--future_subspace", 0) or 0) else "") + \
        ("_cn" if "--cond_norm" in a else "") + ("_ctrl" if "--koop_control" in a else "")


def task_id(line):
    toks = shlex.split(line)
    a = {}
    i = 0
    while i < len(toks):            # flags may be valued (--x v) or bare (--x)
        if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            a[toks[i]] = toks[i + 1]
            i += 2
        else:
            a[toks[i]] = True
            i += 1
    name = f"{a.get('--protocol', 'clean')}_{a['--dataset']}_{_arm(a)}_run{a.get('--run_id', '0')}"
    return name, a


def result_path(a, out):
    return os.path.join(out, a.get("--protocol", "clean"), a["--dataset"], _arm(a),
                        f"run{a.get('--run_id', '0')}_result.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--out", default=os.path.join(_DIR, "runs"))
    p.add_argument("--worker", default="w")
    p.add_argument("--reclaim", action="store_true")
    args = p.parse_args()

    claims = os.path.join(args.out, ".claims")
    logs = os.path.join(args.out, "logs")
    os.makedirs(claims, exist_ok=True)
    os.makedirs(logs, exist_ok=True)

    tried = set()
    while True:
        # re-read every time: tasks can be added or reordered while jobs run
        lines = [l.strip() for l in open(args.tasks) if l.strip() and not l.startswith("#")]
        picked = None
        for line in lines:
            name, a = task_id(line)
            if name in tried or os.path.exists(result_path(a, args.out)):
                continue
            claim = os.path.join(claims, name)
            try:
                os.mkdir(claim)
            except FileExistsError:
                stale_ok = args.reclaim and not any(
                    os.path.exists(os.path.join(claim, f)) for f in ("HOLD", "FAILED"))
                if not stale_ok:
                    continue
            picked = line
            break
        if picked is None:
            break
        line = picked
        tried.add(name)
        with open(os.path.join(claim, "owner"), "w") as f:
            f.write(f"{args.worker} job={os.environ.get('SLURM_JOB_ID')} {time.ctime()}\n")
        print(f"[{args.worker}] {time.strftime('%H:%M:%S')} start {name}", flush=True)
        t0 = time.time()
        with open(os.path.join(logs, f"{name}.log"), "w") as log:
            code = subprocess.call([sys.executable, os.path.join(_DIR, "train.py"),
                                    *shlex.split(line), "--out", args.out, "--verbose", "1"],
                                   stdout=log, stderr=subprocess.STDOUT)
        print(f"[{args.worker}] {time.strftime('%H:%M:%S')} done  {name} exit={code} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        if code != 0:   # keep the claim so nobody retries it blindly; see runs/logs/<task>.log
            with open(os.path.join(claim, "FAILED"), "w") as f:
                f.write(f"exit={code} {time.ctime()}\n")
    print(f"[{args.worker}] no tasks left", flush=True)


if __name__ == "__main__":
    main()

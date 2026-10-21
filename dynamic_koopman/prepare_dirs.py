#!/usr/bin/env python3
"""Create every output folder a task file needs, from the login node, BEFORE submitting.

Workers on different nodes creating the same new folder at once on the shared (NFS) disk
can fail even with retries; pre-creating the folders avoids the race entirely.
Usage: python prepare_dirs.py tasks/<file>.txt
"""
import os
import sys

from queue_worker import task_id, result_path

_DIR = os.path.dirname(os.path.abspath(__file__))
for line in open(sys.argv[1]):
    if line.startswith("--"):
        _, a = task_id(line.strip())
        os.makedirs(os.path.dirname(result_path(a, os.path.join(_DIR, "runs"))), exist_ok=True)
print("output folders ready")

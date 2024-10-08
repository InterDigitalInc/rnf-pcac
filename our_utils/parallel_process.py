# MIT License

# Copyright (c) 2020 Maurice Quach

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# Copyright (c) 2010-2024, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.


import shlex
import sys
import subprocess
from tqdm import tqdm


class Popen(subprocess.Popen):
    def __init__(self, *args, stdout=None, stderr=None, **kwargs):
        super(Popen, self).__init__(*args, stdout=stdout, stderr=stderr, **kwargs)
        self.stdout = stdout
        self.stderr = stderr


def is_file(f):
    return f is not None and f is not sys.stdout and f is not sys.stderr


def safe_close(f):
    if is_file(f):
        f.close()


def parallel_process(f, params, parallelism):
    procs = []
    try:
        with tqdm(total=len(params)) as pbar:
            i = 0
            while len(params) > 0 or len(procs) > 0:
                if len(procs) < parallelism and len(params) > 0:
                    param = params.pop()
                    procs.append(f(*param))
                elif len(procs) > 0:
                    try:
                        p = procs[i]
                        p.wait(0.1)
                        if p.returncode != 0:
                            logs = ''
                            if is_file(p.stdout):
                                with open(p.stdout.name, 'r') as f:
                                    logs = f.read()
                            raise RuntimeError(f'{" ".join([shlex.quote(x) for x in p.args])} returned with code {p.returncode}\n{logs}')
                        safe_close(p.stdout)
                        safe_close(p.stderr)
                        pbar.update(1)
                        pbar.refresh()
                        procs.pop(i)
                    except subprocess.TimeoutExpired:
                        pass
                    i = (i + 1) % max(len(procs), 1)
    finally:
        for p in procs:
            p.terminate()
            safe_close(p.stdout)
            safe_close(p.stderr)

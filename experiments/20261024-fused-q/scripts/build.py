"""Replay an archived build command with relocated experiment/CUTLASS paths."""
import argparse
import pathlib
import shlex
import subprocess
p = argparse.ArgumentParser()
p.add_argument('variant', choices=['int8bf_fusedq_n80', 'int8bf_fusedq_n96', 'int8bf_n96', 'int8bf_ctrl_bf16', 'c128'])
p.add_argument('--cutlass', type=pathlib.Path, required=True)
p.add_argument('--nvcc', default='/usr/local/cuda/bin/nvcc')
p.add_argument('--dry-run', action='store_true')
a = p.parse_args()
root = pathlib.Path(__file__).resolve().parents[1]
original = '/root/gdn35/20261024-fp8-accuracy'
cutlass = original + '/cutlass-7127592069c2fe01b041e174ba4345ef9b279671'
cmd = shlex.split((root / 'evidence' / ('fa3_build_' + a.variant + '.command')).read_text())
cmd = [v.replace(cutlass, str(a.cutlass.resolve())).replace(original, str(root)) for v in cmd]
cmd[0] = a.nvcc
print(shlex.join(cmd), flush=True)
if not a.dry_run:
    if not (a.cutlass / 'include/cutlass/cutlass.h').is_file():
        p.error('--cutlass must point to the CUTLASS source root')
    subprocess.run(cmd, check=True)

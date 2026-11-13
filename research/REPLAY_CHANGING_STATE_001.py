"""Replay saved checkpoints in a fresh directory; never retrain or overwrite.

Usage: python REPLAY_CHANGING_STATE_001.py --archive ORIGINAL.zip --output NEW_DIR
Run with Python3.12 + Torch2.5.1+cpu + NumPy2.3.5. Script lives beside the two
experiment directories, either in the checkout's research/ or delivered ZIP.
"""
import argparse,hashlib,shutil,subprocess,sys,zipfile
from pathlib import Path
EXPECTED='e4a3d3e39e6e8aa3e6f4f3c6c76b705dacf310be4adb8fc82f068712e678b9f2'

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--timing',action='store_true');a=p.parse_args()
    if hashlib.sha256(a.archive.read_bytes()).hexdigest()!=EXPECTED:raise SystemExit('Original archive hash mismatch')
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=False)
    original=output/'recovered';original.mkdir()
    with zipfile.ZipFile(a.archive) as z:
        for info in z.infolist():
            path=Path(info.filename)
            if path.is_absolute() or '..' in path.parts or ((info.external_attr>>16)&0o170000)==0o120000:
                raise SystemExit('Unsafe archive member')
        z.extractall(original)
    source=Path(__file__).resolve().parent
    for name in ('changing_state_001','slot_control_001'):
        shutil.copytree(source/name,output/name,ignore=shutil.ignore_patterns('results','__pycache__','*.pyc','review','runtime','run_console.txt'))
    commands=[
        [sys.executable,str(output/'slot_control_001/run.py'),'evaluate','--source',str(original)],
        [sys.executable,str(output/'changing_state_001/run_frozen.py'),'run','--source',str(original)],
    ]
    if a.timing:commands.append([sys.executable,str(output/'slot_control_001/timing.py'),'--source',str(original)])
    for command in commands:subprocess.run(command,check=True)
    print('Fresh checkpoint replay complete:',output)
if __name__=='__main__':main()

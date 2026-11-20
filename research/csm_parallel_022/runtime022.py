"""CPU software fingerprint, including compiler and deterministic Torch settings."""
import sys,subprocess,os,platform
import torch
def configure():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def fingerprint():
 configure()
 return {'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),'torch':torch.__version__,'torch_threads':torch.get_num_threads(),'torch_deterministic':torch.are_deterministic_algorithms_enabled(),'device':'cpu','compiler':subprocess.run(['g++','--version'],capture_output=True,text=True,check=True).stdout,'pip_freeze':subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout.splitlines(),'omp_environment':{n:os.environ.get(n) for n in ['OMP_NUM_THREADS','OMP_DYNAMIC','OMP_PROC_BIND','OMP_PLACES']}}

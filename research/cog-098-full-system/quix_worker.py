"""Run the pinned upstream test body on one candidate in a disposable process."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import resource
import signal
import sys

import pytest


def run(root, name, partition, code):
    resource.setrlimit(resource.RLIMIT_AS, (384 << 20, 384 << 20))
    sys.path[:0] = [str(root / 'python_testcases'), str(root)]
    pytest.use_correct = False
    pytest.run_slow = False
    module = importlib.import_module('test_' + name)
    check = getattr(module, 'test_' + name)
    namespace = {}
    exec(compile(code, name + '.py', 'exec'), namespace)
    setattr(module, name, namespace[name])
    cases = {json.dumps(pair[0], sort_keys=True): pair for pair in module.testdata}
    keys = sorted(cases, key=lambda k: hashlib.sha256(k.encode()).digest())
    subset = keys[::2] if partition == 'visible' else keys[1::2]
    if not subset:
        return 'no_cases'
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
    checked = 0
    for key in subset:
        signal.setitimer(signal.ITIMER_REAL, .35)
        try:
            check(*copy.deepcopy(cases[key]))
            checked += 1
        except pytest.skip.Exception:
            continue
        except TimeoutError:
            return 'timeout'
        except (AssertionError, ArithmeticError, IndexError, TypeError, ValueError,
                RecursionError, KeyError, MemoryError, StopIteration, AttributeError, NameError):
            return 'failed'
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    return 'passed' if checked else 'no_cases'


if __name__ == '__main__':
    print(run(Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.stdin.read()))

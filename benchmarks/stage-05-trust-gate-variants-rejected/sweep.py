import json, sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from desic.eval.run import _data
from desic.eval.scenarios import SCENARIOS, make_patched
KEYS = ["accuracy", "ece", "cum_log_loss", "forgetting_index", "poison_repeated", "victim_after", "others_drop", "adaptation_half_life", "moved_final", "others_min", "teacher_rate_last", "answered_accuracy_last", "test_accuracy", "test_ece"]
def job(args):
    name, kw, scen, seed = args
    train, test, classes = _data(None)
    h = SCENARIOS[scen](partial(make_patched, **kw), train, test, classes, seed)["headline"]
    return name, scen, seed, {k: h[k] for k in KEYS if k in h}
if __name__ == "__main__":
    jobs = json.loads(sys.argv[1])
    _data(None)
    with ProcessPoolExecutor(4) as ex:
        for r in ex.map(job, jobs): print(*r, flush=True)

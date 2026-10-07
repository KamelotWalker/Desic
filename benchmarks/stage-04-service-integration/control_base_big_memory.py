from concurrent.futures import ProcessPoolExecutor
from desic.eval.run import _data
from desic.eval.scenarios import SCENARIOS, make_desic
from desic.core.experts import MemoryExpert
def make_bigmem(classes):
    t = make_desic(classes); t.memory = MemoryExpert(capacity=20000); return t
def job(a):
    scen, seed = a
    tr, te, cl = _data(None)
    h = SCENARIOS[scen](make_bigmem, tr, te, cl, seed)["headline"]
    return scen, seed, {k: h[k] for k in list(h)[:5]}
if __name__ == "__main__":
    _data(None)
    with ProcessPoolExecutor(2) as ex:
        for r in ex.map(job, [("sorted", 0), ("shuffled", 0)]): print(*r, flush=True)

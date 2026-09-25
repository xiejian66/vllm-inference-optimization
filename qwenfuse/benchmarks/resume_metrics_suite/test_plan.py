import unittest
from collections import Counter, defaultdict
from experiment import make_workload, plan, SEEDS

class PlanTests(unittest.TestCase):
    def test_unique_jobs(self):
        jobs=plan()
        self.assertEqual(len(jobs),94)
        self.assertEqual(len({x['id'] for x in jobs}),len(jobs))
        self.assertEqual(sum(x['kind']=='model' for x in jobs),40)

    def test_random_reproducible_and_budget_invariant(self):
        for seed in SEEDS:
            a=make_workload('random8',seed=seed)
            b=make_workload('random16',seed=seed)
            self.assertEqual(a,b)
            self.assertEqual(a,make_workload('random8',seed=seed))
            self.assertEqual(Counter(x['kind'] for x in a['labels']),{'cold':32,'warm':32})
            self.assertTrue(all(len(p)==2176 for p in a['prompts']))
            seen=Counter()
            for p,label in zip(a['prompts'],a['labels']):
                if label['kind']=='warm':
                    family=label['family']
                    self.assertEqual(p[:2048],a['prefixes'][family])
                    self.assertEqual(label['occurrence'],seen[family]);seen[family]+=1

    def test_known_seed_unchanged(self):
        self.assertEqual(make_workload('random8',seed=191840815)['sha256'],
            'd508b50eeacf68f3bdfbb9df221d2781c263d6d7a870e8efda25ef7e2a6b9f6f')

    def test_distinct_seeds(self):
        self.assertEqual(len({make_workload('random8',seed=s)['sha256'] for s in SEEDS}),3)

    def test_reverse_order(self):
        groups=defaultdict(list)
        for j in plan():
            if j['purpose']=='main':groups[j['seed'],j['round']].append(j['variant'])
        for seed in SEEDS:
            self.assertEqual(groups[seed,1],list(reversed(groups[seed,2])))

    def test_numerical_repetitions(self):
        counts=Counter(j['variant'] for j in plan() if j['purpose']=='numeric')
        self.assertEqual(counts,dict.fromkeys(('A4','C4','S0','S1','JOINT'),2))

if __name__=='__main__':unittest.main()

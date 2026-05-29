import statistics
import scenic
from scenic.simulators.metadrive.simulator import MetaDriveSimulator
import random
import numpy as np

SCENIC_FILE = 'examples/driving/kyle.scenic'
N_RUNS = 100
seeds = list(range(N_RUNS))

simulator = MetaDriveSimulator(sumo_map="assets/maps/CARLA/Town05.net.xml", timestep=0.1, real_time=0)

results = {}

for K_dd in [0.3, 0.5, 0.7, 1.0]:
    for alpha in [0.05, 0.1, 0.2, 0.3]:
        
        scenario = scenic.scenarioFromFile(
            SCENIC_FILE,
            params={'lookahead_gain': K_dd, 'alpha': alpha, 'max_steering_deg': 40.0}, 
            mode2D=True,
            model='scenic.simulators.metadrive.model'
        )
        
        mean_ctes = []
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            
            # scene, _ = scenario.generate(maxIterations=50)
            # sim = simulator.simulate(scene, maxSteps=600, verbosity=0)
            
            # i dont think this is the correct way to add the seed
            scene, _ = scenario.generate(maxIterations=200, feedback=seed)
            sim = simulator.simulate(scene, maxSteps=600, verbosity=2)
             
            # safety guards
            if sim is None or sim.result is None:
                print(f"  K_dd={K_dd} alpha={alpha} seed={seed} REJECTED")
                continue
                
            ctes = sim.result.records.get('cte', [])
            if not ctes:
                print(f"  K_dd={K_dd} alpha={alpha} seed={seed} NO CTE DATA")
                continue
                
            # added absolute value for signed values of cte
            mean_cte = statistics.mean(abs(cte) for _, cte in ctes)
            mean_ctes.append(mean_cte)
            
            # printing per run
            print(f"  K_dd={K_dd} alpha={alpha} seed={seed} mean_cte={mean_cte:.4f}")

        score = statistics.mean(mean_ctes) if mean_ctes else float('nan')
        results[(K_dd, alpha)] = score
        print(f"=== SUMMARY: K_dd={K_dd} alpha={alpha} overall_mean_cte={score:.4f} ===\n")

best = min(results, key=results.get)
print(f"\nBest K_dd={best[0]}  alpha={best[1]}  score={results[best]:.4f}")
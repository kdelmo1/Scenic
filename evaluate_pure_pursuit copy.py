import statistics
import scenic
import numpy as np
from scenic.simulators.metadrive.simulator import MetaDriveSimulator

SCENIC_FILE = 'examples/driving/kyle.scenic'
N_RUNS = 100
seeds = list(range(N_RUNS))

simulator = MetaDriveSimulator(sumo_map="assets/maps/CARLA/Town05.net.xml", timestep=0.1, real_time=0, render=False)

results = {}
for K_dd in [0.3, 0.5, 0.7, 1.0]:
    scenario = scenic.scenarioFromFile(
        SCENIC_FILE,
        params={'lookahead_gain': K_dd},
        mode2D=True,
        model='scenic.simulators.metadrive.model'
    )
    mean_ctes = []
 
    for seed in seeds:
        scene, _ = scenario.generate(maxIterations=50)
        sim = simulator.simulate(scene, maxSteps=600, verbosity=0)
        if sim is None:
            continue
        ctes = sim.result.records['cte']
        # ctes = [(timestep, value), ...]
        mean_cte = statistics.mean(cte for _, cte in ctes)
        mean_ctes.append(mean_cte)
    results[K_dd] = statistics.mean(mean_ctes) if mean_ctes else float('nan')
    print(f"K_dd={K_dd}  mean_cte={results[K_dd]:.4f}")

best = min(results, key=results.get)
print(f"\nBest K_dd={best}  score={results[best]:.4f}")


# K_dd=0.3  mean_cte=0.2622
# K_dd=0.5  mean_cte=0.2292
# K_dd=0.7  mean_cte=0.2295
# K_dd=1.0  mean_cte=0.2400

# Best K_dd=0.5  score=0.2292

# use numpy function to generate random seeds (np.random.seed)
# use numpy function to create evenly spaced values (numpy.linspace)
    # 20 points in between 0 and 1
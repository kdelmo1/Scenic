import statistics
import scenic
import random
import numpy as np
from scenic.simulators.metadrive.simulator import MetaDriveSimulator, MetaDriveSimulation
from scenic.domains.driving.controllers import PurePursuitLateralController, PIDLongitudinalController

SCENIC_FILE = 'examples/driving/kyle.scenic'
SUMO_MAP = "assets/maps/CARLA/Town05.net.xml"
N_RUNS = 100
seeds = list(range(N_RUNS))


def make_simulator(K_dd, alpha):
    """Create a MetaDriveSimulator whose PP controllers use the given params."""

    class PatchedSimulation(MetaDriveSimulation):
        def getPurePursuitControllers(self, agent):
            dt = self.timestep
            lon = PIDLongitudinalController(K_P=0.5, K_D=0.1, K_I=0.7, dt=dt)
            lat = PurePursuitLateralController(
                cl=agent.length, ld=7, dt=dt, clwbr=0.55,
                K_dd=K_dd, alpha=alpha, max_steering_deg=40.0
            )
            print(f"[DEBUG] getPurePursuitControllers K_dd={K_dd} alpha={alpha}")
            return lon, lat

    class PatchedSimulator(MetaDriveSimulator):
        def createSimulation(self, scene, *, timestep, **kwargs):
            self.scenario_number += 1
            return PatchedSimulation(
                scene,
                render=False,
                render3D=self.render3D,
                scenario_number=self.scenario_number,
                timestep=self.timestep,
                sumo_map=self.sumo_map,
                real_time=self.real_time,
                scenic_offset=self.scenic_offset,
                sumo_map_boundary=self.sumo_map_boundary,
                film_size=self.film_size,
                **kwargs,
            )

    return PatchedSimulator(sumo_map=SUMO_MAP, timestep=0.1, real_time=0)


# bypassing veneer
scenario = scenic.scenarioFromFile(
    SCENIC_FILE,
    mode2D=True,
    model='scenic.simulators.metadrive.model'
)

results = {}

for K_dd in [0.3, 0.5, 0.7, 1.0]:
    for alpha in [0.05, 0.1, 0.2, 0.3]:
        simulator = make_simulator(K_dd, alpha)

        mean_ctes = []
        for seed in seeds:
            # random.seed(seed) <- not sure where this actually goes
            # np.random.seed(seed) <- not sure where this actually goes

            scene, _ = scenario.generate(maxIterations=200)
            sim = simulator.simulate(scene, maxSteps=600, verbosity=0)

            # dont ignore the ones that go off road; take those into account
            # fix code so that it uses the purepursuit controllers in turnbehavior
            
            # validate: put print statement in purepursuit controller, turn real time mode on, and watch prints
            # to see its printing during a turn
            # if sim is None or sim.result is None:
            #     print(f"  K_dd={K_dd} alpha={alpha} seed={seed} REJECTED")
            #     continue
            
            # Fix — only reject if sim is None or genuinely failed:
            if sim is None or sim.result is None:
                # Simulation didn't even start — assign max penalty
                mean_ctes.append(10.0)
                print(f"  K_dd={K_dd} alpha={alpha} seed={seed} FAILED TO START")
                continue
            
            ctes = sim.result.records.get('cte', [])
            steps_completed = len(ctes)
            
            if steps_completed == 0:
                mean_ctes.append(10.0)
                continue

            total_steps = max(600, steps_completed)
            OFF_ROAD_PENALTY = 2.0
            cte_values = [abs(cte) for _, cte in ctes]
            cte_values += [OFF_ROAD_PENALTY] * (total_steps - steps_completed)
            

            # mean_cte = statistics.mean(abs(cte) for _, cte in ctes)
            # mean_ctes.append(mean_cte)
            # print(f"  K_dd={K_dd} alpha={alpha} seed={seed} mean_cte={mean_cte:.4f} steps={steps_completed}/600")
            
            mean_cte = statistics.mean(cte_values)
            mean_ctes.append(mean_cte)
            print(f"  K_dd={K_dd} alpha={alpha} seed={seed} mean_cte={mean_cte:.4f} steps={steps_completed}/600")

        n_valid = len(mean_ctes)
        n_rejected = len(seeds) - n_valid
        score = statistics.mean(mean_ctes) if mean_ctes else float('nan')
        results[(K_dd, alpha)] = {
            'score': score,
            'valid': n_valid,
            'rejected': n_rejected
        }
        print(f"=== SUMMARY: K_dd={K_dd} alpha={alpha} overall_mean_cte={score:.4f} "
                f"valid={n_valid}/100 rejected={n_rejected}/100 ===\n")

best = min(results, key=lambda k: results[k]['score'])
print(f"\nBest K_dd={best[0]}  alpha={best[1]}  "
      f"score={results[best]['score']:.4f}  "
      f"valid={results[best]['valid']}/100")
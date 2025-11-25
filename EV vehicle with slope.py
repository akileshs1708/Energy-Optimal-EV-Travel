import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List
import time


class VehicleParameters:

    def __init__(self):
        self.R_bat = 0.05  
        self.V_alim = 150.0 
        self.R_m = 0.03 
        self.K_m = 0.27 
        self.L_m = 0.05 

        self.r = 0.33 
        self.K_r = 10.0 
        self.M = 250.0 
        self.g = 9.81 
        self.K_f = 0.03  
        self.rho = 1.293  
        self.S = 2.0 
        self.C_x = 0.4 
        self.J = 0.1  
        self.i_max = 150.0  
        self.delta = 1.0  


class OptimizationConfig:

    def __init__(self):
        self.s = 10  
        self.st_tf = 2.0  
        self.st_x2 = 0.1  

        self.integration_step = 1e-4 
        self.integration_method = 'RK4' 

        self.heuristic = 'H6'
        self.range_reduction = 20 

        self.acceleration_limit = 0.3 
        self.max_acceleration_g = self.acceleration_limit * 9.81  


class ProblemDefinition:

    def __init__(self,
                 distance=100.0,
                 final_time=10.0,
                 initial_state=(0.0, 0.0, 0.0),
                 final_velocity_constraint=None,
                 velocity_limit=None,
                 slopes=None):

        self.distance = distance
        self.final_time = final_time
        self.initial_state = initial_state
        self.final_velocity_constraint = final_velocity_constraint
        self.velocity_limit = velocity_limit
        self.slopes = slopes if slopes is not None else [(0, 0)]

    def get_slope_at_position(self, position):
        current_slope = 0.0
        for pos, angle in self.slopes:
            if position >= pos:
                current_slope = angle
        return current_slope


class VehicleDynamics:

    def __init__(self, vehicle_params, problem_def=None):
        self.vp = vehicle_params
        self.problem = problem_def

    def control_law(self, x1: float, i_ref: float, u_prev: float) -> float:
        delta_half = self.vp.delta / 2.0

        if x1 > i_ref + delta_half:
            return -1.0
        elif x1 < i_ref - delta_half:
            return 1.0
        else:
            return u_prev

    def get_slope_force(self, x3: float) -> float:
        if self.problem is None:
            return 0.0

        theta_deg = self.problem.get_slope_at_position(x3)
        theta_rad = np.deg2rad(theta_deg)
        return self.vp.M * self.vp.g * np.sin(theta_rad)

    def derivatives(self, t: float, state: np.ndarray, u: float) -> np.ndarray:
        x0, x1, x2, x3 = state

        # energy derivative
        dx0_dt = u * x1 * self.vp.V_alim + self.vp.R_bat * (u**2) * (x1**2)

        # current derivative
        dx1_dt = (u * self.vp.V_alim - self.vp.R_m * x1 - self.vp.K_m * x2) / self.vp.L_m

        # angular velocity derivative
        velocity_linear = x2 * self.vp.r / self.vp.K_r

        # resistance forces
        friction_force = self.vp.M * self.vp.g * self.vp.K_f
        drag_force = 0.5 * self.vp.rho * self.vp.S * self.vp.C_x * (velocity_linear**2)

        # slope force
        slope_force = self.get_slope_force(x3)

        # total force and acceleration
        motor_torque = self.vp.K_m * x1
        resistance_term = (self.vp.r / self.vp.K_r) * (friction_force + drag_force + slope_force)

        dx2_dt = (motor_torque - resistance_term) / self.vp.J

        # position derivative
        dx3_dt = x2 * self.vp.r / self.vp.K_r

        return np.array([dx0_dt, dx1_dt, dx2_dt, dx3_dt])

    def euler_step(self, state: np.ndarray, u: float, t: float, dt: float) -> np.ndarray:
        return state + dt * self.derivatives(t, state, u)

    def rk2_step(self, state: np.ndarray, u: float, t: float, dt: float) -> np.ndarray:
        k1 = self.derivatives(t, state, u)
        k2 = self.derivatives(t + dt/2, state + dt*k1/2, u)
        return state + dt * k2

    def rk4_step(self, state: np.ndarray, u: float, t: float, dt: float) -> np.ndarray:
        k1 = self.derivatives(t, state, u)
        k2 = self.derivatives(t + dt/2, state + dt*k1/2, u)
        k3 = self.derivatives(t + dt/2, state + dt*k2/2, u)
        k4 = self.derivatives(t + dt, state + dt*k3, u)
        return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

    def simulate(self,
                 i_ref: float,
                 t0: float,
                 tf: float,
                 initial_state: np.ndarray,
                 dt: float = 1e-4,
                 method: str = 'RK4') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if method == 'Euler':
            step_func = self.euler_step
        elif method == 'RK2':
            step_func = self.rk2_step
        elif method == 'RK4':
            step_func = self.rk4_step
        else:
            raise ValueError(f"Unknown method: {method}")

        # Time array
        num_steps = int((tf - t0) / dt)
        t_array = np.linspace(t0, tf, num_steps + 1)

        # State history
        state_history = np.zeros((num_steps + 1, 4))
        state_history[0] = initial_state

        # Initial control
        u = 1.0

        # Integrate
        state = initial_state.copy()
        for i in range(num_steps):
            t = t_array[i]

            # Update control based on current
            u = self.control_law(state[1], i_ref, u)

            # Integration step
            state = step_func(state, u, t, dt)

            # Constrain current to bounds
            state[1] = np.clip(state[1], -self.vp.i_max, self.vp.i_max)

            state_history[i + 1] = state

        return state_history[-1], t_array, state_history


class DiscreteBox:

    def __init__(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        self.lower = np.array(lower_bounds)
        self.upper = np.array(upper_bounds)
        self.dimension = len(lower_bounds)

    def is_point(self) -> bool:
        return np.all(self.lower == self.upper)

    def midpoint(self) -> np.ndarray:
        return (self.lower + self.upper) / 2.0

    def width(self) -> np.ndarray:
        return self.upper - self.lower

    def largest_dimension(self) -> int:
        return np.argmax(self.width())

    def bisect(self, step_size: float) -> Tuple['DiscreteBox', 'DiscreteBox']:
        dim = self.largest_dimension()
        mid = (self.lower[dim] + self.upper[dim]) / 2.0

        mid_discrete = np.round(mid / step_size) * step_size

        lower1 = self.lower.copy()
        upper1 = self.upper.copy()
        upper1[dim] = mid_discrete

        lower2 = self.lower.copy()
        upper2 = self.upper.copy()
        lower2[dim] = mid_discrete

        return DiscreteBox(lower1, upper1), DiscreteBox(lower2, upper2)


class BranchAndBound:
    def __init__(self,
                 vehicle_dynamics,
                 vehicle_params,
                 opt_config,
                 problem_def):

        self.vd = vehicle_dynamics
        self.vp = vehicle_params
        self.config = opt_config
        self.problem = problem_def

        self.P = int(problem_def.final_time / opt_config.st_tf)

        self.iterations = 0
        self.nodes_explored = 0

        self.best_energy = np.inf
        self.best_solution = None
        self.best_final_state = None

    def evaluate_solution(self, i_ref_sequence: np.ndarray) -> Tuple[float, np.ndarray]:
        state = np.array([0.0,
                         self.problem.initial_state[0],
                         self.problem.initial_state[1],
                         self.problem.initial_state[2]])

        sample_duration = self.problem.final_time / self.P

        for k in range(self.P):
            i_ref = i_ref_sequence[k]
            t0 = k * sample_duration
            tf = (k + 1) * sample_duration

            final_state, _, _ = self.vd.simulate(
                i_ref, t0, tf, state,
                dt=self.config.integration_step,
                method=self.config.integration_method
            )

            state = final_state

        return state[0], state

    def solve(self,
             initial_box: Optional[DiscreteBox] = None,
             verbose: bool = True,
             max_iterations: int = 100) -> Dict:

        start_time = time.time()

        if initial_box is None:
            lower = np.ones(self.P) * (-self.vp.i_max)
            upper = np.ones(self.P) * self.vp.i_max
            initial_box = DiscreteBox(lower, upper)

        box_list = [initial_box]

        while box_list:
            self.iterations += 1

            current_box = box_list.pop(0)

            box1, box2 = current_box.bisect(self.config.s)

            for box in [box1, box2]:
                self.nodes_explored += 1

                mid = box.midpoint()
                i_ref_seq = np.round(mid / self.config.s) * self.config.s

                try:
                    energy, final_state = self.evaluate_solution(i_ref_seq)

                    position = final_state[3]

                    if (abs(position - self.problem.distance) < 5.0 and
                        energy < self.best_energy):
                        self.best_energy = energy
                        self.best_solution = i_ref_seq.copy()
                        self.best_final_state = final_state.copy()

                        if verbose:
                            print(f"Iter {self.iterations}: Best energy = {energy:.2f} J, "
                                  f"position = {position:.2f} m")

                except Exception as e:
                    if verbose:
                        print(f"Warning: Evaluation failed - {e}")
                    continue

                if not box.is_point():
                    box_list.append(box)

                if self.iterations >= max_iterations:
                    if verbose:
                        print(f"\nReached iteration limit ({max_iterations})")
                    break

            if self.iterations >= max_iterations:
                break

        elapsed_time = time.time() - start_time

        return {
            'energy': self.best_energy,
            'solution': self.best_solution,
            'final_state': self.best_final_state,
            'iterations': self.iterations,
            'nodes_explored': self.nodes_explored,
            'time': elapsed_time
        }

def velocity_to_kmh(x2_rad_s: float, vehicle_params) -> float:
    return x2_rad_s * vehicle_params.r / vehicle_params.K_r * 3.6


def kmh_to_angular(v_kmh: float, vehicle_params) -> float:
    return v_kmh / 3.6 * vehicle_params.K_r / vehicle_params.r


def refine_solution(initial_solution: np.ndarray,
                   bba: BranchAndBound,
                   step_reduction: int = 2,
                   range_factor: float = 20.0) -> Dict:

    # Create refined search space
    lower = np.maximum(initial_solution - range_factor, -bba.vp.i_max)
    upper = np.minimum(initial_solution + range_factor, bba.vp.i_max)

    refined_box = DiscreteBox(lower, upper)

    # Reduce step size
    original_step = bba.config.s
    bba.config.s = original_step // step_reduction

    # Solve with refined parameters
    result = bba.solve(initial_box=refined_box)

    # Restore original step
    bba.config.s = original_step

    return result


def plot_trajectory(result, vd, problem, vp, save_path='trajectory.png'):

    if result['solution'] is None:
        print("No valid solution to plot")
        return

    i_ref_seq = result['solution']
    P = len(i_ref_seq)
    sample_duration = problem.final_time / P

    state = np.array([0.0,
                     problem.initial_state[0],
                     problem.initial_state[1],
                     problem.initial_state[2]])

    time_points = [0]
    energy_history = [0]
    current_history = [problem.initial_state[0]]
    velocity_history = [0]
    position_history = [0]

    for k in range(P):
        i_ref = i_ref_seq[k]
        t0 = k * sample_duration
        tf_sample = (k + 1) * sample_duration

        final_state, t_array, state_hist = vd.simulate(
            i_ref, t0, tf_sample, state,
            dt=0.01,
            method='RK4'
        )

        for i in range(1, len(t_array)):
            time_points.append(t_array[i])
            energy_history.append(state_hist[i, 0])
            current_history.append(state_hist[i, 1])
            velocity_history.append(velocity_to_kmh(state_hist[i, 2], vp))
            position_history.append(state_hist[i, 3])

        state = final_state

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('EV Energy Optimization - Trajectory Visualization',
                 fontsize=14, fontweight='bold')

    axes[0, 0].plot(time_points, current_history, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Current (A)')
    axes[0, 0].set_title('Motor Current')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(time_points, velocity_history, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity (km/h)')
    axes[0, 1].set_title('Vehicle Velocity')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(time_points, energy_history, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Energy (J)')
    axes[1, 0].set_title('Energy Consumed')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(time_points, position_history, 'm-', linewidth=2)
    axes[1, 1].axhline(y=problem.distance, color='k', linestyle='--', label='Target')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Position (m)')
    axes[1, 1].set_title('Vehicle Position')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Trajectory visualization saved to: {save_path}")
    plt.close()


def example_1_flat_travel():
    print("=" * 70)
    print("Example 1: 100m flat travel, free final velocity")
    print("=" * 70)

    # Setup
    vp = VehicleParameters()
    config = OptimizationConfig()
    config.s = 10
    config.st_tf = 2.0

    problem = ProblemDefinition(
        distance=100.0,
        final_time=10.0,
        initial_state=(0.0, 0.0, 0.0),
        final_velocity_constraint=None
    )

    # Create dynamics
    vd = VehicleDynamics(vp, problem)

    # Create optimizer
    bba = BranchAndBound(vd, vp, config, problem)

    # Solve
    print("\nSolving optimization problem...")
    result = bba.solve(verbose=True, max_iterations=100)

    # Display results
    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    print(f"Minimum Energy: {result['energy']:.2f} J")
    print(f"Reference Current Sequence: {result['solution']}")
    if result['final_state'] is not None:
        print(f"Final Position: {result['final_state'][3]:.2f} m")
        print(f"Final Velocity: {velocity_to_kmh(result['final_state'][2], vp):.2f} km/h")
    print(f"Iterations: {result['iterations']}")
    print(f"Time: {result['time']:.2f} s")
    print("=" * 70)

    return result, vp, config, problem, vd


def example_2_with_slope():
    print("\n\n" + "=" * 70)
    print("Example 2: 100m travel with +3° slope at 50m")
    print("=" * 70)

    # Setup
    vp = VehicleParameters()
    config = OptimizationConfig()
    config.s = 10
    config.st_tf = 2.0
    config.acceleration_limit = 0.3

    problem = ProblemDefinition(
        distance=100.0,
        final_time=10.0,
        initial_state=(0.0, 0.0, 0.0),
        final_velocity_constraint=None,
        slopes=[(0, 0), (50, 3)]
    )

    # Create dynamics with slope
    vd = VehicleDynamics(vp, problem)

    # Create optimizer
    bba = BranchAndBound(vd, vp, config, problem)

    # Solve
    print("\nSolving optimization problem with slope...")
    result = bba.solve(verbose=True, max_iterations=100)

    # Display results
    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    print(f"Minimum Energy: {result['energy']:.2f} J")
    print(f"Reference Current Sequence: {result['solution']}")
    if result['final_state'] is not None:
        print(f"Final Position: {result['final_state'][3]:.2f} m")
        print(f"Final Velocity: {velocity_to_kmh(result['final_state'][2], vp):.2f} km/h")
    print(f"Iterations: {result['iterations']}")
    print(f"Time: {result['time']:.2f} s")
    print("=" * 70)

    return result, vp, config, problem, vd


def main():
    print("\n" + "=" * 70)
    print(" Electric Vehicle Energy Optimization")
    print(" Implementation of Branch and Bound Algorithm")
    print("=" * 70)

    # Run Example 1
    result1, vp1, config1, problem1, vd1 = example_1_flat_travel()

    # Run Example 2
    result2, vp2, config2, problem2, vd2 = example_2_with_slope()

    # Visualize results if solution found
    if result1['solution'] is not None:
        print("\nGenerating visualization for Example 1...")
        plot_trajectory(result1, vd1, problem1, vp1, 'trajectory_example1.png')

    if result2['solution'] is not None:
        print("Generating visualization for Example 2...")
        plot_trajectory(result2, vd2, problem2, vp2, 'trajectory_example2.png')

    print("\n\nAll examples completed successfully!")


if __name__ == "__main__":
    main()

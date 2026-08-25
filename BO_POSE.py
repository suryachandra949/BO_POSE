"""
Install:
    pip install torch botorch gpytorch

For AMD/ROCm PyTorch, install the correct PyTorch ROCm wheel first,
then install botorch and gpytorch.

Run:
    python two_gp_bo.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from botorch.acquisition.analytic import (
    LogProbabilityOfImprovement,
    ProbabilityOfImprovement,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf_discrete
from gpytorch.mlls import ExactMarginalLogLikelihood

import math
import numpy as np


#Mneme imports
import argparse
import sys
import statistics

from mneme.async_executor import AsyncReplayExecutor
from mneme.mneme_types import ExperimentConfiguration
from mneme.recorded_execution import RecordedExecution
from mneme.tuning.search_space import (
    CategoricalParam,
    IntRangeParam,
    SearchSpace,
)

# ============================================================
# GLOBAL SETTINGS
# ============================================================

DTYPE = torch.double

# ROCm-enabled PyTorch also uses the "cuda" device API.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of tunable parameters.
D = 2

# Number of initial measurements before BO starts.
N_INITIAL = 2 * D

# Number of BO iterations.
N_ITERATIONS = 20


# Variables for POSE
pmin = 40
pmax = 300
alpha = 1
beta = 1
metric = "EDS"
n = 1
m = 1



# ============================================================
# REAL APPLICATION HOOK
# ============================================================

BLOCK_OPTIONS = list(range(64, 1025, 64))

PASS_OPTIONS = [
    "default<O1>",
    "default<O2>",
    "default<O3>",
    "default<Os>",
    "default<Oz>",
]

# ============================================================
# DISCRETE BO SEARCH SPACE
# ============================================================

choices = []

for block_idx in range(len(BLOCK_OPTIONS)):

    for pass_idx in range(len(PASS_OPTIONS)):

        # Normalized representation of block choice.
        block_value = (
            block_idx
            / (len(BLOCK_OPTIONS) - 1)
        )

        # Normalized representation of pass choice.
        pass_value = (
            pass_idx
            / (len(PASS_OPTIONS) - 1)
        )

        choices.append(
            [
                block_value,
                pass_value,
            ]
        )


CHOICES = torch.tensor(
    choices,
    dtype=DTYPE,
    device=DEVICE,
)


def select_option(value, options):
    """
    Convert a normalized BO value in [0, 1]
    into one of the discrete Mneme options.
    """

    index = round(
        float(value) * (len(options) - 1)
    )

    index = max(
        0,
        min(index, len(options) - 1),
    )

    return options[index]


def decode_candidate(x: torch.Tensor) -> dict:
    """
    Decode one normalized BoTorch candidate.

    x[0] -> block_dim_x
    x[1] -> LLVM pass pipeline
    """

    x = x.detach().cpu()

    return {
        "block_dim_x": select_option(
            x[0].item(),
            BLOCK_OPTIONS,
        ),

        "passes": select_option(
            x[1].item(),
            PASS_OPTIONS,
        ),
    }


def run_application(
    X: torch.Tensor,
    executor: AsyncReplayExecutor,
    space,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate BoTorch candidates using real Mneme replay.

    X has shape:
        [number_of_candidates, D]

    Returns:
        runtime_Y : [number_of_candidates, 1]
        energy_Y  : [number_of_candidates, 1]
    """

    runtimes = []
    energies = []

    for i in range(X.shape[0]):

        # ====================================================
        # 1. Decode BoTorch candidate
        # ====================================================

        params = decode_candidate(
            X[i]
        )

        print()
        print("=" * 60)
        print("Evaluating BO candidate")
        print("=" * 60)

        print(
            f"block_dim_x = {params['block_dim_x']}"
        )

        print(
            f"passes      = {params['passes']}"
        )

        # ====================================================
        # 2. Create Mneme ExperimentConfiguration
        # ====================================================

        config = space.derived(
            params
        )

        if not config.is_valid():
            raise RuntimeError(
                f"Invalid Mneme configuration: {params}"
            )

        # ====================================================
        # 3. Run actual Mneme replay
        # ====================================================

        result = executor.evaluate(
            config
        )

        # ====================================================
        # 4. Verify replay
        # ====================================================

        if not result.verified:
            raise RuntimeError(
                f"Mneme replay failed for {params}. "
                f"Error: {result.error}"
            )

        # ====================================================
        # 5. Runtime
        # ====================================================

        runtime = statistics.mean(
            result.exec_time
        )

        # ====================================================
        # 6. Energy
        # ====================================================

        energy_mj = getattr(
            result,
            "mean_energy_mj",
            None,
        )

        if energy_mj is None or energy_mj == []:
            raise RuntimeError(
                "Mneme did not return mean_energy_mj"
            )

        # Handle list/tuple if necessary
        if isinstance(
            energy_mj,
            (list, tuple),
        ):
            energy_mj = statistics.mean(
                energy_mj
            )

        # Mneme mJ -> Joules
        energy_j = float(
            energy_mj
        ) / 1000.0

        # ====================================================
        # 7. Calculate EDP for information
        # ====================================================

        edp = (
            runtime
            * energy_j
        )

        print(
            f"Runtime = {runtime:.6f}"
        )

        print(
            f"Energy  = {energy_j:.6f} J"
        )

        print(
            f"EDP     = {edp:.6f}"
        )

        # ====================================================
        # 8. Store observations
        # ====================================================

        runtimes.append(
            runtime
        )

        energies.append(
            energy_j
        )

    # ========================================================
    # Convert measurements to BoTorch tensors
    # ========================================================

    runtime_Y = torch.tensor(
        runtimes,
        dtype=DTYPE,
        device=DEVICE,
    ).unsqueeze(-1)

    energy_Y = torch.tensor(
        energies,
        dtype=DTYPE,
        device=DEVICE,
    ).unsqueeze(-1)

    return (
        runtime_Y,
        energy_Y,
    )

# ============================================================
# MODEL FITTING
# ============================================================

def fit_gp(train_X: torch.Tensor, train_Y: torch.Tensor) -> SingleTaskGP:
    """
    Fit a single-output GP.

    Inputs are normalised and the output is standardised internally.
    """

    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=Normalize(d=D),
        outcome_transform=Standardize(m=1),
    ).to(DEVICE)

    mll = ExactMarginalLogLikelihood(
        model.likelihood,
        model,
    )

    fit_gpytorch_mll(mll)

    model.eval()
    return model


# ============================================================
# ACQUISITION
# ============================================================

@dataclass
class Candidate:
    x: torch.Tensor
    probability: float
    target: float
    label: str


def propose_probability_target(
    model: SingleTaskGP,
    target: float,
    label: str,
    X_avoid: torch.Tensor | None = None,
) -> Candidate:
    """
    Find x maximising P(f(x) <= target).

    Since runtime and energy are both minimised, maximize=False.

    LogProbabilityOfImprovement is used for numerical stability while
    optimising. ProbabilityOfImprovement is then used only to report
    an interpretable probability in [0, 1].
    """

    target_tensor = torch.tensor(
        target,
        dtype=DTYPE,
        device=DEVICE,
    )

    log_pi = LogProbabilityOfImprovement(
        model=model,
        best_f=target_tensor,
        maximize=False,
    )

    candidate, _ = optimize_acqf_discrete(
        acq_function=log_pi,
        q=1,
        choices=CHOICES,
        X_avoid=X_avoid,
    )

    pi = ProbabilityOfImprovement(
        model=model,
        best_f=target_tensor,
        maximize=False,
    )

    with torch.no_grad():
        # PI expects shape batch x q x d.
        probability = pi(candidate.unsqueeze(0)).item()

    return Candidate(
        x=candidate.detach(),
        probability=probability,
        target=float(target),
        label=label,
    )


# ============================================================
# HELPERS
# ============================================================
def parse_args(argv: list[str]) -> argparse.Namespace:

    p = argparse.ArgumentParser(
        prog="bo-pose",
        description="Run POSE-guided Bayesian optimisation on a Mneme recorded kernel.",
    )

    p.add_argument(
        "--record-db",
        required=True,
        help="Path to the Mneme recording database JSON file.",
    )

    p.add_argument(
        "--record-id",
        required=True,
        help="Kernel instance id (dynamic hash) to operate on.",
    )

    return p.parse_args(argv)


def initial_design(n: int) -> torch.Tensor:
    """
    Randomly select n unique configurations
    from the valid discrete Mneme space.
    """

    if n > CHOICES.shape[0]:
        raise ValueError(
            "Requested more initial points "
            "than available configurations."
        )

    indices = torch.randperm(
        CHOICES.shape[0],
        device=DEVICE,
    )[:n]

    return CHOICES[indices].clone()


def same_candidate(a: torch.Tensor,b: torch.Tensor) -> bool:

    return torch.equal(a, b)


def format_x(X: torch.Tensor) -> str:
    values = X.squeeze(0).detach().cpu().tolist()
    return "[" + ", ".join(f"{v:.4f}" for v in values) + "]"



def select_initial_reference_by_metric(
    train_X: torch.Tensor,
    runtime_Y: torch.Tensor,
    energy_Y: torch.Tensor,
    metric: str,
    alpha: float,
    beta: float,
    n: float,
    m: float,
) -> tuple[int, float, float, float]:
    """
    Select the measured sample with the lowest requested metric.

    Supported metrics:

        EDP:
            energy^m * runtime^n

        EDS:
            alpha * energy + beta * runtime

        EDD:
            sqrt(
                (alpha * energy)^2
                + (beta * runtime)^2
            )

    Returns
    -------
    selected_index
    selected_runtime
    selected_energy
    selected_metric
    """

    runtime = runtime_Y[:, 0]
    energy = energy_Y[:, 0]

    # ========================================================
    # Calculate metric for every measured configuration
    # ========================================================

    if metric == "EDP":

        metric_values = (
            energy.pow(m)
            * runtime.pow(n)
        )

    elif metric == "EDS":

        metric_values = (
            alpha * energy
            + beta * runtime
        )

    elif metric == "EDD":

        metric_values = torch.sqrt(
            (alpha * energy).pow(2)
            + (beta * runtime).pow(2)
        )

    else:
        raise ValueError(
            f"Unsupported metric: {metric}. "
            "Use 'EDP', 'EDS', or 'EDD'."
        )

    # ========================================================
    # Find lowest metric
    # ========================================================

    selected_index = torch.argmin(
        metric_values
    ).item()

    selected_runtime = runtime[
        selected_index
    ].item()

    selected_energy = energy[
        selected_index
    ].item()

    selected_metric = metric_values[
        selected_index
    ].item()

    # ========================================================
    # Print all measured samples
    # ========================================================

    print("=" * 88)
    print("MEASURED SAMPLES")
    print(f"Selection metric: {metric}")
    print("=" * 88)

    for i in range(train_X.shape[0]):

        marker = (
            f"  <-- selected (lowest {metric})"
            if i == selected_index
            else ""
        )

        print(
            f"[{i + 1:2d}] "
            f"x={format_x(train_X[i:i+1])}  "
            f"runtime={runtime[i].item():.6f} s  "
            f"energy={energy[i].item():.6f} J  "
            f"{metric}={metric_values[i].item():.6f}"
            f"{marker}"
        )

    # ========================================================
    # Print selected reference
    # ========================================================

    print()
    print(f"{metric}-selected reference")

    print(
        "  sample :",
        selected_index + 1,
    )

    print(
        "  x      :",
        format_x(
            train_X[
                selected_index:selected_index + 1
            ]
        ),
    )

    print(
        "  runtime:",
        f"{selected_runtime:.6f} s",
    )

    print(
        "  energy :",
        f"{selected_energy:.6f} J",
    )

    print(
        f"  {metric:<7}:",
        f"{selected_metric:.6f}",
    )

    print()

    return (
        selected_index,
        selected_runtime,
        selected_energy,
        selected_metric,
    )


def compute_intersections(point, pmin, pmax, alpha, beta, metric,n,m):

    #Metric Calculation

    energy_value = point["energy"] 
    runtime_value = point["runtime"]
    power_value = energy_value / runtime_value

    if metric == "EDD":

      EDD_metric = math.sqrt(((alpha*energy_value)**2) + ((beta) * runtime_value)**2)

      EDD_max_runtime = math.sqrt((EDD_metric * EDD_metric) / (((alpha*pmax)**2) + ((beta)**2)))
      EDD_max_energy = pmax * EDD_max_runtime

      EDD_min_runtime = math.sqrt((EDD_metric * EDD_metric) / (((alpha*pmin)**2) + ((beta)**2)))
      EDD_min_energy = pmin * EDD_min_runtime

      Point_D_runtime = runtime_value
      Point_D_energy = runtime_value * pmin
 
      Point_C_runtime = runtime_value * math.sqrt(((alpha*pmin)**2 + beta**2)/(((alpha*power_value)**2 + beta**2)))
      Point_C_energy = pmin * Point_C_runtime

      #Metric Calculations
      EDD_metric_limit = math.sqrt((alpha*Point_C_energy)**2 + (beta * Point_C_runtime)**2)

      Point_A_runtime = math.sqrt((EDD_metric_limit * EDD_metric_limit) / (((alpha*pmax)**2) + (beta**2)))
      Point_A_energy = pmax * Point_A_runtime 

    elif metric == "EDS":

      EDD_metric = (alpha*energy_value) + ((beta) * runtime_value)

      EDD_max_runtime = ((EDD_metric)/ ((alpha*pmax) + beta))
      EDD_max_energy = pmax * EDD_max_runtime

      EDD_min_runtime = ((EDD_metric) / ((alpha*pmin) + beta))
      EDD_min_energy = pmin * EDD_min_runtime

      Point_D_runtime = runtime_value
      Point_D_energy = runtime_value * pmin
 
      Point_C_runtime = runtime_value * (((alpha*pmin) + beta)/((alpha*power_value) + beta))
      Point_C_energy = pmin * Point_C_runtime

      #Metric Calculations
      EDD_metric_limit = ((alpha*Point_C_energy) + (beta * Point_C_runtime))

      Point_A_runtime = ((EDD_metric_limit) / ((alpha*pmax) + beta))
      Point_A_energy = pmax * Point_A_runtime   

    elif metric == "EDP":

      EDD_metric = (energy_value**m) *  (runtime_value**n)

      EDD_max_runtime = (EDD_metric / pmax**m)**(1/(n+m))
      EDD_max_energy = pmax * EDD_max_runtime

      EDD_min_runtime =  (EDD_metric / pmin**m)**(1/(n+m))
      EDD_min_energy = pmin * EDD_min_runtime

      Point_D_runtime = runtime_value
      Point_D_energy = runtime_value * pmin
 
      Point_C_runtime = runtime_value *((pmin/power_value) ** (m/(m+n)))
      Point_C_energy = pmin * Point_C_runtime

      #Metric Calculations
      EDD_metric_limit = (Point_C_energy**m) * (Point_C_runtime**n)

      Point_A_runtime = (EDD_metric_limit / pmax**m)**(1/(n+m))
      Point_A_energy = pmax * Point_A_runtime   

         

    intersections = []


    intersections.append({
        "x": EDD_max_runtime,
        "y": EDD_max_energy,
        "label": "B",
        "line": "pmax"
    })

    intersections.append({
        "x": EDD_min_runtime,
        "y": EDD_min_energy,
        "label": "E",
        "line": "pmin"
    })

    intersections.append({
        "x": Point_A_runtime,
        "y": Point_A_energy,
        "label": "A",
        "line": "pmax"
    })

    intersections.append({
        "x": Point_C_runtime,
        "y": Point_C_energy,
        "label": "C",
        "line": "pmin"
    })

    intersections.append({
        "x": Point_D_runtime,
        "y": Point_D_energy,
        "label": "D",
        "line": "pmin"
    })

    return intersections,EDD_metric,EDD_metric_limit,runtime_value,energy_value

def POSE_compute(point, pmin, pmax, alpha, beta, metric, n, m):
    intersections,EDD_metric,EDD_metric_limit,code_runtime,code_energy = compute_intersections(point, pmin, pmax,alpha,beta,metric,n,m)

    #metrics list
    B = []
    A = []
    C = []
    E = []
    # Extract intersection energies
    e_edd_vals =[]
    e_limit_vals =[]
    D_line_runtime=[]
    D_line_energy=[]
    D_line_energy.append(code_energy)
    D_line_runtime.append(code_runtime)
    power_value = code_energy/code_runtime
    for pt in intersections:
        if pt["label"] == "B" or pt["label"] == "E":
            e_edd_vals.append(pt["y"])
            if pt["label"] == "B":
                B.append(pt["x"])
                B.append(pt["y"])
            else:
                E.append(pt["x"])
                E.append(pt["y"])
        elif pt["label"] == "A" or  pt["label"] == "C":
            e_limit_vals.append(pt["y"])
            if pt["label"] == "C":
                Point_C_runtime = pt["x"]
                C.append(pt["x"])
                C.append(pt["y"])
            else:
                A.append(pt["x"])
                A.append(pt["y"])
        elif pt["label"] == "D":
            D_line_runtime.append(pt["x"])
            D_line_energy.append(pt["y"])

    return A,B,C,D_line_runtime,D_line_energy,E

def update_theta_point(metric,alpha,beta,n,m,measured_runtime,measured_energy,A):

    if metric == "EDP":
        measured_metric = (measured_energy)**m * (measured_runtime)**n
        metric_A = (A[1])**m * (A[0])**n

    elif metric == "EDS":
        measured_metric = (alpha * measured_energy) + (beta * measured_runtime)
        metric_A = (alpha * A[1]) + (beta * A[0])
    else:
        measured_metric = math.sqrt((alpha * measured_energy)**2 + (beta * measured_runtime)**2)
        metric_A = math.sqrt((alpha * A[1])**2 + (beta * A[0])**2)


    if measured_metric < metric_A:
        return True
    else:
        return False


class EntireSpace(SearchSpace):
    """
    Example SearchSpace for tuning a recorded kernel.

    This SearchSpace:
      - Uses the recorded grid/block dims as fixed reference values where needed.
      - Exposes tunable parameters (block_dim_x and pass pipeline).
      - Produces an ExperimentConfiguration via derived(params).
    """

    def __init__(self, recorded_kernel: RecordedExecution.KernelInstance):
        self.grid_dim_x = recorded_kernel.grid_dim.x
        self.grid_dim_y = recorded_kernel.grid_dim.y
        self.grid_dim_z = recorded_kernel.grid_dim.z

        self.block_dim_x = recorded_kernel.block_dim.x
        self.block_dim_y = recorded_kernel.block_dim.y
        self.block_dim_z = recorded_kernel.block_dim.z

        self.shared_mem = recorded_kernel.shared_mem

        self._search_space = {
            "block_dim_x": IntRangeParam("block_dim_x", low=64, high=1024, step=64),
            "passes": CategoricalParam(
                "passes",
                [
                    "default<O1>",
                    "default<O2>",
                    "default<O3>",
                    "default<Os>",
                    "default<Oz>",
                ],
            ),
        }

    def dimensions(self):
        return self._search_space

    def derived(self, params) -> ExperimentConfiguration:
        derived_config = {
            "block": {
                "x": params["block_dim_x"],
                "y": self.block_dim_y,
                "z": self.block_dim_z,
            },
            "grid": {"x": self.grid_dim_x, "y": self.grid_dim_y, "z": self.grid_dim_z},
            "shared_mem": self.shared_mem,
            "passes": params["passes"],
        }
        return ExperimentConfiguration.from_dict(derived_config)

    def constraints(self, params):
        return True

    def baseline(self) -> ExperimentConfiguration:
        return ExperimentConfiguration.from_dict(
            {
                "block": {
                    "x": self.block_dim_x,
                    "y": self.block_dim_y,
                    "z": self.block_dim_z,
                },
                "grid": {
                    "x": self.grid_dim_x,
                    "y": self.grid_dim_y,
                    "z": self.grid_dim_z,
                },
                "shared_mem": self.shared_mem,
            }
        )

# ============================================================
# MAIN BO LOOP
# ============================================================

def main(argv: list[str]) -> int:
    print("=" * 72)
    print("Two-GP Bayesian optimisation")
    print("=" * 72)
    print(f"Device              : {DEVICE}")
    print(f"Dimensions          : {D}")
    print(f"Initial observations: {N_INITIAL}")
    print(f"BO iterations       : {N_ITERATIONS}")
    print()

    args = parse_args(argv)

    record_db = args.record_db
    record_id = args.record_id

    rec = RecordedExecution.from_json(
        record_db
    )

    kernel = rec[record_id]

    space = EntireSpace(
        kernel
    )

    executor = AsyncReplayExecutor(
        record_db=record_db,
        record_id=record_id,
        iterations=5,
        results_db_dir="./results",
        num_workers=1,
    )

    try:
        # --------------------------------------------------------
        # 1. Initial experiments
        # --------------------------------------------------------

        train_X = initial_design(N_INITIAL)
        runtime_Y, energy_Y = run_application(train_X,executor,space)

        runtime_Y = runtime_Y.to(dtype=DTYPE, device=DEVICE)
        energy_Y = energy_Y.to(dtype=DTYPE, device=DEVICE)


        # --------------------------------------------------------
        # Select ONE initial reference point using minimum of the metric required:
        #
        #     EDP or EDS or EDD
        #
        # --------------------------------------------------------
        (
            reference_idx,
            selected_runtime_reference,
            selected_reference_energy,
            selected_reference_metric,
        ) = select_initial_reference_by_metric(
            train_X,
            runtime_Y,
            energy_Y,
            metric,
            alpha,
            beta,
            n,
            m,
        )
        
        point = {}
        point["energy"] = selected_reference_energy
        point["runtime"] = selected_runtime_reference 
        A,B,C,D_runtime,D_energy,E = POSE_compute(point, pmin, pmax, alpha, beta, metric, n, m)



        # --------------------------------------------------------
        # 2. BO iterations
        # --------------------------------------------------------

        for iteration in range(1, N_ITERATIONS + 1):
            print("=" * 72)
            print(f"ITERATION {iteration}")
            print("=" * 72)

            # Fit TWO independent GPs from the same evaluated configurations.
            runtime_gp = fit_gp(train_X, runtime_Y)
            energy_gp = fit_gp(train_X, energy_Y)

            # checking for points in strong runtime region
            runtime_target = A[0]

            runtime_candidate = propose_probability_target(
                model=runtime_gp,
                target=runtime_target,
                label="runtime",
                X_avoid=train_X,
            )

            new_X = runtime_candidate.x

            if runtime_candidate.probability < 0.5 :

                runtime_target = C[0]

                runtime_candidate = propose_probability_target(
                    model=runtime_gp,
                    target=runtime_target,
                    label="runtime",
                    X_avoid=train_X,
                )

                new_X = runtime_candidate.x

                if runtime_candidate.probability < 0.5 :


                    (runtime_reference_idx,
                    selected_runtime_reference,
                    selected_reference_energy,
                    selected_reference_edp,) = select_initial_reference_by_edp(train_X,runtime_Y, energy_Y,)

                    point["energy"] = selected_reference_energy
                    point["runtime"] = selected_runtime_reference 

                    A,B,C,D_runtime,D_energy,E = POSE_compute(point, pmin, pmax, alpha, beta, metric, n, m)
                    runtime_target = B[0]
                    energy_target = C[1]

                    runtime_candidate = propose_probability_target(
                        model=runtime_gp,
                        target=runtime_target,
                        label="runtime",
                        X_avoid=train_X,
                    )

                    energy_candidate = propose_probability_target(
                        model=energy_gp,
                        target=energy_target,
                        label="energy",
                        X_avoid=train_X,

                    )

                    if same_candidate(runtime_candidate.x,energy_candidate.x):
                        new_X = runtime_candidate.x

                    else:
                        new_X = torch.cat([runtime_candidate.x,energy_candidate.x],dim=0)






            new_runtime_Y, new_energy_Y = run_application(new_X,executor,space)

            new_runtime_Y = new_runtime_Y.to(
                dtype=DTYPE,
                device=DEVICE,
            )
            new_energy_Y = new_energy_Y.to(
                dtype=DTYPE,
                device=DEVICE,
            )

            for i in range(new_X.shape[0]):
            
                measured_runtime = new_runtime_Y[i, 0].item()
                measured_energy = new_energy_Y[i, 0].item()

                update_theta = update_theta_point(metric,alpha,beta,n,m,measured_runtime,measured_energy,A)

                if update_theta :
                
                    point["energy"] = measured_energy
                    point["runtime"] = measured_runtime 
                    A,B,C,D_runtime,D_energy,E = POSE_compute(point, pmin, pmax, alpha, beta, metric, n, m)


            # ----------------------------------------------------
            #  Update BOTH GP datasets with ALL new experiments.
            # ----------------------------------------------------

            train_X = torch.cat(
                [train_X, new_X],
                dim=0,
            )

            runtime_Y = torch.cat(
                [runtime_Y, new_runtime_Y],
                dim=0,
            )

            energy_Y = torch.cat(
                [energy_Y, new_energy_Y],
                dim=0,
            )

    finally:

        executor.shutdown()

    return 0




if __name__ == "__main__":

    raise SystemExit(main(sys.argv[1:]))

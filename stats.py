from dataclasses import dataclass, field


@dataclass
class LinearSolveStats:
    time_invert_F: float = -1
    time_prepare_mass: float = -1
    time_prepare_momentum: float = -1
    time_prepare_solver: float = -1
    time_gmres: float = -1
    gmres_iters: int = -1
    time_solve_linear_system: float = -1
    matrix_id: str = ""
    rhs_id: str = ""
    petsc_converged_reason: int = -100
    num_sticking_sliding_open: tuple[int, int, int] = (-1, -1, -1)


@dataclass
class TimeStepStats:
    linear_solves: list[LinearSolveStats] = field(default_factory=list)

    @classmethod
    def from_json(cls, json: str):
        data = cls(**json)
        tmp = []
        for x in data.linear_solves:
            payload = {
                k: v for k, v in x.items() if k in LinearSolveStats.__dataclass_fields__
            }
            tmp.append(LinearSolveStats(**payload))
        data.linear_solves = tmp
        return data
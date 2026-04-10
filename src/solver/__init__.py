"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

from ._solver import BaseSolver
from .clas_solver import ClasSolver
from .pose_solver import PoseSolver
from .kpt_solver import KptSolver


from typing import Dict

TASKS: Dict[str, BaseSolver] = {
    "classification": ClasSolver,
    "pose_estimation": PoseSolver,
    "kpt_estimation": KptSolver,
}

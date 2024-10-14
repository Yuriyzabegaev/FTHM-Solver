import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Sequence

import matplotlib as mpl
import numpy as np
import porepy as pp
import scipy
import scipy.linalg
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from numpy.linalg import norm
from pyamg.krylov import gmres
from scipy.sparse import bmat
from scipy.sparse.linalg import LinearOperator  # , gmres, bicgstab
from tqdm import tqdm

from stats import LinearSolveStats

if TYPE_CHECKING:
    from block_matrix import SolveSchema, BlockMatrixStorage

from mat_utils import PetscGMRES, condest, eigs
from stats import TimeStepStats

BURBERRY = mpl.cycler(
    color=["#A70100", "#513819", "#956226", "#B8A081", "#747674", "#0D100E"]
)

# mpl.rcParams['axes.prop_cycle'] = BURBERRY


def trim_label(label: str) -> str:
    trim = 15
    if len(label) <= trim:
        return label
    return label[:trim] + "..."


def spy(mat, show=True, aspect: Literal["equal", "auto"] = "equal", marker=None):
    if marker is None:
        marker = "+"
        if max(*mat.shape) > 300:
            marker = ","
    plt.spy(mat, marker=marker, markersize=4, color="black", aspect=aspect)
    if show:
        plt.show()


def plot_diff(a, b, log=True):
    diff = a - b
    if log:
        diff = abs(diff)
        plt.yscale("log")
    plt.plot(diff)


def plot_jacobian(model, equations=None):
    if equations is None:
        equations = list(model.equation_system.equations.values())
    try:
        equations[0]
    except IndexError:
        equations = list(equations)

    ax = plt.gca()

    eq_labels = []
    eq_labels_pos = []
    y_offset = 0
    jac_list = []
    for i, eq in enumerate(equations):
        jac = eq.value_and_jacobian(model.equation_system).jac
        jac_list.append([jac])
        eq_labels.append(trim_label(eq.name))
        eq_labels_pos.append(y_offset + jac.shape[0] / 2)
        plt.axhspan(
            y_offset - 0.5, y_offset + jac.shape[0] - 0.5, facecolor=f"C{i}", alpha=0.3
        )
        y_offset += jac.shape[0]

    jac = bmat(jac_list)
    spy(jac, show=False)
    if len(eq_labels) == 1:
        ax.set_title(eq_labels[0])
    else:
        ax.yaxis.set_ticks(eq_labels_pos)
        ax.set_yticklabels(eq_labels, rotation=0)

    labels = []
    labels_pos = []
    for i, var in enumerate(model.equation_system.variables):
        dofs = model.equation_system.dofs_of([var])
        plt.axvspan(dofs[0] - 0.5, dofs[-1] + 0.5, facecolor=f"C{i}", alpha=0.3)
        labels_pos.append(np.average(dofs))
        labels.append(trim_label(var.name))

    ax.xaxis.set_ticks(labels_pos)
    ax.set_xticklabels(labels, rotation=45, ha="left")


def plot_mat(
    mat,
    log=True,
    show=True,
    threshold=1e-30,
    aspect: Literal["equal", "auto"] = "equal",
):
    mat = mat.copy()
    try:
        mat = mat.toarray()
    except AttributeError:
        pass

    mat[abs(mat) < threshold] = np.nan
    if log:
        mat = np.log10(abs(mat))

    plt.matshow(mat, fignum=0, aspect=aspect)
    plt.colorbar()
    if show:
        plt.show()


def plot_eigs(mat, label="", logx=False):
    eigs, _ = scipy.linalg.eig(mat.toarray())
    if logx:
        eigs.real = abs(eigs.real)
    plt.scatter(eigs.real, eigs.imag, label=label, marker=r"$\lambda$", alpha=0.5)
    plt.xlabel(r"Re($\lambda)$")
    plt.ylabel(r"Im($\lambda$)")
    plt.legend()
    plt.grid(True)
    if logx:
        plt.xscale("log")


def solve(
    mat,
    prec=None,
    rhs=None,
    label="",
    plot_residuals=False,
    tol=1e-10,
):
    residuals = []
    residual_vectors = []
    if rhs is None:
        rhs = np.ones(mat.shape[0])

    def callback(x):
        res = mat.dot(x) - rhs
        residual_vectors.append(res)
        residuals.append(float(norm(res)))

    if prec is not None:
        prec = LinearOperator(shape=prec.shape, matvec=prec.dot)

    restart = 50
    t0 = time.time()
    res, info = gmres(
        mat,
        rhs,
        M=prec,
        tol=tol,
        # atol=0,
        restrt=restart,
        callback=callback,
        # callback_type=callback_type,
        # maxiter=20,
        maxiter=20,
    )
    print("Solve", label, "took:", round(time.time() - t0, 2))

    linestyle = "-"
    if info != 0:
        linestyle = "--"

    plt.plot(residuals, label=label, marker=".", linestyle=linestyle)
    plt.yscale("log")
    plt.ylabel("pr. residual")
    plt.xlabel("gmres iter.")
    plt.grid(True)

    if plot_residuals:
        plt.figure()
        residual_vectors = np.array(residual_vectors)
        residual_vectors = abs(residual_vectors)

        # num = len(residual_vectors)
        # show_vectors = np.arange(0, num, num // 2)
        # for iter in show_vectors:
        #     plt.plot(residual_vectors[iter], label=iter, alpha=0.7)
        # plt.legend()
        plt.plot(residual_vectors[-1] / residual_vectors[0], alpha=0.7)
        plt.yscale("log")
    return np.array(residual_vectors)


def color_spy(block_mat, row_idx=None, col_idx=None, row_names=None, col_names=None):
    if row_idx is None:
        row_idx = list(range(block_mat.shape[0]))
    if col_idx is None:
        col_idx = list(range(block_mat.shape[1]))
    if row_names is None:
        row_names = row_idx
    if col_names is None:
        col_names = col_idx
    row_sep = [0]
    col_sep = [0]
    active_submatrices = []
    for i in row_idx:
        active_row = []
        for j in col_idx:
            submat = block_mat[i, j]
            active_row.append(submat)
            if i == row_idx[0]:
                col_sep.append(col_sep[-1] + submat.shape[1])
        row_sep.append(row_sep[-1] + submat.shape[0])
        active_submatrices.append(active_row)
    spy(bmat(active_submatrices), show=False)

    ax = plt.gca()
    row_label_pos = []
    for i in range(len(row_idx)):
        ystart, yend = row_sep[i : i + 2]
        row_label_pos.append(ystart + (yend - ystart) / 2)
        plt.axhspan(ystart - 0.5, yend - 0.5, facecolor=f"C{i}", alpha=0.3)
    ax.yaxis.set_ticks(row_label_pos)
    ax.set_yticklabels(row_names, rotation=0)

    col_label_pos = []
    for i in range(len(col_idx)):
        xstart, xend = col_sep[i : i + 2]
        col_label_pos.append(xstart + (xend - xstart) / 2)
        plt.axvspan(xstart - 0.5, xend - 0.5, facecolor=f"C{i}", alpha=0.3)
    ax.xaxis.set_ticks(col_label_pos)
    ax.set_xticklabels(col_names, rotation=0)


MARKERS = itertools.cycle(
    [
        "x",
        "+",
        # "o",
        # "v",
        # "<",
        # ">",
        # "^",
        "1",
        "2",
        "3",
        "4",
    ]
)


def solve_petsc(
    mat,
    prec=None,
    rhs=None,
    label="",
    logx_eigs=False,
    normalize_residual=False,
    tol=1e-10,
    pc_side: Literal["left", "right"] = "left",
    return_solution: bool = False,
    ksp_view: bool = False,
):
    if rhs is None:
        rhs = np.ones(mat.shape[0])
    gmres = PetscGMRES(mat, pc=prec, tol=tol, pc_side=pc_side)

    if ksp_view:
        gmres.ksp.view()

    t0 = time.time()
    sol = gmres.solve(rhs)
    print("Solve", label, "took:", round(time.time() - t0, 2))
    residuals = gmres.get_residuals()
    info = gmres.ksp.getConvergedReason()
    eigs = gmres.ksp.computeEigenvalues()

    rhs_norm = norm(rhs)
    res_norm = norm(mat @ sol - rhs)
    print("True residual decrease:", res_norm / rhs_norm)

    print("PETSc Converged Reason:", info)
    linestyle = "-"
    if info <= 0:
        linestyle = "--"
        if len(eigs) > 0:
            print("lambda min:", min(abs(eigs)))

    plt.gcf().set_size_inches(14, 4)

    # ax = plt.gca()
    ax = plt.subplot(1, 2, 1)
    if normalize_residual:
        residuals /= residuals[0]
    ax.plot(residuals, label=label, marker=".", linestyle=linestyle)
    ax.set_yscale("log")

    ksp_norm_type = gmres.ksp.getNormType()  # 1-prec, 2-unprec
    if ksp_norm_type == 2:
        ax.set_ylabel("true residual")
    elif ksp_norm_type == 1:
        ax.set_ylabel("preconditioned residual")
    ax.set_xlabel("gmres iter.")
    ax.grid(True)
    if label != "":
        ax.legend()
    ax.set_title("GMRES Convergence")

    ax = plt.subplot(1, 2, 2)
    if logx_eigs:
        eigs.real = abs(eigs.real)
    # ax.scatter(eigs.real, eigs.imag, label=label, marker="$\lambda$", alpha=0.9)
    ax.scatter(eigs.real, eigs.imag, label=label, alpha=1, s=300, marker=next(MARKERS))
    ax.set_xlabel(r"Re($\lambda)$")
    ax.set_ylabel(r"Im($\lambda$)")
    ax.grid(True)
    if label != "":
        ax.legend()
    if logx_eigs:
        plt.xscale("log")
    ax.set_title("Eigenvalues estimate")
    if return_solution:
        return sol


def get_gmres_iterations(x: Sequence[TimeStepStats]) -> list[float]:
    result = []
    for ts in x:
        for ls in ts.linear_solves:
            result.append(ls.krylov_iters)
    return result


def get_newton_iterations(x: Sequence[TimeStepStats]) -> list[float]:
    result = []
    for ts in x:
        result.append(len(ts.linear_solves))
    return result


def get_time_steps(x: Sequence[TimeStepStats]) -> list[float]:
    result = []
    for ts in x:
        result.append(ts.linear_solves[0].simulation_dt)
    return result


def get_F_cond(data: Sequence[TimeStepStats], model):
    res = []
    for i in range(sum(len(x.linear_solves) for x in data)):
        mat, rhs = load_matrix_rhs(data, i)
        sliced_mat = model.slice_jacobian(mat)
        res.append(condest(sliced_mat.F))
    return res


def get_S_Ap_cond(data: Sequence[TimeStepStats], model):
    res = []
    for i in range(sum(len(x.linear_solves) for x in data)):
        mat, rhs = load_matrix_rhs(data, i)
        model.linear_system = mat, rhs
        model._prepare_solver()
        res.append(condest(model.S_Ap_fs))
    return res


def get_Bp_cond(data: Sequence[TimeStepStats], model):
    res = []
    for i in range(sum(len(x.linear_solves) for x in data)):
        mat, rhs = load_matrix_rhs(data, i)
        sliced_mat = model.slice_jacobian(mat)
        omega = model.slice_omega(sliced_mat)
        res.append(condest(omega.Bp))
    return res


def get_Omega_p_cond(data: Sequence[TimeStepStats], model):
    res = []
    for i in range(sum(len(x.linear_solves) for x in data)):
        mat, rhs = load_matrix_rhs(data, i)
        sliced_mat = model.slice_jacobian(mat)
        omega = model.slice_omega(sliced_mat)
        res.append(condest(bmat([[omega.Bp, omega.C2p], [omega.C1p, omega.Ap]])))
    return res


def get_jacobian_cond(data: Sequence[TimeStepStats], model):
    res = []
    for i in range(sum(len(x.linear_solves) for x in data)):
        mat, rhs = load_matrix_rhs(data, i)
        res.append(condest(mat))
    return res


def get_petsc_converged_reason(x: Sequence[TimeStepStats]) -> list[int]:
    result = []
    for ts in x:
        for ls in ts.linear_solves:
            result.append(ls.petsc_converged_reason)
    return result


# def get_num_sticking_sliding_open(
#     x: Sequence[TimeStepStats],
# ) -> tuple[list[int], list[int], list[int]]:
#     st, sl, op, tr = get_num_sticking_sliding_open_transition(
#         x, transition_as_open=True
#     )
#     return st, sl, op


def get_num_sticking_sliding_open(
    x: Sequence[TimeStepStats],
) -> tuple[list[int], list[int], list[int]]:
    num_sticking = [ls.num_sticking for ts in x for ls in ts.linear_solves]
    num_sliding = [ls.num_sliding for ts in x for ls in ts.linear_solves]
    num_open = [ls.num_open for ts in x for ls in ts.linear_solves]
    return num_sticking, num_sliding, num_open


def get_cell_volumes(dofs_info_path: str, cell_size_multiplier: int):
    data = load_data(dofs_info_path)
    data = next(
        entry for entry in data if entry["cell_size_multiplier"] == cell_size_multiplier
    )
    return data["cell_volumes"]


def get_volume_sticking_sliding_open_transition(
    x: Sequence[TimeStepStats],
    dofs_info_path: str,
    cell_size_multiplier: int,
    transition_as_open: bool = True,
):
    st = []
    sl = []
    op = []
    tr = []
    cell_volumes = np.array(
        get_cell_volumes(
            dofs_info_path=dofs_info_path, cell_size_multiplier=cell_size_multiplier
        )
    )
    for ts in x:
        for ls in ts.linear_solves:
            st.append(sum(cell_volumes[ls.sticking]))
            sl.append(sum(cell_volumes[ls.sliding]))
            if transition_as_open:
                op.append(
                    sum(cell_volumes[np.array(ls.open_) | np.array(ls.transition)])
                )
            else:
                op.append(sum(cell_volumes[ls.open_]))
                tr.append(sum(cell_volumes[ls.transition]))
    return st, sl, op, tr


# def get_num_sticking_sliding_open_transition(
#     x: Sequence[TimeStepStats], transition_as_open: bool = True
# ) -> tuple[list[int], list[int], list[int]]:
#     st = []
#     sl = []
#     op = []
#     tr = []
#     for ts in x:
#         for ls in ts.linear_solves:
#             st.append(sum(ls.sticking))
#             sl.append(sum(ls.sliding))
#             if transition_as_open:
#                 op.append(sum(np.array(ls.open_) | np.array(ls.transition)))
#                 tr.append(0)
#             else:
#                 op.append(sum(ls.open_))
#                 tr.append(sum(ls.transition))
#     return st, sl, op, tr


def get_num_transition_cells(x: Sequence[TimeStepStats]) -> np.ndarray:
    transition = []
    for ts in x:
        for ls in ts.linear_solves:
            transition.append(sum(ls.transition_sticking_sliding))
    return np.array(transition)


def get_transition(x: Sequence[TimeStepStats], idx: int):
    linear_solve_data = [ls for ts in x for ls in ts.linear_solves][idx]
    return np.array(linear_solve_data.transition_sticking_sliding)


def get_sticking(x: Sequence[TimeStepStats], idx: int):
    linear_solve_data = [ls for ts in x for ls in ts.linear_solves][idx]
    return np.array(linear_solve_data.sticking)


def get_sliding(x: Sequence[TimeStepStats], idx: int):
    linear_solve_data = [ls for ts in x for ls in ts.linear_solves][idx]
    return np.array(linear_solve_data.sliding)


def get_open(x: Sequence[TimeStepStats], idx: int):
    linear_solve_data = [ls for ts in x for ls in ts.linear_solves][idx]
    return np.array(linear_solve_data.open_)


def get_sticking_sliding_open(x: Sequence[TimeStepStats], idx: int):
    return get_sticking(x, idx), get_sliding(x, idx), get_open(x, idx)


def group_intervals(arr):
    diffs = np.diff(arr)
    change_positions = np.where(diffs != 0)[0] + 1
    intervals = np.concatenate(([0], change_positions, [len(arr)]))
    return intervals


def color_time_steps(
    data: Sequence[TimeStepStats], grid=True, fill=False, legend=False
):
    num_newton_iters = [0] + [len(ts.linear_solves) for ts in data]
    newton_converged = [ts.nonlinear_convergence_status == 1 for ts in data]
    printed_newton_diverged_legend = False
    cumsum_newton_iters = np.cumsum(num_newton_iters, dtype=float)
    cumsum_newton_iters -= 0.5
    for i, (start, end) in enumerate(
        zip(cumsum_newton_iters[:-1], cumsum_newton_iters[1:])
    ):
        kwargs = {}
        if legend and i == 0:
            kwargs["label"] = "Time step sep."
        if fill:
            plt.axvspan(
                start, end, facecolor="white" if i % 2 else "grey", alpha=0.3, **kwargs
            )
        else:
            if i == len(cumsum_newton_iters) - 2:
                continue
            plt.axvline(
                end, linestyle="--", alpha=0.9, color="grey", linewidth=2, **kwargs
            )
        if not newton_converged[i]:
            kwargs = {}
            if legend and not printed_newton_diverged_legend:
                printed_newton_diverged_legend = True
                kwargs["label"] = "Newton diverged"
            plt.axvspan(start, end, fill=False, hatch="/", **kwargs)
    if grid:
        plt.gca().grid(True)
    plt.xlim(-0.5, cumsum_newton_iters[-1])
    set_integer_ticks("horizontal")


def set_integer_ticks(direction: Literal["vertical", "horizontal"]):
    if direction == "vertical":
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    elif direction == "horizontal":
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        raise ValueError(direction)


def color_converged_reason(data: Sequence[TimeStepStats], legend=True, grid=True):
    converged_reason = get_petsc_converged_reason(data)
    intervals = group_intervals(converged_reason)

    reasons_colors = {
        -9: "C0",
        -5: "C1",
        2: "C2",
        3: 'C2',
        -3: "C3",
        -4: "C4",
        -100: "black",
    }

    reasons_explained = {
        -3: "Diverged its",
        -9: "Nan or inf",
        -5: "Diverged breakdown",
        2: "Converged reltol",
        3: "Converged abstol",
        -100: "No data",
        -4: "Diverged dtol",
    }

    reasons_label = set()

    for i in range(len(intervals) - 1):
        reason = converged_reason[intervals[i]]
        kwargs = {}
        if legend and reason not in reasons_label:
            reasons_label.add(reason)
            kwargs["label"] = reasons_explained[reason]
        plt.axvspan(
            intervals[i] - 0.5,
            intervals[i + 1] - 0.5,
            facecolor=reasons_colors[reason],
            alpha=0.3,
            **kwargs,
        )

    plt.xlim(0, len(converged_reason) - 0.5)
    # if legend:
    #     plt.legend()

    if grid:
        plt.gca().grid(True)


def load_matrix_rhs(data: Sequence[TimeStepStats], idx: int):
    flat_data: list[LinearSolveStats] = [y for x in data for y in x.linear_solves]
    load_dir = Path("../matrices")
    mat = scipy.sparse.load_npz(load_dir / flat_data[idx].matrix_id)
    rhs = np.load(load_dir / flat_data[idx].rhs_id)
    return mat, rhs


def load_matrix_rhs_state_iterate_dt(data: Sequence[TimeStepStats], idx: int):
    flat_data: list[LinearSolveStats] = [y for x in data for y in x.linear_solves]
    load_dir = Path("../matrices")
    mat = scipy.sparse.load_npz(load_dir / flat_data[idx].matrix_id)
    rhs = np.load(load_dir / flat_data[idx].rhs_id)
    iterate = np.load(load_dir / flat_data[idx].iterate_id)
    state = np.load(load_dir / flat_data[idx].state_id)
    dt = flat_data[idx].simulation_dt
    return mat, rhs, state, iterate, dt


def load_data(path) -> Sequence[TimeStepStats]:
    with open(path, "r") as f:
        payload = json.load(f)
    try:
        return [TimeStepStats.from_json(x) for x in payload]
    except TypeError:
        return payload


def zoom_in_mat(mat, i, j, ni=200, nj=None):
    if nj is None:
        nj = ni

    radius_i = ni // 2
    radius_j = nj // 2
    radius_i = min(radius_i, mat.shape[0] // 2)
    radius_j = min(radius_j, mat.shape[1] // 2)
    i = max(i, radius_i)
    i = min(i, mat.shape[0] - radius_i)
    j = max(j, radius_j)
    j = min(j, mat.shape[1] - radius_j)

    istart = i - radius_i
    iend = i + radius_i
    jstart = j - radius_j
    jend = j + radius_j

    return istart, iend, jstart, jend


def set_zoomed_frame(istart, iend, jstart, jend):
    i_ticks = np.linspace(0, iend - istart - 1, 5, endpoint=True, dtype=int)
    j_ticks = np.linspace(0, jend - jstart - 1, 5, endpoint=True, dtype=int)
    ax = plt.gca()
    ax.set_yticks(i_ticks)
    ax.set_xticks(j_ticks)
    ax.set_yticklabels(i_ticks + istart)
    ax.set_xticklabels(j_ticks + jstart)


def matshow_around(mat, i, j, ni=200, nj=None, show=True, log=True):
    istart, iend, jstart, jend = zoom_in_mat(mat, i=i, j=j, ni=ni, nj=nj)
    plot_mat(mat[istart:iend, jstart:jend], show=False, log=log)
    set_zoomed_frame(istart, iend, jstart, jend)
    return istart, jstart


def spy_around(mat, i, j, ni=200, nj=None, show=True):
    istart, iend, jstart, jend = zoom_in_mat(mat, i=i, j=j, ni=ni, nj=nj)
    spy(mat[istart:iend, jstart:jend], show=False, aspect="auto")
    set_zoomed_frame(istart, iend, jstart, jend)
    return istart, jstart


COLOR_SLIDING = "green"
COLOR_STICKING = "#8B4513"
COLOR_TRANSITION = "#00bfff"
COLOR_OPEN = "blue"


def color_sticking_sliding_open_transition(
    entry: Sequence[TimeStepStats], transition_as_open: bool = True
):
    st, sl, op, tr = get_num_sticking_sliding_open_transition(
        entry, transition_as_open=transition_as_open
    )
    maximum = np.array([st, sl, op, tr]).max(axis=0)
    seen_sticking = seen_sliding = seen_open = seen_transition = False
    for i in range(maximum.size):
        kwargs = {}
        # if sliding[i] > 0:
        if sl[i] == maximum[i]:
            color = COLOR_SLIDING
            if not seen_sliding:
                kwargs["label"] = "Sliding"
            seen_sliding = True
        elif st[i] == maximum[i]:
            color = COLOR_STICKING
            if not seen_sticking:
                kwargs["label"] = "Sticking"
            seen_sticking = True
        elif tr[i] == maximum[i]:
            color = COLOR_TRANSITION
            if not seen_transition:
                kwargs["label"] = "Transition"
            seen_transition = True
        else:
            color = COLOR_OPEN
            if not seen_open:
                kwargs["label"] = "Open"
            seen_open = True
        # matplotlib.patches.Patch(linewidth=0
        plt.axvspan(
            i - 0.5, i + 0.5, facecolor=color, edgecolor="none", alpha=0.2, **kwargs
        )


def plot_grid(
    data,
    render_element,
    shape: tuple[int, int] = None,
    figsize: tuple[int, int] = (8, 8),
    ylabel: str = "Krylov iters.",
    xlabel: str = "Linear system idx.",
    legend: bool = True,
    ax_titles: dict = None,
    reuse_axes=None,
    return_axes: bool = False,
):
    if shape is None:
        shape = 3, (len(data) // 3 + len(data) % 3)

    if ax_titles is None:
        ax_titles = {}

    if reuse_axes is not None:
        axes = reuse_axes
        fig = plt.gcf()
    else:
        fig, axes = plt.subplots(
            nrows=shape[0], ncols=shape[1], squeeze=False, figsize=figsize
        )
    for i, (name, entry) in enumerate(data.items()):
        ax = axes.ravel()[i]

        try:
            ax.set_title(ax_titles[name])
        except KeyError:
            ax.set_title(name)

        plt.sca(ax)

        num_args = render_element.__code__.co_argcount
        if num_args == 2:
            render_element(name, entry)
        elif num_args == 1:
            render_element(entry)
        else:
            raise TypeError
        if i % shape[1] == 0:
            plt.ylabel(ylabel)
        if i >= (shape[0] - 1) * shape[1]:
            plt.xlabel(xlabel)
    if legend:
        lines = []
        labels = []
        for ax in axes.ravel():
            for line, label in zip(*ax.get_legend_handles_labels()):
                if label not in labels:
                    lines.append(line)
                    labels.append(label)
        fig.legend(
            lines,
            labels,
            # loc="center left",
            # bbox_to_anchor=(1, 0.5),
            loc="upper center",
            bbox_to_anchor=(0.5, 0),
            ncol=5,
            fancybox=True,
        )
    plt.tight_layout()
    if return_axes:
        return axes


def get_friction_bound_norm(model: pp.SolutionStrategy, data: Sequence[TimeStepStats]):
    fractures = model.mdg.subdomains(dim=model.nd - 1)
    num_ls = len([ls for ts in data for ls in ts.linear_solves])
    norms = []
    for i in range(num_ls):
        mat, rhs, state, iterate, dt = load_matrix_rhs_state_iterate_dt(data, i)
        model.equation_system.set_variable_values(iterate, iterate_index=0)
        model.equation_system.set_variable_values(state, time_step_index=0)
        b = model.friction_bound(fractures).value(model.equation_system)
        norms.append(abs(b).max())
    return norms


def plot_sticking_sliding_open_transition(
    entry: Sequence[TimeStepStats], transition_as_open: bool = True
):
    st, sl, op, tr = get_num_sticking_sliding_open_transition(
        entry, transition_as_open=transition_as_open
    )
    color_time_steps(entry, fill=True, grid=False, legend=True)
    plt.plot(st, label="Sticking", marker=".", color=COLOR_STICKING)
    plt.plot(sl, label="Sliding", marker=".", color=COLOR_SLIDING)
    plt.plot(op, label="Open", marker=".", color=COLOR_OPEN)
    if not transition_as_open:
        plt.plot(tr, label="Transition", marker=".", color=COLOR_TRANSITION)


def get_rhs_norms(model: pp.SolutionStrategy, data: Sequence[TimeStepStats], ord=2):
    bmat, prec = model._prepare_solver()
    num_ls = len([ls for ts in data for ls in ts.linear_solves])
    norms = [[] for i in range(6)]
    J_list = [bmat[[i]] for i in range(6)]
    for i in range(num_ls):
        mat, rhs, state, iterate, dt = load_matrix_rhs_state_iterate_dt(data, i)
        for nrm_list, J_i in zip(norms, J_list):
            nrm_list.append(np.linalg.norm(J_i.local_rhs(rhs), ord=ord))
    return norms


def solve_petsc_new(
    mat: "BlockMatrixStorage",
    solve_schema: "SolveSchema" = None,
    rhs_global=None,
    label="",
    logx_eigs=False,
    normalize_residual=False,
    tol=1e-10,
    pc_side: Literal["left", "right"] = "left",
    ksp_view: bool = False,
    rhs_eq_groups: Sequence[np.ndarray] = None,
    Qleft: "BlockMatrixStorage" = None,
    Qright: "BlockMatrixStorage" = None,
    restrict_indices: list[int] = None,
):
    from block_matrix import make_solver

    mat_Q = mat.copy()
    if Qleft is not None:
        assert Qleft.active_groups == mat.active_groups
        mat_Q.mat = Qleft.mat @ mat_Q.mat
        # mat_Q.set_zeros(4, 5)
    if Qright is not None:
        assert Qright.active_groups == mat.active_groups
        mat_Q.mat = mat_Q.mat @ Qright.mat
        # mat_Q.set_zeros(5, 4)

    mat_permuted, prec = make_solver(solve_schema, mat_Q)
    if restrict_indices is not None:
        mat_permuted = mat_permuted[restrict_indices]
        if Qleft is not None:
            Qleft = Qleft[restrict_indices]
        if Qright is not None:
            Qright = Qright[restrict_indices]

    if rhs_global is None:
        rhs_local = np.ones(mat.shape[0])
    else:
        rhs_local = mat_permuted.local_rhs(rhs_global)

    rhs_Q = rhs_local.copy()
    if Qleft is not None:
        Qleft = Qleft[mat_permuted.active_groups]
        rhs_Q = Qleft.mat @ rhs_Q

    gmres = PetscGMRES(mat_permuted.mat, pc=prec, tol=tol, pc_side=pc_side)

    if ksp_view:
        gmres.ksp.view()

    t0 = time.time()
    sol_Q = gmres.solve(rhs_Q)
    print("Solve", label, "took:", round(time.time() - t0, 2))
    residuals = gmres.get_residuals()
    info = gmres.ksp.getConvergedReason()
    eigs = gmres.ksp.computeEigenvalues()

    print(
        "True residual permuted:", norm(mat_permuted.mat @ sol_Q - rhs_Q) / norm(rhs_Q)
    )

    if Qright is not None:
        Qright = Qright[mat_permuted.active_groups]
        sol = mat.local_rhs(Qright.global_rhs(Qright.mat @ sol_Q))
        print(
            "True residual:",
            norm(mat.mat @ sol - mat.local_rhs(rhs_global))
            / norm(mat.local_rhs(rhs_global)),
        )
    else:
        sol = sol_Q

    print("PETSc Converged Reason:", info)
    linestyle = "-"
    if info <= 0:
        linestyle = "--"
        if len(eigs) > 0:
            print("lambda min:", min(abs(eigs)))

    plt.gcf().set_size_inches(14, 4)

    # ax = plt.gca()
    ax = plt.subplot(1, 2, 1)
    if normalize_residual:
        residuals /= residuals[0]
    ax.plot(residuals, label=label, marker=".", linestyle=linestyle)
    ax.set_yscale("log")

    ksp_norm_type = gmres.ksp.getNormType()  # 1-prec, 2-unprec
    if ksp_norm_type == 2:
        ax.set_ylabel("true residual")
    elif ksp_norm_type == 1:
        ax.set_ylabel("preconditioned residual")
    else:
        raise ValueError(ksp_norm_type)
    ax.set_xlabel("gmres iter.")
    ax.grid(True)
    if label != "":
        ax.legend()
    ax.set_title("GMRES Convergence")

    ax = plt.subplot(1, 2, 2)
    if logx_eigs:
        eigs.real = abs(eigs.real)
    # ax.scatter(eigs.real, eigs.imag, label=label, marker="$\lambda$", alpha=0.9)
    ax.scatter(eigs.real, eigs.imag, label=label, alpha=1, s=300, marker=next(MARKERS))
    ax.set_xlabel(r"Re($\lambda)$")
    ax.set_ylabel(r"Im($\lambda$)")
    ax.grid(True)
    if label != "":
        ax.legend()
    if logx_eigs:
        plt.xscale("log")
    ax.set_title("Eigenvalues estimate")
    return {"mat_Q": mat_permuted, "rhs_Q": rhs_Q, "prec": prec}


def dump_json(name, data):
    save_path = Path("./stats")
    save_path.mkdir(exist_ok=True)
    try:
        dict_data = [asdict(x) for x in data]
    except TypeError:
        dict_data = data
    json_data = json.dumps(dict_data)
    with open(save_path / name, "w") as file:
        file.write(json_data)


def write_dofs_info(
    model_name: str, make_model: Callable, cell_size_multipliers: list[int]
):
    filename = f"dofs_info_{model_name}.json"
    data = []
    for cell_size_multiplier in tqdm(cell_size_multipliers):
        model = make_model(cell_size_multiplier=cell_size_multiplier)
        model.prepare_simulation()
        model._initialize_solver()
        model.assemble_linear_system()
        model._prepare_solver()
        data_entry = dict()
        for i in range(6):
            data_entry[f"block {i}"] = model.bmat[5, i].shape[1]
        data_entry["cell_size_multiplier"] = cell_size_multiplier
        cell_volumes = np.concatenate(
            [frac.cell_volumes for frac in model.mdg.subdomains(dim=model.nd - 1)]
        ).tolist()
        data_entry["cell_volumes"] = cell_volumes
        data.append(data_entry)

    dump_json(filename, data)


def plot_eigs_exact(mat, logx: bool = True):
    lambdas = eigs(mat)
    if np.any(lambdas.real <= 0):
        print("Has negative lambda")
    if np.any(lambdas.real == 0):
        print("Has zero lambda")
    imag = lambdas.imag
    real = lambdas.real
    if logx:
        plt.xscale("log")
        real = abs(real)
    plt.scatter(real, imag, marker="x")

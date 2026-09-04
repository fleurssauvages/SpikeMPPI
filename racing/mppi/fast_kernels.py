from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional acceleration dependency
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorate(fn):
            return fn
        return decorate
        
@njit(cache=True, nogil=True, fastmath=False)
def _stadium_project_scalar(
    xw: float, yw: float,
    origin_x: float, origin_y: float,
    rot00: float, rot01: float, rot10: float, rot11: float,
    canonical_start_x: float, canonical_start_y: float,
    radius: float, left_arc_x: float, right_arc_x: float, center_y: float,
    straight_length: float, track_length: float,
) -> tuple[float, float]:
    dx = xw - origin_x
    dy = yw - origin_y
    px = dx * rot00 + dy * rot10 + canonical_start_x
    py = dx * rot01 + dy * rot11 + canonical_start_y
    arc = math.pi * radius

    bx = min(max(px, left_arc_x), right_arc_x)
    ex = px - bx
    ey = py - (center_y - radius)
    best_d2 = ex * ex + ey * ey
    best_s = bx - left_arc_x

    theta = min(max(math.atan2(py - center_y, px - right_arc_x), -0.5 * math.pi), 0.5 * math.pi)
    rx = right_arc_x + radius * math.cos(theta)
    ry = center_y + radius * math.sin(theta)
    ex = px - rx
    ey = py - ry
    d2 = ex * ex + ey * ey
    if d2 < best_d2:
        best_d2 = d2
        best_s = straight_length + radius * (theta + 0.5 * math.pi)

    tx = min(max(px, left_arc_x), right_arc_x)
    ex = px - tx
    ey = py - (center_y + radius)
    d2 = ex * ex + ey * ey
    if d2 < best_d2:
        best_d2 = d2
        best_s = straight_length + arc + (right_arc_x - tx)

    theta = math.atan2(py - center_y, px - left_arc_x)
    if theta < 0.5 * math.pi:
        theta += 2.0 * math.pi
    theta = min(max(theta, 0.5 * math.pi), 1.5 * math.pi)
    lx = left_arc_x + radius * math.cos(theta)
    ly = center_y + radius * math.sin(theta)
    ex = px - lx
    ey = py - ly
    d2 = ex * ex + ey * ey
    if d2 < best_d2:
        best_d2 = d2
        best_s = 2.0 * straight_length + arc + radius * (theta - 0.5 * math.pi)

    return best_s % track_length, best_d2


@njit(cache=True, nogil=True, fastmath=False)
def _stadium_distance_sq_scalar(
    xw: float, yw: float,
    origin_x: float, origin_y: float,
    rot00: float, rot01: float, rot10: float, rot11: float,
    canonical_start_x: float, canonical_start_y: float,
    radius: float, left_arc_x: float, right_arc_x: float, center_y: float,
    straight_length: float, track_length: float,
) -> float:
    _, d2 = _stadium_project_scalar(
        xw, yw, origin_x, origin_y, rot00, rot01, rot10, rot11,
        canonical_start_x, canonical_start_y, radius, left_arc_x,
        right_arc_x, center_y, straight_length, track_length,
    )
    return d2


@njit(cache=True, nogil=True, fastmath=False)
def stadium_rollout_cost_from_states(
    sampled_states: np.ndarray,
    controls: np.ndarray,
    nominal_controls: np.ndarray,
    ctrl_scale: np.ndarray,
    task_qpos_index: int,
    root_qpos_index: int,
    start_time: float,
    current_s: float,
    current_root_s: float,
    initial_task_root_distance: float,
    box_progress_weight: float,
    robot_progress_weight: float,
    reach_weight: float,
    box_max_lift: float,
    box_min_up: float,
    initial_task_height: float,
    origin_x: float,
    origin_y: float,
    rot00: float,
    rot01: float,
    rot10: float,
    rot11: float,
    canonical_start_x: float,
    canonical_start_y: float,
    radius: float,
    left_arc_x: float,
    right_arc_x: float,
    center_y: float,
    straight_length: float,
    track_length: float,
    allowed_sq: float,
    min_height: float,
    min_up: float,
    upright_weight: float,
    control_deviation_weight: float,
    costs_out: np.ndarray,
    terminal_progress_out: np.ndarray,
    failed_out: np.ndarray,
) -> None:
    """Fused exact rollout cost for StadiumTrack.

    This reproduces StadiumTrack.project + the existing rollout failure/progress/
    upright/control costs, but consumes MuJoCo FULLPHYSICS state output directly.
    No simulator setting, timestep, substep, horizon, or candidate is changed.
    """
    n = sampled_states.shape[0]
    h = sampled_states.shape[1]
    nu = controls.shape[2]
    tq = int(task_qpos_index)
    rq = int(root_qpos_index)
    half_track = 0.5 * track_length
    arc = math.pi * radius
    top_offset = straight_length + arc
    left_offset = 2.0 * straight_length + arc

    for i in range(n):
        s_prev = current_s
        root_s_prev = current_root_s
        cumulative = 0.0
        root_cumulative = 0.0
        prefix_sum = 0.0
        root_prefix_sum = 0.0
        approach_prefix_sum = 0.0
        upright_cost = 0.0
        control_cost = 0.0
        prev_time = start_time
        failed = False

        for t in range(h):
            state = sampled_states[i, t]
            # FULLPHYSICS is time followed by qpos, qvel, act. q indexes the
            # free-root qpos inside the packed state (already offset by time).
            # Track progress can belong to a separate free body (the pushed
            # box), while fall/upright constraints always belong to the robot.
            xw = state[tq]
            yw = state[tq + 1]
            z = state[rq + 2]
            qx = state[rq + 4]
            qy = state[rq + 5]
            up = 1.0 - 2.0 * (qx * qx + qy * qy)
            sim_time = state[0]

            # Inverse rigid transform into the canonical stadium frame.
            dx = xw - origin_x
            dy = yw - origin_y
            px = dx * rot00 + dy * rot10 + canonical_start_x
            py = dx * rot01 + dy * rot11 + canonical_start_y

            # Bottom straight.
            bx = px
            if bx < left_arc_x:
                bx = left_arc_x
            elif bx > right_arc_x:
                bx = right_arc_x
            ex = px - bx
            ey = py - (center_y - radius)
            best_d2 = ex * ex + ey * ey
            best_s = bx - left_arc_x

            # Right semicircle.
            theta = math.atan2(py - center_y, px - right_arc_x)
            lo = -0.5 * math.pi
            hi = 0.5 * math.pi
            if theta < lo:
                theta = lo
            elif theta > hi:
                theta = hi
            rx = right_arc_x + radius * math.cos(theta)
            ry = center_y + radius * math.sin(theta)
            ex = px - rx
            ey = py - ry
            d2 = ex * ex + ey * ey
            if d2 < best_d2:
                best_d2 = d2
                best_s = straight_length + radius * (theta + 0.5 * math.pi)

            # Top straight.
            tx = px
            if tx < left_arc_x:
                tx = left_arc_x
            elif tx > right_arc_x:
                tx = right_arc_x
            ex = px - tx
            ey = py - (center_y + radius)
            d2 = ex * ex + ey * ey
            if d2 < best_d2:
                best_d2 = d2
                best_s = top_offset + (right_arc_x - tx)

            # Left semicircle.
            theta = math.atan2(py - center_y, px - left_arc_x)
            if theta < 0.5 * math.pi:
                theta += 2.0 * math.pi
            lo = 0.5 * math.pi
            hi = 1.5 * math.pi
            if theta < lo:
                theta = lo
            elif theta > hi:
                theta = hi
            lx = left_arc_x + radius * math.cos(theta)
            ly = center_y + radius * math.sin(theta)
            ex = px - lx
            ey = py - ly
            d2 = ex * ex + ey * ey
            if d2 < best_d2:
                best_d2 = d2
                best_s = left_offset + radius * (theta - 0.5 * math.pi)

            # best_s is already in [0, length] up to endpoint roundoff. Match
            # np.mod semantics used by StadiumTrack.project.
            best_s = best_s % track_length

            root_d2 = best_d2
            root_s = best_s
            if rq != tq:
                root_s, root_d2 = _stadium_project_scalar(
                    state[rq], state[rq + 1],
                    origin_x, origin_y, rot00, rot01, rot10, rot11,
                    canonical_start_x, canonical_start_y,
                    radius, left_arc_x, right_arc_x, center_y,
                    straight_length, track_length,
                )

            if rq != tq:
                task_z = state[tq + 2]
                task_qx = state[tq + 4]
                task_qy = state[tq + 5]
                task_up = 1.0 - 2.0 * (task_qx * task_qx + task_qy * task_qy)
            else:
                task_z = initial_task_height
                task_up = 1.0

            # Packed FULLPHYSICS state does not contain MuJoCo contact pairs.
            # The stock/Numba fallback therefore uses low root height only as a
            # contact proxy, but still requires the torso to be inverted too.
            fell_proxy = (up < min_up) and (z < min_height)
            if (
                best_d2 > allowed_sq
                or root_d2 > allowed_sq
                or fell_proxy
                or (rq != tq and task_z > initial_task_height + box_max_lift)
                or (rq != tq and task_up < box_min_up)
                or sim_time <= prev_time + 1e-15
            ):
                failed = True
                break

            ds = best_s - s_prev
            if ds > half_track:
                ds -= track_length
            elif ds < -half_track:
                ds += track_length
            cumulative += ds
            prefix_sum += cumulative
            s_prev = best_s

            if rq != tq:
                root_ds = root_s - root_s_prev
                if root_ds > half_track:
                    root_ds -= track_length
                elif root_ds < -half_track:
                    root_ds += track_length
                root_cumulative += root_ds
                coupled = min(root_cumulative, max(cumulative, 0.0))
                root_prefix_sum += coupled
                root_s_prev = root_s

                dx_rb = state[tq] - state[rq]
                dy_rb = state[tq + 1] - state[rq + 1]
                dist_rb = math.sqrt(dx_rb * dx_rb + dy_rb * dy_rb)
                approach_prefix_sum += initial_task_root_distance - dist_rb

            prev_time = sim_time

            du_sq_sum = 0.0
            for u in range(nu):
                scaled = (controls[i, t, u] - nominal_controls[t, u]) / ctrl_scale[u]
                du_sq_sum += scaled * scaled
            control_cost += control_deviation_weight * (du_sq_sum / max(1, nu))
            err_up = 1.0 - up
            upright_cost += upright_weight * err_up * err_up

        terminal_progress_out[i] = cumulative
        failed_out[i] = failed
        if failed:
            costs_out[i] = math.inf
        else:
            inv_h = 1.0 / max(1, h)
            progress_cost = -box_progress_weight * prefix_sum * inv_h
            if rq != tq:
                progress_cost -= robot_progress_weight * root_prefix_sum * inv_h
                progress_cost -= reach_weight * approach_prefix_sum * inv_h
            costs_out[i] = progress_cost + upright_cost + control_cost

@njit(cache=True, nogil=True, fastmath=False)
def _lbps_score_fast(
    costs: np.ndarray,
    alpha: float,
    delta: float,
    rho: float,
    reward_norm: float,
) -> tuple[float, float, float]:
    sw = 0.0
    sw2 = 0.0
    weighted_return = 0.0
    count = 0
    for i in range(costs.shape[0]):
        c = costs[i]
        if not math.isfinite(c):
            continue
        count += 1
        z = -alpha * (c - rho)
        if z < -745.0:
            z = -745.0
        elif z > 0.0:
            z = 0.0
        w = math.exp(z)
        sw += w
        sw2 += w * w
        weighted_return += w * (-c)
    if count == 0 or sw <= 0.0 or sw2 <= 0.0:
        return -math.inf, 0.0, -math.inf
    ess = sw * sw / sw2
    expected_return = weighted_return / sw
    penalty = reward_norm * math.sqrt((1.0 - delta) / (delta * max(ess, 1e-300)))
    return expected_return - penalty, ess, expected_return


@njit(cache=True, nogil=True, fastmath=False)
def lbps_optimize_fast(
    costs: np.ndarray,
    delta: float,
    fallback_temperature: float,
    iterations: int,
) -> tuple[float, float, float, float, int, float, float]:
    """Allocation-free compiled equivalent of optimize_lbps_temperature."""
    fallback_alpha = 1.0 / max(fallback_temperature, 1e-300)
    finite_count = 0
    rho = math.inf
    max_cost = -math.inf
    reward_norm = 0.0
    for i in range(costs.shape[0]):
        c = costs[i]
        if math.isfinite(c):
            finite_count += 1
            if c < rho:
                rho = c
            if c > max_cost:
                max_cost = c
            ac = abs(c)
            if ac > reward_norm:
                reward_norm = ac

    if finite_count == 0:
        return fallback_temperature, fallback_alpha, 0.0, -math.inf, 0, 0.0, -math.inf

    spread = max_cost - rho
    if finite_count <= 1 or spread <= 1e-12 or reward_norm <= 1e-15:
        score, ess, er = _lbps_score_fast(costs, fallback_alpha, delta, rho, reward_norm)
        return fallback_temperature, fallback_alpha, ess, score, finite_count, reward_norm, er

    score0, _, _ = _lbps_score_fast(costs, 0.0, delta, rho, reward_norm)
    alpha1 = 1.0 / spread
    score1, _, _ = _lbps_score_fast(costs, alpha1, delta, rho, reward_norm)
    left = 0.0
    right = alpha1
    if score1 > score0:
        prevprev = 0.0
        prev = alpha1
        prev_score = score1
        bracketed = False
        for _ in range(40):
            nxt = prev * 2.0
            nxt_score, _, _ = _lbps_score_fast(costs, nxt, delta, rho, reward_norm)
            if nxt_score <= prev_score:
                left = prevprev
                right = nxt
                bracketed = True
                break
            prevprev = prev
            prev = nxt
            prev_score = nxt_score
        if not bracketed:
            score, ess, er = _lbps_score_fast(costs, prev, delta, rho, reward_norm)
            return 1.0 / max(prev, 1e-300), prev, ess, score, finite_count, reward_norm, er

    golden = 0.6180339887498949
    a = left
    b = right
    c = b - golden * (b - a)
    d = a + golden * (b - a)
    fc, _, _ = _lbps_score_fast(costs, c, delta, rho, reward_norm)
    fd, _, _ = _lbps_score_fast(costs, d, delta, rho, reward_norm)
    niter = max(8, int(iterations))
    for _ in range(niter):
        if fc > fd:
            b = d
            d = c
            fd = fc
            c = b - golden * (b - a)
            fc, _, _ = _lbps_score_fast(costs, c, delta, rho, reward_norm)
        else:
            a = c
            c = d
            fc = fd
            d = a + golden * (b - a)
            fd, _, _ = _lbps_score_fast(costs, d, delta, rho, reward_norm)

    alpha = 0.5 * (a + b)
    score, ess, er = _lbps_score_fast(costs, alpha, delta, rho, reward_norm)
    if score0 >= score:
        alpha = 0.0
        score, ess, er = _lbps_score_fast(costs, alpha, delta, rho, reward_norm)
    if alpha <= 1e-14 / spread:
        alpha = 1e-14 / spread
        score, ess, er = _lbps_score_fast(costs, alpha, delta, rho, reward_norm)
    return 1.0 / alpha, alpha, ess, score, finite_count, reward_norm, er

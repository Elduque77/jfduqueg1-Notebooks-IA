"""
Gridworld 15x15 resuelto con Reinforcement Learning basado en modelo (MDP):
  - Iteración de Valores (Value Iteration)
  - Iteración de Políticas (Policy Iteration)

Se construye una cuadrícula 15x15 con 5 obstáculos, un punto de inicio y uno
final. Se calcula la función de valor óptima V*(s) y la política óptima
pi*(s) mediante ambos métodos, mostrando el avance (delta) en cada iteración,
y finalmente se traza el mejor camino desde el inicio hasta el final.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ----------------------------------------------------------------------
# 1. Definición del entorno (Gridworld)
# ----------------------------------------------------------------------
GRID_SIZE = 15
START = (0, 0)
GOAL = (14, 14)

# 5 obstáculos fijos (reproducibles)
OBSTACLES = [(3, 3), (5, 7), (7, 2), (9, 10), (12, 5)]

ACTIONS = {
    "UP":    (-1, 0),
    "DOWN":  (1, 0),
    "LEFT":  (0, -1),
    "RIGHT": (0, 1),
}
ACTION_LIST = list(ACTIONS.keys())

GAMMA = 0.95          # factor de descuento
THETA = 1e-4           # umbral de convergencia
STEP_REWARD = -1.0     # costo por moverse
GOAL_REWARD = 100.0    # recompensa al llegar a la meta
OBSTACLE_REWARD = -50.0  # penalización si "cae" en un obstáculo


def in_bounds(s):
    r, c = s
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE


def is_obstacle(s):
    return s in OBSTACLES


def is_terminal(s):
    return s == GOAL


def step(s, a):
    """Devuelve (siguiente_estado, recompensa) al aplicar la acción a en s."""
    if is_terminal(s):
        return s, 0.0

    dr, dc = ACTIONS[a]
    ns = (s[0] + dr, s[1] + dc)

    if not in_bounds(ns):
        ns = s  # choca con el borde, se queda en el mismo lugar
        return ns, STEP_REWARD

    if is_obstacle(ns):
        # no se permite entrar al obstáculo: se queda en el mismo lugar
        # pero recibe una penalización fuerte para que la política lo evite
        return s, OBSTACLE_REWARD

    if ns == GOAL:
        return ns, GOAL_REWARD

    return ns, STEP_REWARD


def all_states():
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            s = (r, c)
            if not is_obstacle(s):
                yield s


# ----------------------------------------------------------------------
# 2. Value Iteration
# ----------------------------------------------------------------------
def value_iteration(verbose=True, max_iter=1000):
    V = {s: 0.0 for s in all_states()}
    history = []

    for it in range(1, max_iter + 1):
        delta = 0.0
        newV = V.copy()
        for s in all_states():
            if is_terminal(s):
                continue
            best = -np.inf
            for a in ACTION_LIST:
                ns, r = step(s, a)
                q = r + GAMMA * V[ns]
                best = max(best, q)
            newV[s] = best
            delta = max(delta, abs(newV[s] - V[s]))
        V = newV
        history.append(delta)
        if verbose:
            print(f"[Value Iteration] Iteración {it:3d}  ->  delta = {delta:.6f}")
        if delta < THETA:
            break

    policy = extract_policy(V)
    return V, policy, history


def extract_policy(V):
    policy = {}
    for s in all_states():
        if is_terminal(s):
            continue
        best_a, best_q = None, -np.inf
        for a in ACTION_LIST:
            ns, r = step(s, a)
            q = r + GAMMA * V[ns]
            if q > best_q:
                best_q, best_a = q, a
        policy[s] = best_a
    return policy


# ----------------------------------------------------------------------
# 3. Policy Iteration
# ----------------------------------------------------------------------
def policy_evaluation(policy, V, max_iter=200):
    for _ in range(max_iter):
        delta = 0.0
        newV = V.copy()
        for s in all_states():
            if is_terminal(s):
                continue
            a = policy[s]
            ns, r = step(s, a)
            newV[s] = r + GAMMA * V[ns]
            delta = max(delta, abs(newV[s] - V[s]))
        V = newV
        if delta < THETA:
            break
    return V


def policy_iteration(verbose=True, max_iter=100):
    # política inicial arbitraria (siempre intenta ir a la derecha)
    policy = {s: "RIGHT" for s in all_states() if not is_terminal(s)}
    V = {s: 0.0 for s in all_states()}
    history = []

    for it in range(1, max_iter + 1):
        V = policy_evaluation(policy, V)

        stable = True
        changes = 0
        for s in all_states():
            if is_terminal(s):
                continue
            old_a = policy[s]
            best_a, best_q = None, -np.inf
            for a in ACTION_LIST:
                ns, r = step(s, a)
                q = r + GAMMA * V[ns]
                if q > best_q:
                    best_q, best_a = q, a
            policy[s] = best_a
            if best_a != old_a:
                stable = False
                changes += 1

        history.append(changes)
        if verbose:
            print(f"[Policy Iteration] Iteración {it:3d}  ->  {changes} acciones cambiadas")
        if stable:
            break

    return V, policy, history


# ----------------------------------------------------------------------
# 4. Extraer el mejor camino siguiendo la política
# ----------------------------------------------------------------------
def trace_path(policy, start=START, goal=GOAL, max_steps=500):
    path = [start]
    s = start
    visited = set()
    for _ in range(max_steps):
        if s == goal:
            break
        if s in visited:
            break  # evita bucles infinitos
        visited.add(s)
        a = policy.get(s)
        if a is None:
            break
        dr, dc = ACTIONS[a]
        ns = (s[0] + dr, s[1] + dc)
        if not in_bounds(ns) or is_obstacle(ns):
            break
        path.append(ns)
        s = ns
    return path


# ----------------------------------------------------------------------
# 5. Visualización
# ----------------------------------------------------------------------
def plot_results(V, policy, path, history_vi, history_pi):
    fig = plt.figure(figsize=(16, 10))

    # --- Panel 1: Grid con obstáculos, política (flechas) y camino ---
    ax1 = fig.add_subplot(2, 2, (1, 3))
    ax1.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax1.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax1.set_xticks(range(GRID_SIZE))
    ax1.set_yticks(range(GRID_SIZE))
    ax1.grid(True, color="lightgray", linewidth=0.5)
    ax1.invert_yaxis()
    ax1.set_title("Gridworld 15x15 – Política óptima y mejor camino", fontsize=13)

    # mapa de calor de valores de fondo
    value_grid = np.full((GRID_SIZE, GRID_SIZE), np.nan)
    for (r, c), v in V.items():
        value_grid[r, c] = v
    im = ax1.imshow(value_grid, cmap="viridis", alpha=0.55,
                     extent=(-0.5, GRID_SIZE - 0.5, GRID_SIZE - 0.5, -0.5))
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="V(s)")

    # obstáculos
    for (r, c) in OBSTACLES:
        ax1.add_patch(patches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                         facecolor="black"))

    # flechas de política
    arrow_map = {"UP": (0, -0.3), "DOWN": (0, 0.3), "LEFT": (-0.3, 0), "RIGHT": (0.3, 0)}
    for (r, c), a in policy.items():
        if (r, c) in OBSTACLES:
            continue
        dx, dy = arrow_map[a]
        ax1.arrow(c, r, dx, dy, head_width=0.12, head_length=0.12,
                   fc="white", ec="white", alpha=0.6)

    # camino óptimo
    path_rows = [p[0] for p in path]
    path_cols = [p[1] for p in path]
    ax1.plot(path_cols, path_rows, color="red", linewidth=2.5, marker="o",
              markersize=4, label="Mejor camino")

    # inicio y fin
    ax1.scatter(*START[::-1], color="lime", s=150, edgecolor="black",
                zorder=5, label="Inicio")
    ax1.scatter(*GOAL[::-1], color="gold", s=150, marker="*",
                edgecolor="black", zorder=5, label="Meta")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.15, 1.0))

    # --- Panel 2: convergencia de Value Iteration ---
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(range(1, len(history_vi) + 1), history_vi, marker="o", color="tab:blue")
    ax2.set_yscale("log")
    ax2.set_xlabel("Iteración")
    ax2.set_ylabel("Delta máximo (log)")
    ax2.set_title("Convergencia – Value Iteration")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # --- Panel 3: convergencia de Policy Iteration ---
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.bar(range(1, len(history_pi) + 1), history_pi, color="tab:orange")
    ax3.set_xlabel("Iteración")
    ax3.set_ylabel("Acciones que cambiaron")
    ax3.set_title("Convergencia – Policy Iteration")
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/grid_rl_resultado.png", dpi=150)
    print("\nImagen guardada en /mnt/user-data/outputs/grid_rl_resultado.png")


# ----------------------------------------------------------------------
# 6. Programa principal
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("ENTORNO: Gridworld 15x15")
    print(f"Inicio: {START} | Meta: {GOAL} | Obstáculos: {OBSTACLES}")
    print("=" * 60)

    print("\n--- VALUE ITERATION ---")
    V_vi, policy_vi, hist_vi = value_iteration()

    print("\n--- POLICY ITERATION ---")
    V_pi, policy_pi, hist_pi = policy_iteration()

    path = trace_path(policy_vi)
    print("\nMejor camino encontrado (Value Iteration):")
    print(" -> ".join(str(p) for p in path))
    print(f"Longitud del camino: {len(path) - 1} pasos")

    plot_results(V_vi, policy_vi, path, hist_vi, hist_pi)

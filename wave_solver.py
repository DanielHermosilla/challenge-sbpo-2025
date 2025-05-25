import gurobipy as gp
from gurobipy import GRB


def read_instance(filename):
    """
    Lee una instancia del problema desde un archivo de texto con el formato SBPO 2025.
    Devuelve:
      - u_oi: dict[(o, i)] → unidades solicitadas del ítem i en la orden o
      - u_ai: dict[(a, i)] → unidades disponibles del ítem i en el pasillo a
      - LB, UB: límites inferior y superior de unidades totales
      - orders, items, aisles: listas de índices
    """
    u_oi = {}
    u_ai = {}
        o, num_items, a = map(int, f.readline().split())
        # Leer órdenes
        for o_idx in range(o):
            parts = list(map(int, f.readline().split()))
            k = parts[0]
            for j in range(k):
                item_idx = parts[1 + 2 * j]
                qty = parts[1 + 2 * j + 1]
                u_oi[(o_idx, item_idx)] = qty
        # Leer pasillos
        for a_idx in range(a):
            parts = list(map(int, f.readline().split()))
            l = parts[0]
            for j in range(l):
                item_idx = parts[1 + 2 * j]
                qty = parts[1 + 2 * j + 1]
                u_ai[(a_idx, item_idx)] = qty
        # Última línea: LB, UB
        LB, UB = map(int, f.readline().split())
    orders = list(range(o))
    items = list(range(num_items))
    aisles = list(range(a))
    return u_oi, u_ai, LB, UB, orders, items, aisles


def solve_fractional(
    u_oi, u_ai, LB, UB, orders, items, aisles, tol=1e-6, max_iter=50, verbose=True
):
    """
    Resuelve el problema fraccional por el método de Dinkelbach:
      max (sum_{o,i} u_oi * x_o) / (sum_a y_a)
    sujeto a las restricciones de capacidad y disponibilidad.
    """
    λ = 0.0  # valor inicial del ratio

    for it in range(1, max_iter + 1):
        print("Iteracion %d", it)
        m = gp.Model()
        # Variables binarias
        x = m.addVars(orders, vtype=GRB.BINARY, name="x")
        y = m.addVars(aisles, vtype=GRB.BINARY, name="y")

        # Expresiones de numerador y denominador
        Numer = gp.quicksum(u_oi.get((o, i), 0) * x[o] for o in orders for i in items)
        Denom = gp.quicksum(y[a] for a in aisles)

        # Evitar división por cero
        m.addConstr(Denom >= 1, name="at_least_one_aisle")

        # Restricciones de capacidad
        m.addConstr(Numer >= LB, name="LB_wave")
        m.addConstr(Numer <= UB, name="UB_wave")

        # Restricciones de disponibilidad de ítems
        for i in items:
            lhs = gp.quicksum(u_oi.get((o, i), 0) * x[o] for o in orders)
            rhs = gp.quicksum(u_ai.get((a, i), 0) * y[a] for a in aisles)
            m.addConstr(lhs <= rhs, name=f"avail_item_{i}")

        # Objetivo Dinkelbach: max N(x) - λ * D(y)
        m.setObjective(Numer - λ * Denom, GRB.MAXIMIZE)
        m.Params.OutputFlag = 0
        m.optimize()

        if m.status != GRB.OPTIMAL:
            raise RuntimeError(f"No se encontró solución óptima en iteración {it}")

        N_val = Numer.getValue()
        D_val = Denom.getValue()
        gap = N_val - λ * D_val

        if verbose:
            print(
                f"Iter {it}: λ = {λ:.6f}, N = {N_val:.2f}, D = {D_val:.2f}, gap = {gap:.2e}"
            )

        # Criterio de parada
        if abs(gap) <= tol:
            if verbose:
                print(f"Convergió en iteración {it}: ratio = {N_val/D_val:.6f}")
            break

        # Actualizar λ
        λ = N_val / D_val

    # Reconstruir soluciones enteras
    x_sol = {o: int(x[o].X > 0.5) for o in orders}
    y_sol = {a: int(y[a].X > 0.5) for a in aisles}
    best_ratio = N_val / D_val

    return x_sol, y_sol, best_ratio


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Uso: python wave_solver.py instancia.txt")
        sys.exit(1)

    inst_file = sys.argv[1]
    u_oi, u_ai, LB, UB, orders, items, aisles = read_instance(inst_file)

    x_sol, y_sol, best_ratio = solve_fractional(
        u_oi, u_ai, LB, UB, orders, items, aisles, verbose=True
    )

    print("\nÓrdenes seleccionadas:")
    print([o for o, v in x_sol.items() if v == 1])
    print("Pasillos visitados:")
    print([a for a, v in y_sol.items() if v == 1])
    print(f"Ratio óptimo: {best_ratio:.6f}")

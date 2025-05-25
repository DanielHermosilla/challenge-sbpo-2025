package org.sbpo2025.challenge;

import com.google.ortools.Loader;
// --- para tu knapsack con MPSolver ---
import com.google.ortools.linearsolver.MPConstraint;
import com.google.ortools.linearsolver.MPObjective;
import com.google.ortools.linearsolver.MPSolver;
import com.google.ortools.linearsolver.MPVariable;

// --- para el CP-SAT de refinamiento ---
import com.google.ortools.sat.LinearExprBuilder;
import com.google.ortools.sat.CpModel;
import com.google.ortools.sat.CpSolver;
import com.google.ortools.sat.CpSolverStatus;
import com.google.ortools.sat.LinearExpr;
import com.google.ortools.sat.BoolVar;
import com.google.ortools.sat.IntVar;

import org.apache.commons.lang3.time.StopWatch;
import java.util.*;
import java.util.concurrent.TimeUnit;

public class ChallengeSolver {
	static {
		// load OR-Tools native libraries
		Loader.loadNativeLibraries();
	}

	private final long MAX_RUNTIME = 600_000; // 10 minutes in ms
	protected List<Map<Integer, Integer>> orders; // orders[o][i] = u_{oi}
	protected List<Map<Integer, Integer>> aisles; // aisles[a][i] = u_{ai}
	protected int nItems;
	protected int waveSizeLB, waveSizeUB;

	public ChallengeSolver(
			List<Map<Integer, Integer>> orders,
			List<Map<Integer, Integer>> aisles,
			int nItems,
			int waveSizeLB,
			int waveSizeUB) {
		this.orders = orders;
		this.aisles = aisles;
		this.nItems = nItems;
		this.waveSizeLB = waveSizeLB;
		this.waveSizeUB = waveSizeUB;
	}

	@SuppressWarnings("ConstantConditions")
	public ChallengeSolution solve(StopWatch stopWatch) {

		int m = aisles.size();
		int maxK = Math.min(m, 5); // Explorar hasta 10 pasillos

		double bestRatio = Double.NEGATIVE_INFINITY;
		Set<Integer> bestOrders = Collections.emptySet();
		Set<Integer> bestAisles = Collections.emptySet();

		// Precomputar el tamaño total de cada pedido
		int[] orderSizes = new int[orders.size()];
		for (int o = 0; o < orders.size(); o++) {
			int sum = 0;
			for (int qty : orders.get(o).values())
				sum += qty;
			orderSizes[o] = sum;
		}

		// Precompute each aisle's total capacity sum_i u_{ai}
		// Precomputar la capacidad total de cada pasillo
		double[] aisleTotals = new double[m];
		for (int a = 0; a < m; a++) {
			int sum = 0;
			for (int qty : aisles.get(a).values())
				sum += qty;
			aisleTotals[a] = sum;
		}

		// Try every k = 1..maxK
		// Intentar cada |k| pasillos, del primero hasta maxK
		for (int k = 1; k <= maxK; k++) {
			// time check
			long remaining = TimeUnit.MILLISECONDS.toMillis(MAX_RUNTIME) - stopWatch.getTime(TimeUnit.MILLISECONDS);
			if (remaining <= 0)
				break;

			// Obtener los k pasillos con mayor capacidad total
			Integer[] idx = new Integer[m];
			for (int a = 0; a < m; a++)
				idx[a] = a;
			Arrays.sort(idx, Comparator.comparingDouble((Integer a) -> aisleTotals[a]).reversed());
			Set<Integer> chosenAisles = new HashSet<>();
			for (int i = 0; i < k; i++)
				chosenAisles.add(idx[i]);

			// Construir el vector de capacidad v_i = sum_{a in chosen} u_{ai}
			int[] cap = new int[nItems];
			for (int a : chosenAisles) {
				for (Map.Entry<Integer, Integer> e : aisles.get(a).entrySet()) {
					cap[e.getKey()] += e.getValue();
				}
			}

			// Resolver el knapsack para poder obtener x_{ij}
			MPSolver solver = MPSolver.createSolver("SCIP");
			if (solver == null)
				continue; // Algo mal salió con el solver

			// Variables de decisión x_o ∈ {0,1}
			MPVariable[] x = new MPVariable[orders.size()];
			for (int o = 0; o < orders.size(); o++) {
				x[o] = solver.makeBoolVar("x_" + o);
			}

			// Restriccion de capacidad, para cada item i, sum_o x_o * u_{oi} <= cap[i]
			for (int i = 0; i < nItems; i++) {
				MPConstraint c = solver.makeConstraint(0.0, cap[i], "item_" + i);
				for (int o = 0; o < orders.size(); o++) {
					int qty = orders.get(o).getOrDefault(i, 0);
					if (qty > 0)
						c.setCoefficient(x[o], qty);
				}
			}

			// Restricción de tamaño del wave: LB <= sum_o x_o * orderSize_o <= UB
			MPConstraint lb = solver.makeConstraint(waveSizeLB, Double.POSITIVE_INFINITY, "LB");
			MPConstraint ub = solver.makeConstraint(Double.NEGATIVE_INFINITY, waveSizeUB, "UB");
			for (int o = 0; o < orders.size(); o++) {
				lb.setCoefficient(x[o], orderSizes[o]);
				ub.setCoefficient(x[o], orderSizes[o]);
			}

			// objective: Maximizar todas las unidades = sum_o x_o * orderSize_o
			MPObjective obj = solver.objective();
			for (int o = 0; o < orders.size(); o++) {
				obj.setCoefficient(x[o], orderSizes[o]);
			}
			obj.setMaximization();

			// Poner un límite de tiempo
			solver.setTimeLimit(remaining);

			MPSolver.ResultStatus status = solver.solve();

			if (status == MPSolver.ResultStatus.OPTIMAL ||
					status == MPSolver.ResultStatus.FEASIBLE) {
				// Total de unidades obtenidas
				double totalUnits = obj.value();
				double ratio = totalUnits / k;

				// Si el ratio mejora, guardamos la solución
				if (ratio > bestRatio) {
					bestRatio = ratio;
					bestAisles = new HashSet<>(chosenAisles);
					bestOrders = new HashSet<>();
					for (int o = 0; o < orders.size(); o++) {
						if (x[o].solutionValue() > 0.5) {
							bestOrders.add(o);
						}
					}
				}
			}

			// Revisamos para ver si superamos el tiempo máximo, muy poco probable...
			if (stopWatch.getTime(TimeUnit.MILLISECONDS) > MAX_RUNTIME)
				break;
		}
		// ———————————————— A PARTIR DE AQUÍ ARMAMOS EL CP-SAT ————————————————
		long elapsed = stopWatch.getTime(TimeUnit.MILLISECONDS);
		double remainingSec = Math.max(0, (MAX_RUNTIME - elapsed - 5000) / 1000.0);

		// 1) Creamos el modelo
		CpModel model = new CpModel();

		// 2) Variables binarias x[o] y y[a]
		int O = orders.size(), A = aisles.size();
		BoolVar[] x = new BoolVar[O];
		for (int o = 0; o < O; o++) {
			x[o] = model.newBoolVar("x_" + o);
		}
		BoolVar[] y = new BoolVar[A];
		for (int a = 0; a < A; a++) {
			y[a] = model.newBoolVar("y_" + a);
		}

		// 3) Restricciones de capacidad de ítems
		for (int i = 0; i < nItems; i++) {
			// sum_o x[o]*u_{oi} <= sum_a y[a]*u_{ai}
			LinearExprBuilder left = LinearExpr.newBuilder();
			for (int o = 0; o < O; o++) {
				int u_oi = orders.get(o).getOrDefault(i, 0);
				if (u_oi > 0)
					left.addTerm(x[o], u_oi);
			}
			LinearExprBuilder right = LinearExpr.newBuilder();
			for (int a = 0; a < A; a++) {
				int u_ai = aisles.get(a).getOrDefault(i, 0);
				if (u_ai > 0)
					right.addTerm(y[a], u_ai);
			}
			model.addLessOrEqual(left, right);
		}

		// 4) Restricciones de wave‐size
		// LB <= sum_o x[o]*orderSize[o] <= UB
		LinearExprBuilder waveSum = LinearExpr.newBuilder();
		for (int o = 0; o < O; o++) {
			waveSum.addTerm(x[o], orderSizes[o]);
		}
		model.addGreaterOrEqual(waveSum, waveSizeLB);
		model.addLessOrEqual(waveSum, waveSizeUB);

		// 5) Variables auxiliares para función objetivo “aprox. linealizada”
		// totalUnits = sum_o x[o]*orderSizes[o]
		// aisleCount = sum_a y[a]
		IntVar totalUnits = model.newIntVar(0, Arrays.stream(orderSizes).sum(), "totalUnits");
		model.addEquality(totalUnits, waveSum);

		IntVar aisleCount = model.newIntVar(0, A, "aisleCount");
		model.addEquality(aisleCount, LinearExpr.sum(y));

		// 6) Objetivo: max (totalUnits*1000 - aisleCount * floor(1000*bestRatio))
		// con la escala evitamos fracciones y empujamos a ratios > bestRatio
		long scale = (long) Math.floor(1000.0 * bestRatio);
		LinearExprBuilder objExpr = LinearExpr.newBuilder()
				.addTerm(totalUnits, 1000)
				.addTerm(aisleCount, -scale);
		model.maximize(objExpr);

		// 7) “Hints” para arrancar con la solución heurística
		for (int o : bestOrders)
			model.addHint(x[o], 1);
		for (int a : bestAisles)
			model.addHint(y[a], 1);

		// 8) Resolución con límite de tiempo
		CpSolver solver = new CpSolver();
		solver.getParameters().setMaxTimeInSeconds(remainingSec);
		CpSolverStatus status = solver.solve(model);

		// 9) Lectura de la solución final
		Set<Integer> finalOrders = new HashSet<>();
		Set<Integer> finalAisles = new HashSet<>();
		double finalRatio = bestRatio; // en caso no mejore
		if (status == CpSolverStatus.OPTIMAL || status == CpSolverStatus.FEASIBLE) {
			int chosenAisles = 0;
			for (int a = 0; a < A; a++) {
				if (solver.booleanValue(y[a])) {
					finalAisles.add(a);
					chosenAisles++;
				}
			}
			int pickedUnits = (int) solver.value(totalUnits);
			finalRatio = (double) pickedUnits / (chosenAisles > 0 ? chosenAisles : 1);
			for (int o = 0; o < O; o++) {
				if (solver.booleanValue(x[o]))
					finalOrders.add(o);
			}
		}

		// 10) Devolvemos la solución exacta “mejorada”
		return new ChallengeSolution(finalOrders, finalAisles);

	}

	protected long getRemainingTime(StopWatch sw) {
		return Math.max(
				TimeUnit.MILLISECONDS.toMillis(MAX_RUNTIME) - sw.getTime(TimeUnit.MILLISECONDS),
				0L);
	}
}

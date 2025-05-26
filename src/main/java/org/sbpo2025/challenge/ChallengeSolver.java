package org.sbpo2025.challenge;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
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
		int maxK = Math.min(m, 20);

		double bestRatio = Double.NEGATIVE_INFINITY;
		Set<Integer> bestOrders = Collections.emptySet();
		Set<Integer> bestAisles = Collections.emptySet();

		int[] orderSizes = new int[orders.size()];
		for (int o = 0; o < orders.size(); o++) {
			int sum = 0;
			for (int qty : orders.get(o).values())
				sum += qty;
			orderSizes[o] = sum;
		}

		double[] aisleTotals = new double[m];
		for (int a = 0; a < m; a++) {
			int sum = 0;
			for (int qty : aisles.get(a).values())
				sum += qty;
			aisleTotals[a] = sum;
		}

		System.out.println("[INFO] Lanzando threads para evaluar K = 2 ... " + maxK);
		ExecutorService executor = Executors
				.newFixedThreadPool(Math.min(maxK, Runtime.getRuntime().availableProcessors()));
		List<Future<KResult>> futures = new ArrayList<>();
		for (int k = 2; k <= maxK; k++) {
			final int fk = k;
			futures.add(executor.submit(() -> {
				long remaining = getRemainingTime(stopWatch);
				if (remaining <= 0) {
					System.out.println("[THREAD K=" + fk + "] Tiempo agotado antes de empezar.");
					return null;
				}

				Integer[] idx = new Integer[m];
				for (int a = 0; a < m; a++)
					idx[a] = a;
				Arrays.sort(idx, Comparator.comparingDouble((Integer a) -> aisleTotals[a]).reversed());
				Set<Integer> chosenAisles = new HashSet<>();
				for (int i = 0; i < fk; i++)
					chosenAisles.add(idx[i]);

				int[] cap = new int[nItems];
				for (int a : chosenAisles) {
					for (Map.Entry<Integer, Integer> e : aisles.get(a).entrySet()) {
						cap[e.getKey()] += e.getValue();
					}
				}

				MPSolver solver = MPSolver.createSolver("SCIP");
				if (solver == null) {
					System.out.println("[THREAD K=" + fk + "] No se pudo inicializar solver.");
					return null;
				}

				solver.setTimeLimit(Math.max(200, remaining / (maxK - 1)));

				MPVariable[] x = new MPVariable[orders.size()];
				for (int o = 0; o < orders.size(); o++) {
					x[o] = solver.makeBoolVar("x_" + o);
				}

				for (int i = 0; i < nItems; i++) {
					MPConstraint c = solver.makeConstraint(0.0, cap[i], "item_" + i);
					for (int o = 0; o < orders.size(); o++) {
						int qty = orders.get(o).getOrDefault(i, 0);
						if (qty > 0)
							c.setCoefficient(x[o], qty);
					}
				}

				MPConstraint lb = solver.makeConstraint(waveSizeLB, Double.POSITIVE_INFINITY, "LB");
				MPConstraint ub = solver.makeConstraint(Double.NEGATIVE_INFINITY, waveSizeUB, "UB");
				for (int o = 0; o < orders.size(); o++) {
					lb.setCoefficient(x[o], orderSizes[o]);
					ub.setCoefficient(x[o], orderSizes[o]);
				}

				MPObjective obj = solver.objective();
				for (int o = 0; o < orders.size(); o++) {
					obj.setCoefficient(x[o], orderSizes[o]);
				}
				obj.setMaximization();

				MPSolver.ResultStatus status = solver.solve();

				if (status == MPSolver.ResultStatus.OPTIMAL ||
						status == MPSolver.ResultStatus.FEASIBLE) {
					double totalUnits = obj.value();
					double ratio = totalUnits / fk;
					Set<Integer> resultOrders = new HashSet<>();
					for (int o = 0; o < orders.size(); o++) {
						if (x[o].solutionValue() > 0.5) {
							resultOrders.add(o);
						}
					}
					System.out.println("[THREAD K=" + fk + "] Factible: Ratio=" + ratio + ", Unidades=" + totalUnits
							+ ", Pasillos=" + chosenAisles);
					return new KResult(ratio, new HashSet<>(chosenAisles), resultOrders);
				} else {
					System.out.println("[THREAD K=" + fk + "] No factible.");
				}
				return null;
			}));
		}
		executor.shutdown();

		int k_actual = 2;
		for (Future<KResult> fut : futures) {
			try {
				KResult kr = fut.get(Math.max(100, getRemainingTime(stopWatch)), TimeUnit.MILLISECONDS);
				if (kr != null && kr.ratio > bestRatio) {
					bestRatio = kr.ratio;
					bestAisles = kr.aisles;
					bestOrders = kr.orders;
					System.out.println("[INFO] Mejor solución parcial en K=" + k_actual + ": Ratio=" + bestRatio
							+ ", Pasillos=" + bestAisles);
				}
			} catch (Exception e) {
				System.out.println("[THREAD K=" + k_actual + "] Timeout o error: " + e.getMessage());
			}
			k_actual++;
		}

		System.out.println("[INFO] Mejor solución previa a CP-SAT: Ratio=" + bestRatio + ", Pasillos=" + bestAisles
				+ ", Pedidos=" + bestOrders);

		// ============ Resto: CP-SAT REFINAMIENTO ============
		long elapsed = stopWatch.getTime(TimeUnit.MILLISECONDS);
		double remainingSec = Math.max(0, (MAX_RUNTIME - elapsed - 5000) / 1000.0);

		System.out.println("[INFO] Refinando con CP-SAT, tiempo restante (s): " + remainingSec);

		CpModel model = new CpModel();

		int O = orders.size(), A = aisles.size();
		BoolVar[] x = new BoolVar[O];
		for (int o = 0; o < O; o++) {
			x[o] = model.newBoolVar("x_" + o);
		}
		BoolVar[] y = new BoolVar[A];
		for (int a = 0; a < A; a++) {
			y[a] = model.newBoolVar("y_" + a);
		}

		for (int i = 0; i < nItems; i++) {
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

		LinearExprBuilder waveSum = LinearExpr.newBuilder();
		for (int o = 0; o < O; o++) {
			waveSum.addTerm(x[o], orderSizes[o]);
		}
		model.addGreaterOrEqual(waveSum, waveSizeLB);
		model.addLessOrEqual(waveSum, waveSizeUB);

		IntVar totalUnits = model.newIntVar(0, Arrays.stream(orderSizes).sum(), "totalUnits");
		model.addEquality(totalUnits, waveSum);

		IntVar aisleCount = model.newIntVar(0, A, "aisleCount");
		model.addEquality(aisleCount, LinearExpr.sum(y));

		long scale = (long) Math.floor(1000.0 * bestRatio);
		LinearExprBuilder objExpr = LinearExpr.newBuilder()
				.addTerm(totalUnits, 1000)
				.addTerm(aisleCount, -scale);
		model.maximize(objExpr);

		for (int o : bestOrders)
			model.addHint(x[o], 1);
		for (int a : bestAisles)
			model.addHint(y[a], 1);

		CpSolver solver = new CpSolver();
		solver.getParameters().setMaxTimeInSeconds(remainingSec - 10);
		CpSolverStatus status = solver.solve(model);

		Set<Integer> finalOrders = new HashSet<>();
		Set<Integer> finalAisles = new HashSet<>();
		double finalRatio = bestRatio;
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
			System.out.println("[INFO] CP-SAT: Mejorado a Ratio=" + finalRatio + ", Pasillos=" + finalAisles
					+ ", Pedidos=" + finalOrders);
		} else {
			System.out.println("[INFO] CP-SAT: No mejoró la solución previa.");
			finalOrders = bestOrders;
			finalAisles = bestAisles;
		}
		return new ChallengeSolution(finalOrders, finalAisles);
	}

	static class KResult {
		public final double ratio;
		public final Set<Integer> aisles;
		public final Set<Integer> orders;

		KResult(double ratio, Set<Integer> aisles, Set<Integer> orders) {
			this.ratio = ratio;
			this.aisles = aisles;
			this.orders = orders;
		}
	}

	protected long getRemainingTime(StopWatch sw) {
		return Math.max(
				TimeUnit.MILLISECONDS.toMillis(MAX_RUNTIME) - sw.getTime(TimeUnit.MILLISECONDS),
				0L);
	}
}

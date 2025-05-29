package org.sbpo2025.challenge;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.apache.commons.lang3.time.StopWatch;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import com.google.ortools.Loader;
// --- para tu knapsack con MPSolver ---
import com.google.ortools.linearsolver.MPConstraint;
import com.google.ortools.linearsolver.MPObjective;
import com.google.ortools.linearsolver.MPSolver;
import com.google.ortools.linearsolver.MPVariable;
// --- para el CP-SAT de refinamiento ---
import com.google.ortools.sat.CpModel;
import com.google.ortools.sat.CpSolver;
import com.google.ortools.sat.CpSolverStatus;
import com.google.ortools.sat.LinearExpr;
import com.google.ortools.sat.LinearExprBuilder;
import com.google.ortools.sat.BoolVar;
import com.google.ortools.sat.IntVar;

/**
 * ChallengeSolver con fases: inicial (top-K), búsqueda local aleatoria (Random
 * Walk),
 * GRASP multistart (construcción + LS), y refinamiento CP-SAT.
 */
public class ChallengeSolver {
	static {
		Loader.loadNativeLibraries();
	}

	private final long MAX_RUNTIME = 600_000; // 10 minutos en ms
	protected final List<Map<Integer, Integer>> orders;
	protected final List<Map<Integer, Integer>> aisles;
	protected final int nItems;
	protected final int waveSizeLB;
	protected final int waveSizeUB;

	// Parámetros GRASP
	private final double alpha = 0.3; // nivel de aleatoriedad en GRASP
	private final int maxGraspIters = 50; // iteraciones GRASP

	// Resultado global compartido entre fases
	private volatile double bestRatio;
	private volatile Set<Integer> bestAisles;
	private volatile Set<Integer> bestOrders;

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
		this.bestRatio = Double.NEGATIVE_INFINITY;
		this.bestAisles = new HashSet<>();
		this.bestOrders = new HashSet<>();
	}

	/** Flujo principal de la solución. */
	public ChallengeSolution solve(StopWatch sw) {
		int m = aisles.size();
		System.out.println("[INFO] Cantidad de pasillos = " + m);

		// Pre-cálculos: tamaños de órdenes y capacidad total de cada pasillo
		int[] orderSizes = new int[orders.size()];
		for (int o = 0; o < orders.size(); o++) {
			orderSizes[o] = orders.get(o).values().stream().mapToInt(i -> i).sum();
		}
		double[] aisleTotals = new double[m];
		for (int a = 0; a < m; a++) {
			aisleTotals[a] = aisles.get(a).values().stream().mapToInt(i -> i).sum();
		}

		// 1) Fase inicial: top-K knapsack paralelo
		runInitialPhase(aisleTotals, orderSizes, sw);

		// 2) Búsqueda local aleatoria (Random Walk)
		System.out.println("[RW] Iniciando búsqueda local aleatoria");
		randomWalk(sw, orderSizes);
		System.out.println("[RW] Finalizado RW, mejor ratio=" + bestRatio + " pasillos=" + bestAisles);

		// System.out.println("[RW] Iniciando búsqueda local aleatoria 2");
		// randomWalk(sw, orderSizes);
		// System.out.println("[RW] Finalizado RW 2, mejor ratio=" + bestRatio + "
		// pasillos=" + bestAisles);

		// 3) GRASP multistart tras RW
		// System.out.println("[GRASP] Iniciando GRASP multistart");
		// KResult graspRes = runGRASP(bestAisles, orderSizes, sw);
		// updateBest(graspRes);
		// System.out.println("[GRASP] Mejor GRASP: ratio=" + bestRatio + " pasillos=" +
		// bestAisles);

		// 4) Refinamiento final con CP-SAT
		double remSec = Math.max(0,
				(MAX_RUNTIME - sw.getTime(TimeUnit.MILLISECONDS) - 5000) / 1000.0);
		System.out.println("[INFO] Refinando con CP-SAT, tiempo restante (s)=" + remSec);
		return cpSatDinkelbach(bestOrders, bestAisles, orderSizes, sw, remSec);
	}

	// ========================= FASE INICIAL =========================
	private void runInitialPhase(double[] aisleTotals, int[] orderSizes, StopWatch sw) {
		int m = aisleTotals.length;
		int candidateK = Math.max(1, (int) Math.round(m / 9.0));
		int maxK = Math.min(m, candidateK);
		System.out.println("[INIT] Iniciando fase inicial K=1.." + maxK);
		ExecutorService exec = Executors.newFixedThreadPool(
				Math.min(maxK, Runtime.getRuntime().availableProcessors()));
		List<Future<KResult>> futures = new ArrayList<>();
		for (int k = 1; k <= maxK; k++) {
			final int fk = k;
			futures.add(exec.submit(() -> solveFixedTopK(fk, aisleTotals, orderSizes, sw)));
		}
		exec.shutdown();
		collectKResults(futures, sw);

		// Fallback greedy si no se encontró solución
		if (bestRatio == Double.NEGATIVE_INFINITY) {
			System.out.println("[INIT] Fallback greedy factible...");
			KResult kr = greedyFeasible(new HashSet<>(), orderSizes, sw);
			updateBest(kr);
		}
		System.out.println("[INIT] Fase inicial completa: ratio=" + bestRatio + " pasillos=" + bestAisles);
	}

	private KResult solveFixedTopK(int k, double[] totals, int[] orderSizes, StopWatch sw) {
		System.out.println("[INIT K=" + k + "] seleccionando top-" + k);
		Integer[] idx = new Integer[totals.length];
		for (int i = 0; i < totals.length; i++)
			idx[i] = i;
		Arrays.sort(idx, (a, b) -> Double.compare(totals[b], totals[a]));
		Set<Integer> chosen = new HashSet<>();
		for (int i = 0; i < k; i++)
			chosen.add(idx[i]);
		return solveFixed(chosen, orderSizes, getRemainingTime(sw), "INIT");
	}

	private void collectKResults(List<Future<KResult>> futures, StopWatch sw) {
		int kAct = 1;
		for (Future<KResult> f : futures) {
			try {
				KResult kr = f.get(Math.max(100, getRemainingTime(sw)), TimeUnit.MILLISECONDS);
				if (kr != null) {
					System.out.println("[INIT K=" + kAct + "] ratio=" + kr.ratio + " pasillos=" + kr.aisles);
					updateBest(kr);
				}
			} catch (Exception e) {
				System.out.println("[INIT K=" + kAct + "] error: " + e.getMessage());
			}
			kAct++;
		}
	}

	private KResult greedyFeasible(Set<Integer> seed, int[] orderSizes, StopWatch sw) {
		Set<Integer> candidate = new HashSet<>(seed);
		Set<Integer> remaining = new HashSet<>();
		for (int i = 0; i < aisles.size(); i++)
			if (!candidate.contains(i))
				remaining.add(i);
		while (!remaining.isEmpty() && getRemainingTime(sw) > 0) {
			int best = -1, bestCap = -1;
			for (int a : remaining) {
				int cap = aisles.get(a).values().stream().mapToInt(i -> i).sum();
				if (cap > bestCap) {
					bestCap = cap;
					best = a;
				}
			}
			if (best < 0)
				break;
			candidate.add(best);
			remaining.remove(best);
			KResult kr = solveFixed(candidate, orderSizes, getRemainingTime(sw), "GREEDY");
			if (kr != null)
				return kr;
		}
		return null;
	}

	// ========================= FASE RANDOM WALK =========================
	/**
	 * Random Walk (búsqueda local aleatoria) usando tu código original,
	 * con los mismos prints, adaptada a la firma de “randomWalk”.
	 *
	 * @param sw         StopWatch para controlar el timeout total.
	 * @param orderSizes arreglo pre-calculado de tamaños de órdenes.
	 */
	private void randomWalk(StopWatch sw, int[] orderSizes) {
		int m = aisles.size();
		long startLocal = System.currentTimeMillis();
		long remaining = Math.max(0, getRemainingTime(sw) - 5_000L);
		long localTime = Math.min(remaining, (m > 250 ? 240_000L : 120_000L));

		System.out.println("[LOCAL] Iniciando búsqueda local por " + localTime + " ms");
		ExecutorService localExec = Executors.newFixedThreadPool(
				Math.min(4, Runtime.getRuntime().availableProcessors()));

		for (int t = 0; t < Math.min(4, Runtime.getRuntime().availableProcessors()); t++) {
			localExec.submit(() -> {
				while (System.currentTimeMillis() - startLocal < localTime) {
					// generar vecino
					List<Integer> vec = new ArrayList<>(bestAisles);
					int act = ThreadLocalRandom.current().nextInt(3);
					if (act == 0 && !vec.isEmpty()) {
						int rem = vec.remove(ThreadLocalRandom.current().nextInt(vec.size()));
						// System.out.println("[LOCAL] remove " + rem);
					} else if (act == 1 && vec.size() < m) {
						int add;
						do {
							add = ThreadLocalRandom.current().nextInt(m);
						} while (vec.contains(add));
						vec.add(add);
						// System.out.println("[LOCAL] add " + add);
					} else if (act == 2 && !vec.isEmpty()) {
						int idxSwap = ThreadLocalRandom.current().nextInt(vec.size());
						int old = vec.get(idxSwap);
						int add;
						do {
							add = ThreadLocalRandom.current().nextInt(m);
						} while (vec.contains(add));
						vec.set(idxSwap, add);
						// System.out.println("[LOCAL] swap " + old + "->" + add);
					}

					// evaluar vecino
					KResult kr = solveFixed(new HashSet<>(vec), orderSizes, getRemainingTime(sw), "LOCAL");
					if (kr != null) {
						// System.out.println("[LOCAL] probado " + vec + " ratio=" + kr.ratio);
						synchronized (this) {
							if (kr.ratio > bestRatio) {
								bestRatio = kr.ratio;
								bestAisles = new HashSet<>(vec);
								bestOrders = new HashSet<>(kr.orders);
								System.out.println("[LOCAL] ** mejora nueva ratio="
										+ bestRatio + " aisles=" + bestAisles);
							}
						}
					} else {
						// System.out.println("[LOCAL] probado " + vec + " infactible");
					}
				}
			});
		}

		localExec.shutdown();
		try {
			localExec.awaitTermination(localTime + 1000, TimeUnit.MILLISECONDS);
		} catch (InterruptedException ignored) {
			Thread.currentThread().interrupt();
		}
		System.out.println("[LOCAL] Búsqueda local finalizada: ratio="
				+ bestRatio + " aisles=" + bestAisles);
	}

	// ========================= FASE GRASP =========================
	private KResult runGRASP(Set<Integer> seed, int[] orderSizes, StopWatch sw) {
		KResult bestGrasp = new KResult(bestRatio, seed, bestOrders);
		// Semilla que iremos actualizando
		Set<Integer> currentSeed = new HashSet<>(seed);

		for (int it = 0; it < maxGraspIters && getRemainingTime(sw) > 0; it++) {
			System.out.println("[GRASP] iter=" + it);
			// Construcción greedy-aleatorizada a partir de la semilla actual
			Set<Integer> sol = greedyRandomizedConstruct(currentSeed, orderSizes, sw);
			KResult kr = solveFixed(sol, orderSizes, getRemainingTime(sw), "GRC");
			if (kr != null) {
				// Mejora local
				KResult improved = sequentialLocalSearch(kr, orderSizes, sw);
				if (improved.ratio > bestGrasp.ratio) {
					bestGrasp = improved;
					// Actualizamos la semilla para la siguiente iteración
					currentSeed = new HashSet<>(improved.aisles);
					System.out.println("[GRASP] mejora iter=" + it
							+ " ratio=" + improved.ratio);
				}
			}
		}
		return bestGrasp;
	}

	private Set<Integer> greedyRandomizedConstruct(Set<Integer> reference,
			int[] orderSizes,
			StopWatch sw) {
		// 1) Partimos de la mejor solución hasta ahora:
		Set<Integer> sol = new HashSet<>(reference);

		// 2) Rompemos el óptimo local eliminando un pasillo al azar
		if (!sol.isEmpty()) {
			int rem = sol.iterator().next();
			sol.remove(rem);
			System.out.println("[GRC] diversifico eliminando pasillo=" + rem);
		}

		int targetSize = reference.size();
		// 3) Completamos hasta volver al tamaño inicial
		while (sol.size() < targetSize && getRemainingTime(sw) > 0) {
			// Calcular ganancia marginal para cada candidato
			Map<Integer, Double> delta = new HashMap<>();
			for (int a = 0; a < aisles.size(); a++) {
				if (!sol.contains(a)) {
					Set<Integer> cand = new HashSet<>(sol);
					cand.add(a);
					KResult kr = solveFixed(cand, orderSizes, getRemainingTime(sw), "GRC");
					delta.put(a, (kr != null) ? kr.ratio : Double.NEGATIVE_INFINITY);
				}
			}
			if (delta.isEmpty())
				break;

			double maxD = Collections.max(delta.values());
			double minD = Collections.min(delta.values());
			double thr = maxD - alpha * (maxD - minD);

			// Construir la RCL y escoger aleatoriamente
			List<Integer> rcl = delta.entrySet().stream()
					.filter(e -> e.getValue() >= thr)
					.map(Map.Entry::getKey)
					.collect(Collectors.toList());

			int pick = rcl.get(ThreadLocalRandom.current().nextInt(rcl.size()));
			sol.add(pick);
			System.out.println("[GRC] add pasillo=" + pick);
		}

		return sol;
	}

	private KResult sequentialLocalSearch(KResult start, int[] orderSizes, StopWatch sw) {
		Set<Integer> currentA = new HashSet<>(start.aisles);
		double currentR = start.ratio;
		Set<Integer> currentO = new HashSet<>(start.orders);
		boolean improved;
		do {
			improved = false;
			for (int a : new HashSet<>(currentA)) {
				Set<Integer> cand = new HashSet<>(currentA);
				cand.remove(a);
				KResult kr = solveFixed(cand, orderSizes, getRemainingTime(sw), "LS");
				if (kr != null && kr.ratio > currentR) {
					currentA = kr.aisles;
					currentO = kr.orders;
					currentR = kr.ratio;
					System.out.println("[LS] remove mejora=" + a + " ratio=" + currentR);
					improved = true;
					break;
				}
			}
			if (!improved) {
				for (int a = 0; a < aisles.size(); a++) {
					if (!currentA.contains(a)) {
						Set<Integer> cand = new HashSet<>(currentA);
						cand.add(a);
						KResult kr = solveFixed(cand, orderSizes, getRemainingTime(sw), "LS");
						if (kr != null && kr.ratio > currentR) {
							currentA = kr.aisles;
							currentO = kr.orders;
							currentR = kr.ratio;
							System.out.println("[LS] add mejora=" + a + " ratio=" + currentR);
							improved = true;
							break;
						}
					}
				}
			}
		} while (improved && getRemainingTime(sw) > 0);
		return new KResult(currentR, currentA, currentO);
	}

	// ========================= KNAPSACK (MPSolver) =========================
	private KResult solveFixed(
			Set<Integer> chosenA,
			int[] orderSizes,
			long remaining,
			String tag) {
		// Construir capacidad por ítem
		int[] cap = new int[nItems];
		for (int a : chosenA) {
			for (var e : aisles.get(a).entrySet()) {
				cap[e.getKey()] += e.getValue();
			}
		}
		MPSolver solver = MPSolver.createSolver("SCIP");
		if (solver == null)
			return null;
		solver.setTimeLimit(Math.max(200, remaining / 10));
		int O = orders.size();
		MPVariable[] x = new MPVariable[O];
		for (int o = 0; o < O; o++)
			x[o] = solver.makeBoolVar("x_" + o);
		// Restricciones de capacidad
		for (int i = 0; i < nItems; i++) {
			MPConstraint c = solver.makeConstraint(0, cap[i], "item_" + i);
			for (int o = 0; o < O; o++) {
				int q = orders.get(o).getOrDefault(i, 0);
				if (q > 0)
					c.setCoefficient(x[o], q);
			}
		}
		// Restricciones waveSize LB/UB
		MPConstraint lb = solver.makeConstraint(waveSizeLB, Double.POSITIVE_INFINITY, "lb");
		MPConstraint ub = solver.makeConstraint(Double.NEGATIVE_INFINITY, waveSizeUB, "ub");
		for (int o = 0; o < O; o++) {
			lb.setCoefficient(x[o], orderSizes[o]);
			ub.setCoefficient(x[o], orderSizes[o]);
		}
		// Objetivo: maximizar unidades / pasillos
		MPObjective obj = solver.objective();
		for (int o = 0; o < O; o++)
			obj.setCoefficient(x[o], orderSizes[o]);
		obj.setMaximization();
		MPSolver.ResultStatus status = solver.solve();
		if (status == MPSolver.ResultStatus.OPTIMAL || status == MPSolver.ResultStatus.FEASIBLE) {
			double totalUnits = obj.value();
			double ratio = totalUnits / Math.max(1, chosenA.size());
			Set<Integer> selOrders = new HashSet<>();
			for (int o = 0; o < O; o++) {
				if (x[o].solutionValue() > 0.5)
					selOrders.add(o);
			}
			return new KResult(ratio, chosenA, selOrders);
		}
		return null;
	}

	// ========================= REFINAMIENTO CP-SAT =========================
	/**
	 * Refinamiento iterativo con método de Dinkelbach usando CP-SAT.
	 * Imprime el estado (N, D, gap, ratio) en cada iteración.
	 *
	 * @param ordersSeed Órdenes semilla para hints
	 * @param aislesSeed Pasillos semilla para hints
	 * @param orderSizes Pre-cálculo de tamaños de órdenes
	 * @param sw         StopWatch para controlar timeout total
	 * @param remSec     Segundos restantes para CP-SAT
	 * @return Solución refinada
	 */

	/**
	 * Refinamiento iterativo con Dinkelbach, iniciando λ en el ratio de la semilla.
	 */
	private ChallengeSolution cpSatDinkelbach(
			Set<Integer> ordersSeed,
			Set<Integer> aislesSeed,
			int[] orderSizes,
			StopWatch sw,
			double remSec) {
		int O = orders.size(), A = aisles.size();
		// Empezamos λ en el ratio actual (de bestRatio o calculado a partir de la
		// semilla)
		double lambda = bestRatio;
		double tol = 1e-6;
		double bestRatioLocal = lambda;
		ChallengeSolution bestSol = new ChallengeSolution(bestOrders, bestAisles);

		long deadline = System.currentTimeMillis() + (long) ((remSec - 1.0) * 1000);

		for (int it = 1; it <= 50 && System.currentTimeMillis() < deadline; it++) {
			System.out.println("[DINK] Iteración " + it + " con λ=" + lambda);

			CpModel model = new CpModel();
			BoolVar[] x = new BoolVar[O];
			for (int i = 0; i < O; i++)
				x[i] = model.newBoolVar("x_" + i);
			BoolVar[] y = new BoolVar[A];
			for (int i = 0; i < A; i++)
				y[i] = model.newBoolVar("y_" + i);

			// Restricciones de capacidad
			for (int item = 0; item < nItems; item++) {
				LinearExprBuilder lhs = LinearExpr.newBuilder();
				for (int o = 0; o < O; o++) {
					int u = orders.get(o).getOrDefault(item, 0);
					if (u > 0)
						lhs.addTerm(x[o], u);
				}
				LinearExprBuilder rhs = LinearExpr.newBuilder();
				for (int a = 0; a < A; a++) {
					int u = aisles.get(a).getOrDefault(item, 0);
					if (u > 0)
						rhs.addTerm(y[a], u);
				}
				model.addLessOrEqual(lhs, rhs);
			}

			// LB/UB de wave
			LinearExprBuilder waveSum = LinearExpr.newBuilder();
			for (int o = 0; o < O; o++)
				waveSum.addTerm(x[o], orderSizes[o]);
			model.addGreaterOrEqual(waveSum, waveSizeLB);
			model.addLessOrEqual(waveSum, waveSizeUB);

			// Numerador y denominador
			LinearExprBuilder numer = LinearExpr.newBuilder();
			for (int o = 0; o < O; o++)
				numer.addTerm(x[o], orderSizes[o]);
			LinearExprBuilder denom = LinearExpr.newBuilder();
			for (int a = 0; a < A; a++)
				denom.addTerm(y[a], 1);
			model.addGreaterOrEqual(denom, 1);

			// Objetivo Dinkelbach
			// Redondea λ al entero hacia arriba
			long lambdaInt = (long) Math.ceil(lambda);

			// Construye la expresión objetivo manualmente
			LinearExprBuilder objB = LinearExpr.newBuilder();
			// N(x)
			for (int o = 0; o < O; o++) {
				objB.addTerm(x[o], orderSizes[o]);
			}
			// –λ·D(y)
			for (int a = 0; a < A; a++) {
				objB.addTerm(y[a], -lambdaInt);
			}
			// Pasa un LinearExpr construido a maximize
			model.maximize(objB.build());

			// Hints
			for (int o : ordersSeed)
				model.addHint(x[o], 1);
			for (int a : aislesSeed)
				model.addHint(y[a], 1);

			CpSolver solver = new CpSolver();
			solver.getParameters().setMaxTimeInSeconds(
					Math.max(1.0, (deadline - System.currentTimeMillis()) / 1000.0));
			CpSolverStatus status = solver.solve(model);
			if (status != CpSolverStatus.OPTIMAL && status != CpSolverStatus.FEASIBLE) {
				System.out.println("[DINK] sin solución en iteración " + it);
				break;
			}

			// Extraer valores
			double Nval = 0, Dval = 0;
			Set<Integer> solO = new HashSet<>(), solA = new HashSet<>();
			for (int o = 0; o < O; o++)
				if (solver.booleanValue(x[o])) {
					Nval += orderSizes[o];
					solO.add(o);
				}
			for (int a = 0; a < A; a++)
				if (solver.booleanValue(y[a])) {
					Dval += 1;
					solA.add(a);
				}

			double gap = Nval - lambda * Dval;
			double currRatio = Nval / Dval;
			System.out.printf(
					"[DINK] Iter %d: N=%.2f, D=%.2f, gap=%.2e, ratio=%.6f%n",
					it, Nval, Dval, gap, currRatio);

			if (Math.abs(gap) <= tol) {
				bestRatioLocal = currRatio;
				bestSol = new ChallengeSolution(solO, solA);
				System.out.println("[DINK] convergió en iter " + it + " ratio=" + currRatio);
				break;
			}

			if (currRatio > bestRatioLocal) {
				bestRatioLocal = currRatio;
				bestSol = new ChallengeSolution(solO, solA);
				ordersSeed = solO;
				aislesSeed = solA;
			}
			lambda = currRatio;
		}

		System.out.println("[DINK] ratio final=" + bestRatioLocal);
		return bestSol;
	}

	// ========================= UTILITIES =========================
	protected long getRemainingTime(StopWatch sw) {
		return Math.max(0, MAX_RUNTIME - sw.getTime(TimeUnit.MILLISECONDS));
	}

	private synchronized void updateBest(KResult kr) {
		if (kr != null && kr.ratio > bestRatio) {
			bestRatio = kr.ratio;
			bestAisles = new HashSet<>(kr.aisles);
			bestOrders = new HashSet<>(kr.orders);
		}
	}

	// Clase para resultados intermedios (ratio, aisles, orders)
	static class KResult {
		public final double ratio;
		public final Set<Integer> aisles;
		public final Set<Integer> orders;

		public KResult(double ratio, Set<Integer> aisles, Set<Integer> orders) {
			this.ratio = ratio;
			this.aisles = aisles;
			this.orders = orders;
		}
	}
}

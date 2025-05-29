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
import java.util.concurrent.ThreadLocalRandom;

public class ChallengeSolver {
	static {
		Loader.loadNativeLibraries();
	}

	private final long MAX_RUNTIME = 600_000; // 10 minutes in ms
	protected final List<Map<Integer, Integer>> orders;
	protected final List<Map<Integer, Integer>> aisles;
	protected final int nItems;
	protected final int waveSizeLB;
	protected final int waveSizeUB;

	// Campos para compartir resultados entre lambdas
	private volatile double bestRatio;
	private volatile Set<Integer> bestAisles;
	private volatile Set<Integer> bestOrders;

	public ChallengeSolver(List<Map<Integer, Integer>> orders,
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

	@SuppressWarnings("ConstantConditions")
	public ChallengeSolution solve(StopWatch sw) {
		int m = aisles.size();
		System.out.println("[INFO] Cantidad de pasillos " + m);
		// Cálculo de cuántos pasillos considerar como candidatos en heurística inicial
		int candidateK = (int) Math.round(m / 9.0); // proporcional al total de pasillos
		int maxK = Math.min(m, Math.max(1, candidateK)); // pero no más que m

		// Pre-calcular tamaño de las ordenes
		int[] orderSizes = new int[orders.size()];
		for (int o = 0; o < orders.size(); o++) {
			int sum = 0;
			for (int q : orders.get(o).values())
				sum += q;
			orderSizes[o] = sum;
		}
		// Pre-calcular capacidad total de cada pasillo
		double[] aisleTotals = new double[m];
		for (int a = 0; a < m; a++) {
			int sum = 0;
			for (int q : aisles.get(a).values())
				sum += q;
			aisleTotals[a] = sum;
		}

		// 1) Heurística inicial paralela knapsack por K
		System.out.println("[INFO] Iniciando heurística inicial K=2.." + maxK);
		ExecutorService exec = Executors.newFixedThreadPool(
				Math.min(maxK, Runtime.getRuntime().availableProcessors()));
		List<Future<KResult>> futures = new ArrayList<>();
		for (int k = 1; k <= maxK; k++) {
			final int fk = k;
			futures.add(exec.submit(() -> {
				long remaining = getRemainingTime(sw);
				if (remaining <= 0) {
					System.out.println("[THREAD K=" + fk + "] tiempo agotado");
					return null;
				}
				// seleccionar top-k por capacidad total
				Integer[] idx = new Integer[m];
				for (int i = 0; i < m; i++)
					idx[i] = i;
				Arrays.sort(idx, (a, b) -> Double.compare(aisleTotals[b], aisleTotals[a]));
				Set<Integer> chosen = new HashSet<>();
				for (int i = 0; i < fk; i++)
					chosen.add(idx[i]);
				// evaluar
				KResult kr = solveFixed(chosen, orderSizes, remaining, "INIT K=" + fk);
				return kr;
			}));
		}
		exec.shutdown();
		int kAct = 2;
		boolean foundAny = false;
		for (Future<KResult> f : futures) {
			try {
				KResult kr = f.get(Math.max(100, getRemainingTime(sw)), TimeUnit.MILLISECONDS);
				if (kr != null) {
					foundAny = true;
					System.out.println("[INFO] K=" + kAct + " result ratio=" + kr.ratio + " aisles=" + kr.aisles);
					if (kr.ratio > bestRatio) {
						bestRatio = kr.ratio;
						bestAisles = new HashSet<>(kr.aisles);
						bestOrders = new HashSet<>(kr.orders);
					}
				}
			} catch (Exception e) {
				System.out.println("[THREAD K=" + kAct + "] error: " + e.getMessage());
			}
			kAct++;
		}
		if (!foundAny) {
			System.out.println("[INFO] No se encontró solución con la heurística. Activando fallback greedy...");
			Set<Integer> seed = new HashSet<>();
			KResult kr = greedyFeasible(seed, orderSizes, sw);
			if (kr != null) {
				bestRatio = kr.ratio;
				bestAisles = new HashSet<>(kr.aisles);
				bestOrders = new HashSet<>(kr.orders);
			}
		}
		System.out.println("[INFO] Heurística inicial completa: ratio=" + bestRatio + " aisles=" + bestAisles
				+ " orders=" + bestOrders);

		// 2) Búsqueda local paralela durante 2 minutos
		long startLocal = System.currentTimeMillis();
		long remaining = Math.max(0L, getRemainingTime(sw) - 5_000L); // Me creo otra variable para el caso de negativos
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
						System.out.println("[LOCAL] remove " + rem);
					} else if (act == 1 && vec.size() < m) {
						int add;
						do {
							add = ThreadLocalRandom.current().nextInt(m);
						} while (vec.contains(add));
						vec.add(add);
						System.out.println("[LOCAL] add " + add);
					} else if (act == 2 && !vec.isEmpty()) {
						int idxSwap = ThreadLocalRandom.current().nextInt(vec.size());
						int old = vec.get(idxSwap);
						int add;
						do {
							add = ThreadLocalRandom.current().nextInt(m);
						} while (vec.contains(add));
						vec.set(idxSwap, add);
						System.out.println("[LOCAL] swap " + old + "->" + add);
					}
					KResult kr = solveFixed(new HashSet<>(vec), orderSizes, getRemainingTime(sw), "LOCAL");
					if (kr != null) {
						System.out.println("[LOCAL] probado " + vec + " ratio=" + kr.ratio);
						synchronized (this) {
							if (kr.ratio > bestRatio) {
								bestRatio = kr.ratio;
								bestAisles = new HashSet<>(vec);
								bestOrders = new HashSet<>(kr.orders);
								System.out.println(
										"[LOCAL] ** mejora nueva ratio=" + bestRatio + " aisles=" + bestAisles);
							}
						}
					} else {
						System.out.println("[LOCAL] probado " + vec + " infactible");
					}
				}
			});
		}
		localExec.shutdown();
		try {
			localExec.awaitTermination(localTime + 1000, TimeUnit.MILLISECONDS);
		} catch (Exception ignored) {
		}
		System.out.println("[LOCAL] Búsqueda local finalizada: ratio=" + bestRatio + " aisles=" + bestAisles);

		// 3) Refinamiento CP-SAT
		long elapsed = sw.getTime(TimeUnit.MILLISECONDS);
		double remSec = Math.max(0, (MAX_RUNTIME - elapsed - 5000) / 1000.0);
		System.out.println("[INFO] Refinando con CP-SAT, tiempo restante (s): " + remSec);
		ChallengeSolution refined = cpSatRefinement(bestOrders, bestAisles, orderSizes, sw, remSec);
		return refined;
	}

	private KResult solveFixed(Set<Integer> chosenAisles,
			int[] orderSizes,
			long remaining,
			String tag) {
		// calcula capacidad por ítem
		int[] cap = new int[nItems];
		for (int a : chosenAisles) {
			for (Map.Entry<Integer, Integer> e : aisles.get(a).entrySet())
				cap[e.getKey()] += e.getValue();
		}
		MPSolver solver = MPSolver.createSolver("SCIP");
		if (solver == null) {
			System.out.println("[" + tag + "] no SCIP");
			return null;
		}
		solver.setTimeLimit(Math.max(200, remaining / 10));
		MPVariable[] x = new MPVariable[orders.size()];
		for (int o = 0; o < orders.size(); o++)
			x[o] = solver.makeBoolVar("x_" + o);
		for (int i = 0; i < nItems; i++) {
			MPConstraint c = solver.makeConstraint(0, cap[i], "item_" + i);
			for (int o = 0; o < orders.size(); o++) {
				int q = orders.get(o).getOrDefault(i, 0);
				if (q > 0)
					c.setCoefficient(x[o], q);
			}
		}
		MPConstraint lb = solver.makeConstraint(waveSizeLB, Double.POSITIVE_INFINITY, "lb");
		MPConstraint ub = solver.makeConstraint(Double.NEGATIVE_INFINITY, waveSizeUB, "ub");
		for (int o = 0; o < orders.size(); o++) {
			lb.setCoefficient(x[o], orderSizes[o]);
			ub.setCoefficient(x[o], orderSizes[o]);
		}
		MPObjective obj = solver.objective();
		for (int o = 0; o < orders.size(); o++)
			obj.setCoefficient(x[o], orderSizes[o]);
		obj.setMaximization();
		MPSolver.ResultStatus st = solver.solve();
		if (st == MPSolver.ResultStatus.OPTIMAL || st == MPSolver.ResultStatus.FEASIBLE) {
			double total = obj.value();
			double ratio = total / Math.max(1, chosenAisles.size());
			Set<Integer> sel = new HashSet<>();
			for (int o = 0; o < orders.size(); o++)
				if (x[o].solutionValue() > 0.5)
					sel.add(o);
			return new KResult(ratio, new HashSet<>(chosenAisles), sel);
		}
		return null;
	}

	private ChallengeSolution cpSatRefinement(Set<Integer> bestOrders,
			Set<Integer> bestAisles,
			int[] orderSizes,
			StopWatch sw,
			double remSec) {
		int O = orders.size(), A = aisles.size();
		CpModel model = new CpModel();
		BoolVar[] x = new BoolVar[O];
		for (int o = 0; o < O; o++)
			x[o] = model.newBoolVar("x_" + o);
		BoolVar[] y = new BoolVar[A];
		for (int a = 0; a < A; a++)
			y[a] = model.newBoolVar("y_" + a);
		// restricciones capacidad
		for (int i = 0; i < nItems; i++) {
			LinearExprBuilder left = LinearExpr.newBuilder();
			for (int o = 0; o < O; o++) {
				int u = orders.get(o).getOrDefault(i, 0);
				if (u > 0)
					left.addTerm(x[o], u);
			}
			LinearExprBuilder right = LinearExpr.newBuilder();
			for (int a = 0; a < A; a++) {
				int u = aisles.get(a).getOrDefault(i, 0);
				if (u > 0)
					right.addTerm(y[a], u);
			}
			model.addLessOrEqual(left, right);
		}
		// restricciones wave
		LinearExprBuilder waveSum = LinearExpr.newBuilder();
		for (int o = 0; o < O; o++)
			waveSum.addTerm(x[o], orderSizes[o]);
		model.addGreaterOrEqual(waveSum, waveSizeLB);
		model.addLessOrEqual(waveSum, waveSizeUB);
		IntVar totalUnits = model.newIntVar(0, Arrays.stream(orderSizes).sum(), "totalUnits");
		model.addEquality(totalUnits, waveSum);
		IntVar aisleCount = model.newIntVar(0, A, "aisleCount");
		model.addEquality(aisleCount, LinearExpr.sum(y));
		long scale = (long) Math.floor(1000.0 * bestRatio);
		LinearExprBuilder obj = LinearExpr.newBuilder()
				.addTerm(totalUnits, 1000)
				.addTerm(aisleCount, -scale);
		model.maximize(obj);
		for (int o : bestOrders)
			model.addHint(x[o], 1);
		for (int a : bestAisles)
			model.addHint(y[a], 1);
		CpSolver solver = new CpSolver();
		solver.getParameters().setMaxTimeInSeconds(remSec - 1);
		CpSolverStatus st = solver.solve(model);
		Set<Integer> finalO = new HashSet<>(), finalA = new HashSet<>();
		if (st == CpSolverStatus.OPTIMAL || st == CpSolverStatus.FEASIBLE) {
			for (int a = 0; a < A; a++)
				if (solver.booleanValue(y[a]))
					finalA.add(a);
			for (int o = 0; o < O; o++)
				if (solver.booleanValue(x[o]))
					finalO.add(o);
			return new ChallengeSolution(finalO, finalA);
		}
		return new ChallengeSolution(bestOrders, bestAisles);
	}

	private KResult greedyFeasible(Set<Integer> seedAisles, int[] orderSizes, StopWatch sw) {
		Set<Integer> candidate = new HashSet<>(seedAisles);
		Set<Integer> remaining = new HashSet<>();
		for (int i = 0; i < aisles.size(); i++) {
			if (!candidate.contains(i))
				remaining.add(i);
		}
		while (!remaining.isEmpty() && getRemainingTime(sw) > 0) {
			System.out.println("[GREEDY] Intentando con pasillos: " + candidate);
			int best = -1;
			int bestCap = -1;
			for (int a : remaining) {
				int cap = 0;
				for (int q : aisles.get(a).values())
					cap += q;
				if (cap > bestCap) {
					bestCap = cap;
					best = a;
				}
			}
			if (best == -1)
				break;
			candidate.add(best);
			remaining.remove(best);
			KResult kr = solveFixed(candidate, orderSizes, getRemainingTime(sw), "GREEDY");
			if (kr != null)
				return kr;
		}
		System.out.println("[GREEDY] No se halló solución factible.");
		return null;
	}

	protected long getRemainingTime(StopWatch sw) {
		return Math.max(TimeUnit.MILLISECONDS.toMillis(MAX_RUNTIME) - sw.getTime(TimeUnit.MILLISECONDS), 0L);
	}

	static class KResult {
		public final double ratio;
		public final Set<Integer> aisles;
		public final Set<Integer> orders;

		KResult(double r, Set<Integer> a, Set<Integer> o) {
			ratio = r;
			aisles = a;
			orders = o;
		}
	}
}

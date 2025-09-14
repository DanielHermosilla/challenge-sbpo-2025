package org.sbpo2025.challenge;

import org.sbpo2025.challenge.ChallengeSolution;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.apache.commons.lang3.time.StopWatch;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;

import com.google.ortools.Loader;

// --- MIP (knapsack) ---
import com.google.ortools.linearsolver.MPConstraint;
import com.google.ortools.linearsolver.MPObjective;
import com.google.ortools.linearsolver.MPSolver;
import com.google.ortools.linearsolver.MPVariable;

// --- CP-SAT ---
import com.google.ortools.sat.CpModel;
import com.google.ortools.sat.CpSolver;
import com.google.ortools.sat.CpSolverStatus;
import com.google.ortools.sat.LinearExpr;
import com.google.ortools.sat.LinearExprBuilder;
import com.google.ortools.sat.BoolVar;

public class ChallengeSolver {
	static {
		Loader.loadNativeLibraries();
	}

	// ====== Parámetros globales ======
	private final long MAX_RUNTIME = 600_000; // 10 min
	protected final List<Map<Integer, Integer>> orders;
	protected final List<Map<Integer, Integer>> aisles;
	protected final int nItems;
	protected final int waveSizeLB;
	protected final int waveSizeUB;

	// Visitados para RW
	private final Set<Set<Integer>> visitedAisles = Collections.newSetFromMap(new ConcurrentHashMap<>());
	// "Hotness" por pasillo descubierto en RW semilla (para poblar el GA)
	private final Map<Integer, Integer> aisleHotness = new ConcurrentHashMap<>();
	// Pool de subconjuntos élite provenientes del RW semilla
	private final List<Set<Integer>> rwElitePool = Collections.synchronizedList(new ArrayList<>());

	// Resultado global (incumbente)
	private volatile double bestRatio = Double.NEGATIVE_INFINITY;
	private volatile Set<Integer> bestAisles = new HashSet<>();
	private volatile Set<Integer> bestOrders = new HashSet<>();

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

	/** Flujo completo */
	public ChallengeSolution solve(StopWatch sw) {
		int m = aisles.size();
		System.out.println("[INFO] Pasillos=" + m);

		// Pre-cálculos simples
		int[] orderSizes = new int[orders.size()];
		for (int o = 0; o < orders.size(); o++)
			orderSizes[o] = orders.get(o).values().stream().mapToInt(i -> i).sum();

		double[] aisleTotals = new double[m];
		for (int a = 0; a < m; a++)
			aisleTotals[a] = aisles.get(a).values().stream().mapToInt(i -> i).sum();

		// 1) Fase inicial (incumbente)
		runInitialPhase(aisleTotals, orderSizes, sw);

		// 2) Random Walk SEMILLA corto para detectar "mejores pasillos" (hotness) y
		// pool élite
		System.out.println("[SEED-RW] Exploración breve para guiar GA…");
		seedRandomWalk(sw, orderSizes);

		// 3) GA usando incumbente + distribución ponderada por hotness + pool élite RW
		System.out.println("[GA] Iniciando GA con incumbente + hotness RW");
		runGeneticPhase(orderSizes, sw);
		System.out.println("[GA] Terminado | ratio=" + bestRatio + " | aisles=" + bestAisles);

		// 4) Random Walk intensivo (explotación local final)
		System.out.println("[RW] Intensificación final");
		randomWalk(sw, orderSizes);
		System.out.println("[RW] Finalizado | ratio=" + bestRatio + " | aisles=" + bestAisles);

		// 5) Refinamiento CP-SAT (Dinkelbach)
		double remSec = Math.max(0, (MAX_RUNTIME - sw.getTime(TimeUnit.MILLISECONDS) - 5000) / 1000.0);
		System.out.println("[INFO] CP-SAT con tiempo restante(s)=" + remSec);
		return cpSatDinkelbach(bestOrders, bestAisles, orderSizes, sw, remSec);
	}

	// ========================= FASE INICIAL =========================
	private void runInitialPhase(double[] aisleTotals, int[] orderSizes, StopWatch sw) {
		int m = aisleTotals.length;
		int candidateK = Math.max(1, (int) Math.round(m / 9.0));
		int maxK = Math.min(m, candidateK);
		System.out.println("[INIT] K=1.." + maxK);

		ExecutorService exec = Executors.newFixedThreadPool(Math.min(maxK, Runtime.getRuntime().availableProcessors()));
		List<Future<KResult>> futures = new ArrayList<>();
		for (int k = 1; k <= maxK; k++) {
			final int fk = k;
			futures.add(exec.submit(() -> solveFixedTopK(fk, aisleTotals, orderSizes, sw)));
		}
		exec.shutdown();
		collectKResults(futures, sw);

		if (bestRatio == Double.NEGATIVE_INFINITY) {
			System.out.println("[INIT] Greedy fallback…");
			KResult kr = greedyFeasible(new HashSet<>(), orderSizes, sw);
			updateBest(kr);
		}
		System.out.println("[INIT] Hecho | ratio=" + bestRatio + " | aisles=" + bestAisles);
	}

	private KResult solveFixedTopK(int k, double[] totals, int[] orderSizes, StopWatch sw) {
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
					System.out.println("[INIT K=" + kAct + "] ratio=" + kr.ratio + " aisles=" + kr.aisles);
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

	// ========================= RW SEMILLA PARA GA =========================
	/**
	 * RW corto para construir "hotness" por pasillo y un pool élite de
	 * subconjuntos.
	 */
	private void seedRandomWalk(StopWatch sw, int[] orderSizes) {
		final int m = aisles.size();
		long budget = Math.max(3_000L, Math.min(20_000L, getRemainingTime(sw) / 10)); // 10% del tiempo restante
		long start = System.currentTimeMillis();

		Set<Integer> current = new HashSet<>(bestAisles);
		double currScore = bestRatio;

		while (System.currentTimeMillis() - start < budget) {
			List<Integer> vec = new ArrayList<>(current);
			int act = ThreadLocalRandom.current().nextInt(3);
			if (act == 0 && !vec.isEmpty()) {
				vec.remove(ThreadLocalRandom.current().nextInt(vec.size()));
			} else if (act == 1 && vec.size() < m) {
				int add;
				do
					add = ThreadLocalRandom.current().nextInt(m);
				while (vec.contains(add));
				vec.add(add);
			} else if (act == 2 && !vec.isEmpty()) {
				int idxSwap = ThreadLocalRandom.current().nextInt(vec.size());
				int add;
				do
					add = ThreadLocalRandom.current().nextInt(m);
				while (vec.contains(add));
				vec.set(idxSwap, add);
			}

			Set<Integer> cand = new HashSet<>(vec);
			if (!visitedAisles.add(cand))
				continue;

			KResult kr = solveFixed(cand, orderSizes, Math.max(200, getRemainingTime(sw) / 15), "SEED-RW");
			if (kr == null)
				continue;

			// Actualiza hotness por cada pasillo presente en una solución con buen ratio
			double score = kr.ratio;
			if (score >= currScore || ThreadLocalRandom.current().nextDouble() < 0.15) {
				for (int a : cand)
					aisleHotness.merge(a, 1, Integer::sum);
			}
			// Guarda un pool élite pequeño
			if (rwElitePool.size() < 20 || score > currScore) {
				rwElitePool.add(new HashSet<>(cand));
				while (rwElitePool.size() > 30)
					rwElitePool.remove(0);
			}
			if (score > currScore) {
				currScore = score;
				current = cand;
				updateBest(kr);
			}
		}
		if (aisleHotness.isEmpty()) {
			// fallback: da algo de peso a los incumbentes
			for (int a : bestAisles)
				aisleHotness.put(a, 2);
		}
		System.out.println("[SEED-RW] Hotness size=" + aisleHotness.size() + " | ElitePool=" + rwElitePool.size());
	}

	// ========================= RW INTENSIVO =========================
	private void randomWalk(StopWatch sw, int[] orderSizes) {
		int m = aisles.size();
		long startLocal = System.currentTimeMillis();
		long remaining = Math.max(0, getRemainingTime(sw) - 5_000L);
		long localTime = Math.min(remaining, (m > 250 ? 240_000L : 120_000L));

		System.out.println("[LOCAL] Tiempo=" + localTime + " ms");
		ExecutorService localExec = Executors
				.newFixedThreadPool(Math.min(4, Runtime.getRuntime().availableProcessors()));

		for (int t = 0; t < Math.min(4, Runtime.getRuntime().availableProcessors()); t++) {
			localExec.submit(() -> {
				ThreadLocalRandom rnd = ThreadLocalRandom.current();
				while (System.currentTimeMillis() - startLocal < localTime) {
					List<Integer> vec = new ArrayList<>(bestAisles);
					int act = rnd.nextInt(3);
					if (act == 0 && !vec.isEmpty())
						vec.remove(rnd.nextInt(vec.size()));
					else if (act == 1 && vec.size() < m) {
						int add;
						do
							add = rnd.nextInt(m);
						while (vec.contains(add));
						vec.add(add);
					} else if (act == 2 && !vec.isEmpty()) {
						int idxSwap = rnd.nextInt(vec.size());
						int add;
						do
							add = rnd.nextInt(m);
						while (vec.contains(add));
						vec.set(idxSwap, add);
					}
					Set<Integer> cand = new HashSet<>(vec);
					if (!visitedAisles.add(cand))
						continue;

					KResult kr = solveFixed(cand, orderSizes, Math.max(200, getRemainingTime(sw) / 12), "LOCAL");
					if (kr != null && kr.ratio > bestRatio) {
						synchronized (this) {
							updateBest(kr);
						}
						System.out.println("[LOCAL] ** mejora ratio=" + bestRatio + " aisles=" + bestAisles);
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

		System.out.println("[LOCAL] Fin | ratio=" + bestRatio + " | aisles=" + bestAisles);
	}

	// ========================= GA =========================
	/**
	 * GA sobre selección de pasillos. Población inicial:
	 * - Incumbente (mejorAisles) + perturbaciones locales
	 * - Individuos muestreados ponderando pasillos por "hotness" del RW semilla
	 * - Subconjuntos élite del RW
	 */
	private void runGeneticPhase(int[] orderSizes, StopWatch sw) {
		final int m = aisles.size();

		final int POP_SIZE = Math.max(40, Math.min(120, m));
		final int ELITES = Math.max(2, POP_SIZE / 10);
		final int TOURN_K = 3;
		final double PC = 0.9;
		final double PM = Math.min(0.25, 3.0 / Math.max(4, m));
		final int MAX_GENS = 60;
		final long GA_BUDGET_MS = Math.max(5_000L, Math.min(60_000L, getRemainingTime(sw) - 8_000L));

		System.out.printf(Locale.US,
				"[GA] POP=%d elites=%d Pc=%.2f Pm=%.4f gens=%d time=%dms%n",
				POP_SIZE, ELITES, PC, PM, MAX_GENS, GA_BUDGET_MS);

		// Helpers
		final java.util.function.Function<boolean[], Set<Integer>> toSet = (bits) -> {
			Set<Integer> s = new HashSet<>();
			for (int i = 0; i < bits.length; i++)
				if (bits[i])
					s.add(i);
			return s;
		};
		final java.util.function.Function<Set<Integer>, boolean[]> fromSet = (set) -> {
			boolean[] b = new boolean[m];
			for (int a : set)
				if (a >= 0 && a < m)
					b[a] = true;
			return b;
		};

		// ---- Población inicial
		ThreadLocalRandom tlr = ThreadLocalRandom.current();
		List<boolean[]> population = new ArrayList<>(POP_SIZE);

		// 0) Inserta incumbente (si existe)
		boolean[] seed = fromSet.apply(bestAisles);
		population.add(seed.clone());

		// 1) Variaciones del incumbente (flips pequeños)
		while (population.size() < Math.max(5, POP_SIZE / 3)) {
			boolean[] ind = seed.clone();
			int flips = 1 + tlr.nextInt(Math.max(1, Math.max(2, m / 12)));
			for (int f = 0; f < flips; f++) {
				int pos = tlr.nextInt(m);
				ind[pos] = !ind[pos];
			}
			population.add(ind);
		}

		// 2) Individuos tomados de la élite del RW (si hay)
		Collections.shuffle(rwElitePool);
		for (Set<Integer> elite : rwElitePool) {
			if (population.size() >= POP_SIZE)
				break;
			population.add(fromSet.apply(elite));
		}

		// 3) Resto con muestreo ponderado por "hotness" (si no hay hotness ->
		// aleatorio)
		int[] hotKeys = aisleHotness.keySet().stream().mapToInt(Integer::intValue).toArray();
		double[] hotW = Arrays.stream(hotKeys).mapToDouble(k -> Math.max(1, aisleHotness.getOrDefault(k, 1))).toArray();
		double sumW = Arrays.stream(hotW).sum();

		while (population.size() < POP_SIZE) {
			boolean[] ind = new boolean[m];
			int ones = 1 + tlr.nextInt(Math.max(2, m / 6));
			for (int k = 0; k < ones; k++) {
				int a = weightedPick(hotKeys, hotW, sumW, tlr, m);
				ind[a] = true;
			}
			population.add(ind);
		}

		long gaStart = System.currentTimeMillis();
		int gen = 0;
		Scored globalBest = new Scored(fromSet.apply(bestAisles), new KResult(bestRatio, bestAisles, bestOrders));

		while (gen < MAX_GENS && System.currentTimeMillis() - gaStart < GA_BUDGET_MS && getRemainingTime(sw) > 5_000L) {
			final int genIdx = gen + 1;
			final long evalTimeout = Math.max(300, getRemainingTime(sw) / 10);
			final List<boolean[]> popSnapshot = Collections.unmodifiableList(new ArrayList<>(population));

			List<Scored> scored = popSnapshot.parallelStream().unordered()
					.map(ind -> {
						Set<Integer> A = toSet.apply(ind);
						KResult kr = solveFixed(A, orderSizes, evalTimeout, "GA");
						return new Scored(ind, kr);
					}).collect(Collectors.toList());

			double best = scored.stream().mapToDouble(s -> s.score).max().orElse(Double.NEGATIVE_INFINITY);
			double avg = scored.stream().mapToDouble(s -> s.score)
					.filter(d -> d > Double.NEGATIVE_INFINITY / 2).average().orElse(Double.NEGATIVE_INFINITY);
			long feas = scored.stream().filter(s -> s.score > Double.NEGATIVE_INFINITY / 2).count();

			Optional<Scored> bestNow = scored.stream().max(Comparator.comparingDouble(s -> s.score));
			if (bestNow.isPresent() && bestNow.get().score > globalBest.score) {
				globalBest = bestNow.get();
				updateBest(globalBest.kr);
				System.out.printf(Locale.US,
						"[GA][gen=%d] ** MEJORA GLOBAL ** ratio=%.6f | aisles=%s | orders=%s%n",
						genIdx, globalBest.score, globalBest.kr.aisles, globalBest.kr.orders);
			}

			System.out.printf(Locale.US,
					"[GA][gen=%d] feas=%d/%d | best=%.6f | avg=%.6f | t=%.1fs%n",
					genIdx, feas, scored.size(), best, avg, (System.currentTimeMillis() - gaStart) / 1000.0);

			// ---- Elitismo
			scored.sort(Comparator.comparingDouble((Scored s) -> s.score).reversed());
			List<boolean[]> nextPop = new ArrayList<>(POP_SIZE);
			int elitesAdded = 0;
			for (Scored s : scored) {
				if (elitesAdded >= ELITES)
					break;
				if (s.score > Double.NEGATIVE_INFINITY / 2) {
					nextPop.add(s.ind.clone());
					elitesAdded++;
				}
			}
			if (elitesAdded == 0 && !scored.isEmpty())
				nextPop.add(scored.get(0).ind.clone());

			// ---- Selección por torneo + reproducción
			final int poolSize = Math.max(ELITES, Math.min(scored.size(), POP_SIZE));
			final List<Scored> pool = scored.subList(0, poolSize);
			ThreadLocalRandom rnd = ThreadLocalRandom.current();

			while (nextPop.size() < POP_SIZE) {
				boolean[] p1 = tournamentSelect(pool, TOURN_K).ind;
				boolean[] p2 = tournamentSelect(pool, TOURN_K).ind;
				boolean[] c1 = p1.clone(), c2 = p2.clone();
				if (rnd.nextDouble() < PC)
					uniformCrossover(c1, c2, rnd);
				mutateBits(c1, PM, rnd);
				mutateBits(c2, PM, rnd);
				nextPop.add(c1);
				if (nextPop.size() < POP_SIZE)
					nextPop.add(c2);
			}
			population = nextPop;
			gen++;
		}

		if (globalBest.kr != null && globalBest.kr.ratio > bestRatio)
			updateBest(globalBest.kr);
	}

	// Pick ponderado por hotness; si no hay hotness, pick uniforme
	private int weightedPick(int[] hotKeys, double[] hotW, double sumW, ThreadLocalRandom tlr, int m) {
		if (hotKeys.length == 0 || sumW <= 0)
			return tlr.nextInt(m);
		double r = tlr.nextDouble(sumW);
		double acc = 0;
		for (int i = 0; i < hotKeys.length; i++) {
			acc += hotW[i];
			if (r <= acc)
				return hotKeys[i];
		}
		return hotKeys[hotKeys.length - 1];
	}

	// Selección torneo
	private Scored tournamentSelect(List<Scored> pool, int k) {
		ThreadLocalRandom tlr = ThreadLocalRandom.current();
		Scored best = null;
		for (int i = 0; i < k; i++) {
			Scored cand = pool.get(tlr.nextInt(pool.size()));
			if (best == null || cand.score > best.score)
				best = cand;
		}
		return best;
	}

	// Crossover uniforme
	private void uniformCrossover(boolean[] a, boolean[] b, ThreadLocalRandom tlr) {
		for (int i = 0; i < a.length; i++)
			if (tlr.nextBoolean()) {
				boolean t = a[i];
				a[i] = b[i];
				b[i] = t;
			}
	}

	// Mutación bit-flip
	private void mutateBits(boolean[] ind, double p, ThreadLocalRandom tlr) {
		for (int i = 0; i < ind.length; i++)
			if (tlr.nextDouble() < p)
				ind[i] = !ind[i];
	}

	// ========================= MIP (knapsack) =========================
	private KResult solveFixed(Set<Integer> chosenA, int[] orderSizes, long remaining, String tag) {
		int[] cap = new int[nItems];
		for (int a : chosenA)
			for (var e : aisles.get(a).entrySet())
				cap[e.getKey()] += e.getValue();

		MPSolver solver = MPSolver.createSolver("SCIP");
		if (solver == null)
			return null;

		solver.setTimeLimit(Math.max(200, remaining / 10));
		int O = orders.size();
		MPVariable[] x = new MPVariable[O];
		for (int o = 0; o < O; o++)
			x[o] = solver.makeBoolVar("x_" + o);

		for (int i = 0; i < nItems; i++) {
			MPConstraint c = solver.makeConstraint(0, cap[i], "item_" + i);
			for (int o = 0; o < O; o++) {
				int q = orders.get(o).getOrDefault(i, 0);
				if (q > 0)
					c.setCoefficient(x[o], q);
			}
		}
		MPConstraint lb = solver.makeConstraint(waveSizeLB, Double.POSITIVE_INFINITY, "lb");
		MPConstraint ub = solver.makeConstraint(Double.NEGATIVE_INFINITY, waveSizeUB, "ub");
		for (int o = 0; o < O; o++) {
			lb.setCoefficient(x[o], orderSizes[o]);
			ub.setCoefficient(x[o], orderSizes[o]);
		}

		MPObjective obj = solver.objective();
		for (int o = 0; o < O; o++)
			obj.setCoefficient(x[o], orderSizes[o]);
		obj.setMaximization();

		MPSolver.ResultStatus status = solver.solve();
		if (status == MPSolver.ResultStatus.OPTIMAL || status == MPSolver.ResultStatus.FEASIBLE) {
			double totalUnits = obj.value();
			double ratio = totalUnits / Math.max(1, chosenA.size());
			Set<Integer> selOrders = new HashSet<>();
			for (int o = 0; o < O; o++)
				if (x[o].solutionValue() > 0.5)
					selOrders.add(o);
			return new KResult(ratio, chosenA, selOrders);
		}
		return null;
	}

	// ========================= CP-SAT (Dinkelbach) =========================
	private ChallengeSolution cpSatDinkelbach(
			Set<Integer> ordersSeed, Set<Integer> aislesSeed, int[] orderSizes, StopWatch sw, double remSec) {

		int O = orders.size(), A = aisles.size();
		double lambda = bestRatio, tol = 1e-6, bestRatioLocal = lambda;
		ChallengeSolution bestSol = new ChallengeSolution(bestOrders, bestAisles);
		long deadline = System.currentTimeMillis() + (long) ((remSec - 1.0) * 1000);

		for (int it = 1; it <= 50 && System.currentTimeMillis() < deadline; it++) {
			System.out.println("[DINK] iter=" + it + " λ=" + lambda);

			CpModel model = new CpModel();
			BoolVar[] x = new BoolVar[O];
			for (int i = 0; i < O; i++)
				x[i] = model.newBoolVar("x_" + i);
			BoolVar[] y = new BoolVar[A];
			for (int i = 0; i < A; i++)
				y[i] = model.newBoolVar("y_" + i);

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

			LinearExprBuilder waveSum = LinearExpr.newBuilder();
			for (int o = 0; o < O; o++)
				waveSum.addTerm(x[o], orderSizes[o]);
			model.addGreaterOrEqual(waveSum, waveSizeLB);
			model.addLessOrEqual(waveSum, waveSizeUB);

			LinearExprBuilder numer = LinearExpr.newBuilder();
			for (int o = 0; o < O; o++)
				numer.addTerm(x[o], orderSizes[o]);
			LinearExprBuilder denom = LinearExpr.newBuilder();
			for (int a = 0; a < A; a++)
				denom.addTerm(y[a], 1);
			model.addGreaterOrEqual(denom, 1);

			long lambdaInt = (long) Math.ceil(lambda);
			LinearExprBuilder objB = LinearExpr.newBuilder();
			for (int o = 0; o < O; o++)
				objB.addTerm(x[o], orderSizes[o]);
			for (int a = 0; a < A; a++)
				objB.addTerm(y[a], -lambdaInt);
			model.maximize(objB.build());

			for (int o : ordersSeed)
				model.addHint(x[o], 1);
			for (int a : aislesSeed)
				model.addHint(y[a], 1);

			CpSolver solver = new CpSolver();
			int cores = Runtime.getRuntime().availableProcessors();
			solver.getParameters().setNumSearchWorkers(cores);
			solver.getParameters().setMaxTimeInSeconds(Math.max(1.0, (deadline - System.currentTimeMillis()) / 1000.0));
			CpSolverStatus status = solver.solve(model);
			if (status != CpSolverStatus.OPTIMAL && status != CpSolverStatus.FEASIBLE)
				break;

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
			System.out.printf(Locale.US, "[DINK] N=%.2f D=%.2f gap=%.2e ratio=%.6f%n", Nval, Dval, gap, currRatio);

			if (Math.abs(gap) <= tol) {
				bestRatioLocal = currRatio;
				bestSol = new ChallengeSolution(solO, solA);
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

	// ---- GA helper: individuo evaluado ----
	private static final class Scored {
		final boolean[] ind;
		final KResult kr;
		final double score;

		Scored(boolean[] ind, KResult kr) {
			this.ind = ind;
			this.kr = kr;
			this.score = (kr == null ? Double.NEGATIVE_INFINITY : kr.ratio);
		}
	}

	// ---- Resultado knapsack ----
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

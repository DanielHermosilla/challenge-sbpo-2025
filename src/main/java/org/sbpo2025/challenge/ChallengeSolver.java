package org.sbpo2025.challenge;

import com.google.ortools.Loader;
import com.google.ortools.linearsolver.*;

import org.apache.commons.lang3.time.StopWatch;
import java.util.*;
import java.util.concurrent.*;

public class ChallengeSolver {
	static {
		Loader.loadNativeLibraries();
	}

	private final long MAX_RUNTIME = 600_000; // 10 minutos en ms
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
		int maxK = Math.min(m, 5); // Cambia el 5 si quieres otro tope

		double bestRatio = Double.NEGATIVE_INFINITY;
		Set<Integer> bestOrders = Collections.emptySet();
		Set<Integer> bestAisles = Collections.emptySet();

		// Precomputar tamaño total de cada pedido
		int[] orderSizes = new int[orders.size()];
		for (int o = 0; o < orders.size(); o++) {
			int sum = 0;
			for (int qty : orders.get(o).values())
				sum += qty;
			orderSizes[o] = sum;
		}

		// Precomputar la capacidad total de cada pasillo
		double[] aisleTotals = new double[m];
		for (int a = 0; a < m; a++) {
			int sum = 0;
			for (int qty : aisles.get(a).values())
				sum += qty;
			aisleTotals[a] = sum;
		}

		// --------- Paralelización en k ---------
		ExecutorService executor = Executors.newFixedThreadPool(
				Math.min(maxK, Runtime.getRuntime().availableProcessors()));
		List<Future<TabuResult>> futures = new ArrayList<>();

		for (int k = 1; k <= maxK; k++) {
			final int kVal = k;
			futures.add(executor.submit(() -> {
				long remaining = getRemainingTime(stopWatch);
				if (remaining <= 0)
					return null;

				// Heurística inicial para k pasillos de mayor capacidad
				Integer[] idx = new Integer[m];
				for (int a = 0; a < m; a++)
					idx[a] = a;
				Arrays.sort(idx, Comparator.comparingDouble((Integer a) -> aisleTotals[a]).reversed());
				Set<Integer> chosenAisles = new HashSet<>();
				for (int i = 0; i < kVal; i++)
					chosenAisles.add(idx[i]);

				// Resolver knapsack
				KnapsackResult heurRes = solveKnapsack(stopWatch, chosenAisles, orderSizes, remaining);
				if (heurRes == null)
					return null;

				System.out.println("\n=== [k=" + kVal + "] Solución heurística inicial ===");
				System.out.println("Ratio: " + heurRes.ratio);
				System.out.println("Pasillos: " + chosenAisles);
				System.out.println("Pedidos: " + heurRes.selectedOrders);
				System.out.println("==============================================\n");

				// Tabu Search desde heurística
				TabuResult tabuRes = tabuSearchFromInitial(
						chosenAisles, orderSizes, stopWatch, m, aisleTotals, heurRes.selectedOrders, heurRes.ratio,
						kVal);

				if (tabuRes != null) {
					System.out.println("\n=== [k=" + kVal + "] Solución FINAL tras Tabu Search ===");
					System.out.println("Ratio: " + tabuRes.ratio);
					System.out.println("Pasillos: " + tabuRes.aisles);
					System.out.println("Pedidos: " + tabuRes.orders);
					System.out.println("=============================================\n");
				}

				return tabuRes;
			}));
		}

		// Recoger el mejor resultado
		try {
			for (Future<TabuResult> fut : futures) {
				TabuResult res = fut.get();
				if (res != null && res.ratio > bestRatio) {
					bestRatio = res.ratio;
					bestAisles = res.aisles;
					bestOrders = res.orders;
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		executor.shutdown();

		return new ChallengeSolution(bestOrders, bestAisles);
	}

	// ---------------- TABU SEARCH & KNAPSACK HELPERS ----------------------

	private static class TabuResult {
		double ratio;
		Set<Integer> aisles, orders;

		TabuResult(double ratio, Set<Integer> aisles, Set<Integer> orders) {
			this.ratio = ratio;
			this.aisles = aisles;
			this.orders = orders;
		}
	}

	private TabuResult tabuSearchFromInitial(
			Set<Integer> initialAisles,
			int[] orderSizes,
			StopWatch stopWatch,
			int m,
			double[] aisleTotals,
			Set<Integer> initialOrders,
			double initialRatio,
			int k // para los prints
	) {
		Set<Integer> currentAisles = new HashSet<>(initialAisles);
		Set<Integer> currentOrders = new HashSet<>(initialOrders);
		double bestRatio = initialRatio;
		Set<Integer> bestAisles = new HashSet<>(currentAisles);
		Set<Integer> bestOrders = new HashSet<>(currentOrders);

		LinkedList<TabuMove> tabuList = new LinkedList<>();
		int tabuTenure = 7;
		int maxIterations = 30;

		for (int iter = 0; iter < maxIterations; iter++) {
			if (getRemainingTime(stopWatch) <= 0)
				break;

			// Construye la lista de todos los swaps posibles OUT/IN
			List<TabuNeighbor> neighbors = new ArrayList<>();
			for (int out : currentAisles) {
				for (int in = 0; in < m; in++) {
					if (currentAisles.contains(in))
						continue;
					TabuMove move = new TabuMove(out, in);
					if (tabuList.contains(move))
						continue;
					Set<Integer> neighborAisles = new HashSet<>(currentAisles);
					neighborAisles.remove(out);
					neighborAisles.add(in);
					neighbors.add(new TabuNeighbor(neighborAisles, move, out, in));
				}
			}

			// Paraleliza la evaluación de los vecinos
			ForkJoinPool pool = new ForkJoinPool(
					Math.min(neighbors.size(), Runtime.getRuntime().availableProcessors()));
			List<TabuNeighborResult> results;
			try {
				results = pool.submit(() -> neighbors.parallelStream().map(neighbor -> {
					KnapsackResult neighborRes = solveKnapsack(stopWatch, neighbor.aisles, orderSizes,
							getRemainingTime(stopWatch));
					if (neighborRes != null) {
						System.out.println("[Tabu k=" + k +
								" | OUT " + neighbor.out + ", IN " + neighbor.in +
								" | Pasillos: " + neighbor.aisles + " | RATIO: " + neighborRes.ratio);
						return new TabuNeighborResult(neighbor.aisles, neighbor.move, neighborRes.ratio,
								neighborRes.selectedOrders);
					} else {
						System.out.println("[Tabu k=" + k +
								" | OUT " + neighbor.out + ", IN " + neighbor.in +
								" | Pasillos: " + neighbor.aisles + " | NO FACTIBLE");
						return null;
					}
				}).filter(Objects::nonNull).toList()).get();
			} catch (Exception e) {
				e.printStackTrace();
				break;
			} finally {
				pool.shutdown();
			}

			if (results.isEmpty())
				break;

			// Busca el mejor vecino
			TabuNeighborResult bestNeighbor = Collections.max(results, Comparator.comparingDouble(r -> r.ratio));

			currentAisles = bestNeighbor.aisles;
			currentOrders = bestNeighbor.orders;
			tabuList.add(bestNeighbor.move);
			if (tabuList.size() > tabuTenure)
				tabuList.removeFirst();

			if (bestNeighbor.ratio > bestRatio) {
				System.out.println("¡[k=" + k + "] Mejor solución mejorada! Ratio: " + bestNeighbor.ratio +
						" | Pasillos: " + bestNeighbor.aisles +
						" | Pedidos: " + bestNeighbor.orders);
				bestRatio = bestNeighbor.ratio;
				bestAisles = new HashSet<>(bestNeighbor.aisles);
				bestOrders = new HashSet<>(bestNeighbor.orders);
			}
		}

		if (bestAisles.isEmpty() || bestOrders == null || bestOrders.isEmpty())
			return null;
		return new TabuResult(bestRatio, bestAisles, bestOrders);
	}

	// Ayudantes para la paralelización
	private static class TabuNeighbor {
		Set<Integer> aisles;
		TabuMove move;
		int out, in;

		TabuNeighbor(Set<Integer> aisles, TabuMove move, int out, int in) {
			this.aisles = aisles;
			this.move = move;
			this.out = out;
			this.in = in;
		}
	}

	private static class TabuNeighborResult {
		Set<Integer> aisles;
		TabuMove move;
		double ratio;
		Set<Integer> orders;

		TabuNeighborResult(Set<Integer> aisles, TabuMove move, double ratio, Set<Integer> orders) {
			this.aisles = aisles;
			this.move = move;
			this.ratio = ratio;
			this.orders = orders;
		}
	}

	private static class TabuMove {
		int out, in;

		TabuMove(int out, int in) {
			this.out = out;
			this.in = in;
		}

		@Override
		public boolean equals(Object o) {
			if (!(o instanceof TabuMove))
				return false;
			TabuMove t = (TabuMove) o;
			return t.out == out && t.in == in;
		}

		@Override
		public int hashCode() {
			return Objects.hash(out, in);
		}
	}

	private static class KnapsackResult {
		double ratio;
		Set<Integer> selectedOrders;

		KnapsackResult(double ratio, Set<Integer> sel) {
			this.ratio = ratio;
			this.selectedOrders = sel;
		}
	}

	private KnapsackResult solveKnapsack(
			StopWatch stopWatch,
			Set<Integer> aislesSet,
			int[] orderSizes,
			long timeLimitMs) {
		int nItems = this.nItems;
		int nOrders = orders.size();
		int k = aislesSet.size();

		int[] cap = new int[nItems];
		for (int a : aislesSet) {
			for (Map.Entry<Integer, Integer> e : aisles.get(a).entrySet()) {
				cap[e.getKey()] += e.getValue();
			}
		}

		MPSolver solver = MPSolver.createSolver("SCIP");
		if (solver == null)
			return null;

		MPVariable[] x = new MPVariable[nOrders];
		for (int o = 0; o < nOrders; o++) {
			x[o] = solver.makeBoolVar("x_" + o);
		}

		for (int i = 0; i < nItems; i++) {
			MPConstraint c = solver.makeConstraint(0.0, cap[i], "item_" + i);
			for (int o = 0; o < nOrders; o++) {
				int qty = orders.get(o).getOrDefault(i, 0);
				if (qty > 0)
					c.setCoefficient(x[o], qty);
			}
		}

		MPConstraint lb = solver.makeConstraint(waveSizeLB, Double.POSITIVE_INFINITY, "LB");
		MPConstraint ub = solver.makeConstraint(Double.NEGATIVE_INFINITY, waveSizeUB, "UB");
		for (int o = 0; o < nOrders; o++) {
			lb.setCoefficient(x[o], orderSizes[o]);
			ub.setCoefficient(x[o], orderSizes[o]);
		}

		MPObjective obj = solver.objective();
		for (int o = 0; o < nOrders; o++) {
			obj.setCoefficient(x[o], orderSizes[o]);
		}
		obj.setMaximization();

		solver.setTimeLimit(timeLimitMs);

		MPSolver.ResultStatus status = solver.solve();
		if (status != MPSolver.ResultStatus.OPTIMAL && status != MPSolver.ResultStatus.FEASIBLE) {
			return null;
		}

		double totalUnits = obj.value();
		double ratio = totalUnits / k;
		Set<Integer> selOrders = new HashSet<>();
		for (int o = 0; o < nOrders; o++) {
			if (x[o].solutionValue() > 0.5)
				selOrders.add(o);
		}
		return new KnapsackResult(ratio, selOrders);
	}

	protected long getRemainingTime(StopWatch sw) {
		return Math.max(
				TimeUnit.MILLISECONDS.toMillis(MAX_RUNTIME) - sw.getTime(TimeUnit.MILLISECONDS),
				0L);
	}
}

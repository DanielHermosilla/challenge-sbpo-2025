package org.sbpo2025.challenge;

import com.google.ortools.Loader;
import com.google.ortools.linearsolver.*;  // OR-Tools MPSolver, MPVariable, etc.
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
    int maxK = Math.min(m, 10); // only explore up to 5 aisles
    
    double bestRatio = Double.NEGATIVE_INFINITY;
    Set<Integer> bestOrders = Collections.emptySet();
    Set<Integer> bestAisles = Collections.emptySet();

    // Precompute for each order its total size sum_i u_{oi}
    int[] orderSizes = new int[orders.size()];
    for (int o = 0; o < orders.size(); o++) {
      int sum = 0;
      for (int qty : orders.get(o).values()) sum += qty;
      orderSizes[o] = sum;
    }

    // Precompute each aisle's total capacity sum_i u_{ai}
    double[] aisleTotals = new double[m];
    for (int a = 0; a < m; a++) {
      int sum = 0;
      for (int qty : aisles.get(a).values()) sum += qty;
      aisleTotals[a] = sum;
    }

    // Try every k = 1..maxK
    for (int k = 1; k <= maxK; k++) {
      // time check
      long remaining = TimeUnit.MILLISECONDS.toMillis(MAX_RUNTIME) - stopWatch.getTime(TimeUnit.MILLISECONDS);
      if (remaining <= 0) break;

      // 1) pick top-k aisles by total capacity
      Integer[] idx = new Integer[m];
      for (int a = 0; a < m; a++) idx[a] = a;
      Arrays.sort(idx, Comparator.comparingDouble((Integer a) -> aisleTotals[a]).reversed());
      Set<Integer> chosenAisles = new HashSet<>();
      for (int i = 0; i < k; i++) chosenAisles.add(idx[i]);

      // 2) build capacity vector v_i = sum_{a in chosen} u_{ai}
      int[] cap = new int[nItems];
      for (int a : chosenAisles) {
        for (Map.Entry<Integer,Integer> e : aisles.get(a).entrySet()) {
          cap[e.getKey()] += e.getValue();
        }
      }

      // 3) set up the 0–1 knapsack over orders
      MPSolver solver = MPSolver.createSolver("SCIP");
      if (solver == null) continue;  // solver not available

      // decision vars x_o ∈ {0,1}
      MPVariable[] x = new MPVariable[orders.size()];
      for (int o = 0; o < orders.size(); o++) {
        x[o] = solver.makeBoolVar("x_"+o);
      }

      // capacity constraints: for each item i, sum_o x_o * u_{oi} <= cap[i]
      for (int i = 0; i < nItems; i++) {
        MPConstraint c = solver.makeConstraint(0.0, cap[i], "item_"+i);
        for (int o = 0; o < orders.size(); o++) {
          int qty = orders.get(o).getOrDefault(i, 0);
          if (qty>0) c.setCoefficient(x[o], qty);
        }
      }

      // wave‐size constraints: LB <= sum_o x_o * orderSize_o <= UB
      MPConstraint lb = solver.makeConstraint(waveSizeLB, Double.POSITIVE_INFINITY, "LB");
      MPConstraint ub = solver.makeConstraint(Double.NEGATIVE_INFINITY, waveSizeUB, "UB");
      for (int o = 0; o < orders.size(); o++) {
        lb.setCoefficient(x[o], orderSizes[o]);
        ub.setCoefficient(x[o], orderSizes[o]);
      }

      // objective: maximize total units = sum_o x_o * orderSize_o
      MPObjective obj = solver.objective();
      for (int o = 0; o < orders.size(); o++) {
        obj.setCoefficient(x[o], orderSizes[o]);
      }
      obj.setMaximization();

      // set a time limit so we never exceed remaining time
            
		// enforce a time limit (in milliseconds) on this solve
		solver.setTimeLimit(remaining);

      MPSolver.ResultStatus status = solver.solve();

      if (status == MPSolver.ResultStatus.OPTIMAL ||
          status == MPSolver.ResultStatus.FEASIBLE) {
        // total units picked
        double totalUnits = obj.value();
        double ratio = totalUnits / k;

        // if ratio improves, record the solution
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

      // small interruption check
      if (stopWatch.getTime(TimeUnit.MILLISECONDS) > MAX_RUNTIME) break;
    }

    // stopWatch.stop();
    return new ChallengeSolution(bestOrders, bestAisles);
  }

  protected long getRemainingTime(StopWatch sw) {
    return Math.max(
      TimeUnit.MILLISECONDS.toMillis(MAX_RUNTIME) - sw.getTime(TimeUnit.MILLISECONDS),
      0L);
  }
}

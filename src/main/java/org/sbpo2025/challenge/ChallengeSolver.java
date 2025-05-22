// Column Generation using Lagrangian Guidance - SBPO 2025
// Autor: ChatGPT para Daniel Hermosilla
// Este código implementa un método de generación de columnas heurística con guía dual (lagrangiana),
// construyendo subwaves iterativamente que maximizan el costo reducido con base en multiplicadores lambda.

package org.sbpo2025.challenge;

import org.apache.commons.lang3.time.StopWatch;

import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ChallengeSolver {
    private final long MAX_RUNTIME = 600000;
    protected List<Map<Integer, Integer>> orders;
    protected List<Map<Integer, Integer>> aisles;
    protected int nItems;
    protected int waveSizeLB;
    protected int waveSizeUB;

    public ChallengeSolver(List<Map<Integer, Integer>> orders, List<Map<Integer, Integer>> aisles, int nItems, int waveSizeLB, int waveSizeUB) {
        this.orders = orders;
        this.aisles = aisles;
        this.nItems = nItems;
        this.waveSizeLB = waveSizeLB;
        this.waveSizeUB = waveSizeUB;
    }

    public ChallengeSolution solve(StopWatch stopWatch) {
        double[] lambda = new double[nItems];
        Arrays.fill(lambda, 0.0);

        ChallengeSolution bestSolution = null;
        double bestZ = -1;

        for (int iter = 0; iter < 30 && getRemainingTime(stopWatch) > 1; iter++) {
            Subwave column = generateColumn(lambda);
            ChallengeSolution candidate = new ChallengeSolution(column.orders, column.aisles);

            if (!isSolutionFeasible(candidate)) continue;

            double z = computeObjectiveFunction(candidate);
            if (z > bestZ) {
                bestZ = z;
                bestSolution = candidate;
            }

            // actualizar lambda (pseudo-dual update)
            int[] picked = new int[nItems];
            int[] available = new int[nItems];

            for (int o : column.orders) {
                for (Map.Entry<Integer, Integer> e : orders.get(o).entrySet()) {
                    picked[e.getKey()] += e.getValue();
                }
            }
            for (int a : column.aisles) {
                for (Map.Entry<Integer, Integer> e : aisles.get(a).entrySet()) {
                    available[e.getKey()] += e.getValue();
                }
            }

            for (int i = 0; i < nItems; i++) {
                double viol = picked[i] - available[i];
                lambda[i] = Math.max(0.0, lambda[i] + 0.1 * viol); // subgradient update
            }
        }

        return bestSolution;
    }

    protected Subwave generateColumn(double[] lambda) {
        List<Integer> orderIndices = IntStream.range(0, orders.size()).boxed().collect(Collectors.toList());
        orderIndices.sort((o1, o2) -> Double.compare(-dualCost(o1, lambda), -dualCost(o2, lambda)));

        Set<Integer> selectedOrders = new HashSet<>();
        Set<Integer> selectedAisles = new HashSet<>();
        int[] totalPicked = new int[nItems];

        for (int o : orderIndices) {
            Map<Integer, Integer> itemMap = orders.get(o);
            int[] newPicked = Arrays.copyOf(totalPicked, nItems);

            for (Map.Entry<Integer, Integer> e : itemMap.entrySet()) {
                newPicked[e.getKey()] += e.getValue();
            }
            int total = Arrays.stream(newPicked).sum();
            if (total > waveSizeUB) continue;

            selectedOrders.add(o);
            totalPicked = newPicked;

            for (int i : itemMap.keySet()) {
                for (int a = 0; a < aisles.size(); a++) {
                    if (aisles.get(a).containsKey(i)) selectedAisles.add(a);
                }
            }

            if (total >= waveSizeLB) break;
        }
		System.out.println("Trying subwave:");
		System.out.println("  Orders: " + selectedOrders);
		System.out.println("  Aisles: " + selectedAisles);
		System.out.println("  Total items: " + Arrays.stream(totalPicked).sum());
        return new Subwave(selectedOrders, selectedAisles, Arrays.stream(totalPicked).sum());
    }

    protected double dualCost(int order, double[] lambda) {
        return orders.get(order).entrySet().stream()
            .mapToDouble(e -> e.getValue() * (1.0 - lambda[e.getKey()]))
            .sum();
    }

    protected long getRemainingTime(StopWatch stopWatch) {
        return Math.max(TimeUnit.SECONDS.convert(MAX_RUNTIME - stopWatch.getTime(TimeUnit.MILLISECONDS), TimeUnit.MILLISECONDS), 0);
    }

    protected boolean isSolutionFeasible(ChallengeSolution challengeSolution) {
        Set<Integer> selectedOrders = challengeSolution.orders();
        Set<Integer> visitedAisles = challengeSolution.aisles();
        if (selectedOrders == null || visitedAisles == null || selectedOrders.isEmpty() || visitedAisles.isEmpty()) return false;

        int[] totalUnitsPicked = new int[nItems];
        int[] totalUnitsAvailable = new int[nItems];

        for (int order : selectedOrders) {
            for (Map.Entry<Integer, Integer> entry : orders.get(order).entrySet()) {
                totalUnitsPicked[entry.getKey()] += entry.getValue();
            }
        }
        for (int aisle : visitedAisles) {
            for (Map.Entry<Integer, Integer> entry : aisles.get(aisle).entrySet()) {
                totalUnitsAvailable[entry.getKey()] += entry.getValue();
            }
        }

        int totalUnits = Arrays.stream(totalUnitsPicked).sum();
        if (totalUnits < waveSizeLB || totalUnits > waveSizeUB) return false;

        for (int i = 0; i < nItems; i++) {
            if (totalUnitsPicked[i] > totalUnitsAvailable[i]) return false;
        }

        return true;
    }

    protected double computeObjectiveFunction(ChallengeSolution challengeSolution) {
        Set<Integer> selectedOrders = challengeSolution.orders();
        Set<Integer> visitedAisles = challengeSolution.aisles();
        if (selectedOrders == null || visitedAisles == null || selectedOrders.isEmpty() || visitedAisles.isEmpty()) return 0.0;

        int totalUnitsPicked = 0;
        for (int order : selectedOrders) {
            totalUnitsPicked += orders.get(order).values().stream().mapToInt(Integer::intValue).sum();
        }
        return (double) totalUnitsPicked / visitedAisles.size();
    }
}

class Subwave {
    public Set<Integer> orders;
    public Set<Integer> aisles;
    public int totalItems;

    public Subwave(Set<Integer> orders, Set<Integer> aisles, int totalItems) {
        this.orders = orders;
        this.aisles = aisles;
        this.totalItems = totalItems;
    }

    public String toString() {
        return "Orders: " + orders + ", Aisles: " + aisles + ", Items: " + totalItems;
    }
}

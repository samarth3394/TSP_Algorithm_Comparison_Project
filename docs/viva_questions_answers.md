# Viva Questions and Answers

## A. Problem and theory

**1. What is the Travelling Salesperson Problem?**
Given `n` cities and the cost of travelling between each pair, find the shortest closed tour that visits every city exactly
once and returns to the start. Formally, minimise `sum d(pi(k), pi(k+1))` over all permutations `pi` with the start fixed.

**2. Why is the TSP hard?**
The number of tours is `(n-1)!` (or `(n-1)!/2` if symmetric). The decision version ("is there a tour of length <= K?") is
NP-complete (Karp, 1972), and no polynomial-time exact algorithm is known.

**3. What is the difference between exact algorithms, heuristics and approximation algorithms?**
Exact algorithms always return an optimal tour but take exponential time in the worst case. Heuristics (e.g. Nearest Neighbor)
are fast but give no worst-case guarantee in general. Approximation algorithms are fast *and* come with a proven bound on
`cost / OPT` (e.g. 2 or 1.5 under stated assumptions).

**4. What is the metric TSP and why does it matter here?**
Distances are symmetric, non-negative and satisfy the triangle inequality `d(i,k) <= d(i,j) + d(j,k)`. The 2- and
1.5-approximation guarantees are proven only for metric instances. Euclidean distances are metric. Without the
triangle inequality no polynomial-time algorithm can guarantee any constant factor unless P = NP (Sahni & Gonzalez, 1976).

**5. Why do we fix city 0 as the start city?**
A tour is a cycle, so any city can be the start; rotating a tour does not change its length. Fixing the start removes the
`n` equivalent rotations and leaves `(n-1)!` permutations to consider.

**6. What is symmetric versus asymmetric TSP, and which of your algorithms handle asymmetric matrices?**
In symmetric TSP `d(i,j) = d(j,i)`. Brute force, Held-Karp, Branch and Bound and Nearest Neighbor work with asymmetric
matrices; MST preorder and Christofides need symmetry (minimum spanning trees and Euler-circuit arguments are for undirected graphs)
and raise `ValueError` otherwise.

## B. Exact algorithms

**7. How does Brute Force work and when does it stop being practical?**
It evaluates every permutation of cities `1..n-1`. Its cost grows factorially: each extra city multiplies the work by about `n`.
In this project even 11 cities take seconds, so the benchmark stops there.

**8. Explain the Held-Karp recurrence.**
`dp[S][j]` is the cheapest path that starts at city 0, visits exactly the set `S` and ends in `j`. Base: `dp[{j}][j] = d(0,j)`.
Transition: `dp[S][j] = min over k in S\{j} of dp[S\{j}][k] + d(k,j)`. Answer: `min_j dp[all][j] + d(j,0)`.

**9. What are the time and space complexity of Held-Karp and what limits it in practice?**
`O(2^n n^2)` time and `O(2^n n)` space. Memory is the practical limit because the table doubles with every extra city.

**10. Why does dynamic programming work for the TSP?**
Because of optimal sub-structure and overlapping sub-problems: the best path through a set `S` ending at `j` only depends on
the best paths through `S\{j}`, and the same sub-problem appears in many different city orders.

**11. How does Branch and Bound work?**
It builds tours incrementally and computes a lower bound for each partial tour. If the bound is no better than the best complete
tour found so far, the branch is pruned. A good initial upper bound (Nearest Neighbor here) and closest-first ordering make
pruning effective.

**12. What lower bound did you use and why is it valid?**
Cost so far + the cheapest edge leaving the current city + for every unvisited city the cheapest edge it could use to leave
(to another unvisited city or back to the start). Any completion leaves the current city and every unvisited city exactly once, so it
pays at least that much. Because it never overestimates, pruning cannot discard an optimal tour. A unit test checks this against
exhaustive enumeration.

**13. Is Branch and Bound always faster than Brute Force? Than Held-Karp?**
Its worst case is as bad as brute force, but with a good bound it usually expands far fewer nodes. Against Held-Karp the
winner depends on the instance; in this project they are of the same order of magnitude around 16-18 cities, and Branch and Bound
needs far less memory.

**14. How can you be sure your three exact algorithms are correct?**
They agree on the optimal cost on many random, asymmetric, duplicate-city and hand-crafted instances (unit square = 4, collinear points = 14),
every returned tour is validated, and the cost is recomputed independently with a separate function. Equivalent optimal tours
(rotations/reversals) are recognised with `canonical_tour`.

## C. Approximation algorithms

**15. Explain the MST-preorder 2-approximation.**
Build a minimum spanning tree and list cities in depth-first preorder. The MST weighs at most OPT (removing an edge from the optimal tour leaves a spanning tree).
A walk around the doubled tree has length at most `2 * MST`; preorder shortcuts repeated cities, which cannot increase the
length under the triangle inequality. So the tour is at most `2 * OPT`.

**16. Explain Christofides' algorithm step by step.**
MST; collect odd-degree vertices; compute a minimum-weight perfect matching on them; unite the tree and matching into a connected
multigraph with all degrees even; find an Eulerian circuit; shortcut repeated cities.

**17. Why is the matching weight at most OPT/2?**
Shortcut the optimal tour to the odd vertices to get a cycle through them (no longer than OPT by the triangle inequality). An even cycle splits into two perfect matchings; the lighter has weight at most half of the cycle.

**18. Why is the number of odd-degree vertices even?**
Handshake lemma: the sum of all degrees equals twice the number of edges, so the number of odd-degree vertices is even. That is why a perfect matching on them exists in the complete graph.

**19. Why does the combined graph have an Eulerian circuit?**
It is connected (it contains the spanning tree) and every vertex has even degree (tree degree + matching edge for odd vertices). A connected graph with all degrees even has an Euler circuit (Euler's theorem).

**20. What is the guarantee of Nearest Neighbor?**
There is no universal constant factor. On general instances it can be arbitrarily bad; on metric instances its worst-case ratio grows
logarithmically with `n` (Rosenkrantz et al., 1977). Its typical quality on random instances is much better than the worst case, but that is an empirical observation, not a guarantee.

**21. If an approximation algorithm has a 1.5 guarantee, does it always return tours within 1.5 of OPT in practice and is it usually that bad?**
It *always* does on metric instances (that is the guarantee), but typical results are far closer to optimal: the guarantee describes the worst case. Our experiments report the largest observed ratios, which are well below the guaranteed factors.

**22. What happens if the triangle inequality is violated?**
MST preorder and Christofides still return valid tours, but the guarantees no longer apply. The code emits a `RuntimeWarning` in that case.

**23. How do you evaluate quality when the optimum is unknown (50- and 200-city datasets)?**
By dividing the tour length by a valid lower bound on OPT (the 1-tree bound). Since the bound is at most OPT, the ratio is an upper bound on the true ratio `tour / OPT`.

## D. Implementation and experiments

**24. How did you make the comparison fair?**
All algorithms receive exactly the same validated distance matrix for the same dataset; seeds are fixed; timing uses
`time.perf_counter` with the median of repeated runs; memory is measured separately with `tracemalloc` so tracing overhead does not distort timing.

**25. How do you verify that a tour is valid?**
`validate_tour` checks that the tour has `n+1` entries, starts and ends at the same city, and the first `n` entries are a permutation of `0..n-1`. `recompute_tour_cost` recalculates the length with its own loop to cross-check the reported length.

**26. What is the optimality gap?**
`100 * (L - OPT) / OPT` percent, where `L` is the algorithm's tour length. It is only computable when the optimum is known, i.e. for instances solved by an exact algorithm.

**27. Why did you limit the sizes of the exact algorithms?**
Their running time or memory explodes exponentially or factorially. The functions raise `ValueError` above a configurable `max_n`; the benchmark limits (11, 18, 20) were chosen from measured run times.

**28. Why are the memory numbers missing for some larger runs?**
`tracemalloc` slows code down a lot, so memory is measured only for runs that take less than a fixed number of seconds. For slower runs only time is reported.

**29. What would you do to improve tours of the heuristics?**
Apply local search such as 2-opt, Or-opt or Lin-Kernighan to the starting tour, or use metaheuristics (simulated annealing, genetic algorithms); for exact solutions on large instances, branch-and-cut with integer programming (e.g. Concorde).

**30. What are real-world applications of the TSP?**
Delivery and vehicle routing, PCB drilling, warehouse picking, telescope scheduling, DNA sequencing, robotic path planning, and network design.

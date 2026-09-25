# The 19 Sep experiment plan

- [registry.json](registry.json): all 110 experiment records (75 individual + 35 combinations), their prerequisites, and the 117 staged checks (227 run slots in total), plus the 33 completion checks.
- [experiment_matrix.json](experiment_matrix.json): fixed settings for each planned run (seed 42, fold hashes, budgets of 1 epoch for smoke tests, 10 for screens, 50 / 80 for full runs). Entries that depended on earlier results are marked not runnable until those results exist.

What happened to each item is in [docs/3-plan-vs-what-was-done.md](../docs/3-plan-vs-what-was-done.md).

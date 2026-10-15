# tessellate_asteroid/periods -- code context

Project narrative and results: ~/Documents/work/code/tess/asteroid/y3_4/CONTEXT.md. Live copies of these scripts on
ozstar: /fred/oz335/rridden/asteroids/ (scp after editing; jobs never inherit the environment, set it in --wrap).
All UNCOMMITTED in tessellate_asteroid.

Production
- sector_report_pipeline.py: store-mode period pipeline (plot_asteroid = per-object summary row; reliability_score
  reads the bundle's feature list: every feature the model uses must be in its scoring dict). Gates: fold rule,
  search edge, floors. run_store_periods.sh (1000 tasks) -> --gather. Deployed model: round 3 (fold_vn).
- train_round3_ozstar.py (deployed recipe), train_raw_ozstar.py (build_table: rewrites raw_training_table.csv --
  never call it from parallel tasks).

Labels / evaluation
- render_label_sample.py, faint_label_batch.py (faint rounds), bin_range_test.py (recomputed rows for all user labels
  -> asteroids_store/bin_range_test.csv), lcdb_real_test.py (898-object LCDB holdout: the real-truth test).

Synthetic set (asteroids_store/blank_injection)
- blank_injection.py (plan / phot / pipe / check / blind; env RUN, SEED, NUM_OFFSET, EXCLUDE, SNR_RANGE, AMP_RANGE),
  injection_detect.py (detection power, reach labels), relabel_sigma.py, injection_test.py (old host-based method:
  do not reuse).
- synthetic_classifier.py / synthetic_classifier_soft.py / synthetic_feature_test.py / strengthen_c.py
  (VARIANTS=npc|evdep) / c_l_examples.py (MODEL=C_L_ev) / synthetic_examples.py / synthetic_neighbours.py.
- event_dependence.py (max_1day / max_3day_power_drop; candidates / cand stages for the catalogue).
- period_selection.py (candidate-period model), split_half_test.py (flawed: halves lose resolution; ignore).

Rejected experiments (kept for reference): vavilov_periods.py / vavilov_test.py / vavilov_compare_plot.py.

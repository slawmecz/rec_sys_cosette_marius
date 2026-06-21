# REACH summary: Beauty vs Sports_and_Outdoors

- Beauty: MARIUS top-10 reaches 70.7% of the catalog (SASRec++ 81.3%); 2099 items (17.3%) are unreached in EVERY seed (SASRec++ 1115), carrying 5.8% of all train interactions; 1444 items only MARIUS misses.
- Sports_and_Outdoors: MARIUS top-10 reaches 39.2% of the catalog (SASRec++ 73.7%); 9260 items (50.4%) are unreached in EVERY seed (SASRec++ 2597), carrying 22.2% of all train interactions; 6968 items only MARIUS misses.
- Mechanism: the beam never prunes whole L1 prefixes (Beauty: 0 items in never-emitted L1 codes; Sports_and_Outdoors: 0 items in never-emitted L1 codes), but L1:L2 pairs die at decode step 2: never-emitted pairs hold 6.6% of the Beauty catalog, 38.2% of its unreached set; 25.6% of the Sports_and_Outdoors catalog, 50.7% of its unreached set.
- The L1 5x2 table is INVERTED (high-prefix-mass items MORE unreached at fixed own popularity): collapse is sibling competition inside crowded prefixes, not step-1 pruning. Strongest predictor is log own-pop-share within the L1 prefix (r=-0.48; r=-0.62), beating own popularity (r=-0.38; r=-0.58).
- L1:L2 prefix mass works in the hypothesized direction (low pair mass -> unreached: r=-0.25; r=-0.45). Mid-popularity (Q3) unreached rates, low vs high pair mass: Beauty 13.8% vs 10.1%; Sports_and_Outdoors 66.8% vs 42.8%.

See reach_<category>.md for the full tables and reach_unreached_by_decile.png for the decile profile.

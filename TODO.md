# What still needs doing

Working list. `NOTES.md` holds the findings this refers to; nothing here repeats a result,
it only says what to do next and why. Items are roughly in priority order within sections.

## Where things stand

Finding 13's depth table is recorded for seed 0 in `experiments/findings.yaml`. The specs are `experiments/specs/{forward,balanced}-{128,256-128,256-128-64}.yaml`. Seeds 1 and 2 of that table have not been run.

Every other number in `NOTES.md` findings 8 to 15 is still a single seed with one training settling step (`infer_steps=1`) and no result directory.

## Broken or stale

- `scratch_init_diagnostics.py` predates `--regime balanced` and the precision work, so its
  comparisons no longer cover the current options. Either update it or retire it.
- `main.py` takes `--infer-steps` (default 1, the value findings 8 to 15 used). Do not hardcode a second settling budget.

## Balancing: open questions

- **Depth comparison is one seed.** Seed 0 of finding 13 is recorded. Run seeds 1 and 2
  before treating the gaps as stable. Finding 15, the four-layer leapfrog comparison, still
  has no result directory.
- **Why balanced and leapfrog conflict** (finding 15). Those runs trained init edges to copy
  settled states. The current regime trains them with image-only cross-entropy, so a
  re-run is a different experiment. The recorded hypothesis was that balancing moves the
  settled states the init edges were trained to predict.
- **The balancer has not converged at four hidden layers.** `imbalance` was still 1.22 and
  falling after 20 epochs. Check whether `interval` and `step` need to scale with the number
  of edges, and whether a genuinely converged depth-four state is any good.
- **Does balanced help leapfrog at depths one to three?** Seed 0, 8 epochs, one settling step is recorded in the `leapfrog-*` and `balanced-leapfrog-*` findings. It helps at one hidden layer, is flat at two, and hurts at three, where plain leapfrog is the best of the four cells. Final imbalance is about 1, so this is not the unconverged four-layer balancer. Finding 15's four-layer, 20-epoch grid still has no result directory.
- **Label drift under balanced.** Run the 0/5/10/30 eval-step curve (finding 4) with
  `--regime balanced` to see how much of the remaining 2.5-point cost it removes.
- **Evaluation uses weights calibrated with the label clamped.** The measured norms barely
  differ at one hidden layer (finding 12); confirm that holds on deeper graphs.
- **Combine with two-layer jumps.** The `--deep-jumps` path in the scratch script and
  `--regime balanced` are the two best interventions found and have never been run together.

## Language as a byproduct: next

Stage 1 (hub) and stage 2 (consensus graph, one seed under occlusion, balancer on/off
and own-module CE init) are done; `NOTES.md` summarises them. Consensus numbers are the
`consensus-*` entries in `experiments/findings.yaml`. Specs under
`experiments/specs/consensus-occlude-14*.yaml`.

1. **Scale at `s` without the balancer's interior-edge trap.** Stay on shared-init
   nobalance (`consensus-nobalance-ratio-s`): image vote / clue vote is 1.64, and the
   `is k` listen is 6.3 points against the hub's 27 at ratio 1.00. Own-module CE init
   (`consensus-ceinit-does-not-listen`) recovered 89.07% image-only but the clue no
   longer reaches the label, even at ratio 1.09, so do not switch `share_init` off for
   language. The balancer cannot be turned back on as-is (`consensus-balancer-collapse`).
   A scalar on the three vote edges, or a balancer that only sees `z_* -> s`, is the next
   measurement on the nobalance wiring.
2. **Input-dependent variance on the hub `z -> msg`.** `hub-gain-by-type`: `between` and
   `not k or j` make accuracy worse while `is k` gains 27 points, and the ratio at `z` is
   1.00, so the problem is not scale but that a vague clue pulls `z` toward the mean of its
   allowed digits as hard as a precise one pins it. `GaussianEnergy` is one scalar per edge
   today; this needs a module that maps the message to a log-variance.
3. **Negation on the hub.** `hub-negation-not-learned`: clue-only `not k or j` is 0% on
   every seed. Try depth in the clue pathway (`msg -> h_msg -> z` as an energy chain) on the
   hub, where the clue edge still has a say, and keep the per-type clue-only table as the check.
4. **Describing an image regresses to `not even`/`not odd`** (`hub-describes-images`).
   Squared error on one-hot tokens predicts the mean message. Candidates: a per-position
   cross-entropy edge loss for `z -> msg`, or report the ranked list (already in
   `thoughts.txt`) as the description rather than the single nearest statement.
5. **Imagined digits are blurred means** (`hub-imagines-from-clues`): the reference misses
   the prototypes for 1, 4, 5, 8, 9. Committing to the read-off label helps (`between` 81.6 to
   93.2% in-set). Longer settling at evaluation for the imagine path, and a wider `h_img`, are
   the cheap knobs; a style part of `z` is the structural one.
6. **Seeds for the clean hub run** and the noise level, so the degradation comparison is not
   single-seed.

Later stages, once the clue pathway handles vague clues and negation:

- **Equations.** `2 + 4` paired with an image of a 6. Needs depth in the clue pathway. The real
  test is holding out pairs such as `3 + 4` and checking whether 7 is still inferred.
- **Blob counting.** Images of n blobs on their own node `z -> h_count -> count_image`.
  Hypothesis: digits give `z` arbitrary class identities, blobs force an ordered notion of
  quantity. Measure whether distance between class centroids in `z` tracks numeric difference,
  and whether held-out addition improves.

## Carried over from the init work

- Run `forward-matched` against leapfrog at all depths with 1, 10 and 30 training settling
  steps, to settle the parameter confound in finding 16.
- Check the four-layer forward collapse on seeds 1 and 2.
- Red-black refinement: alternately update odd and even layers from both neighbours for k
  rounds, generalising leapfrog into learned self-correcting settling.

## Parked, with reasons

Do not revisit these without new evidence; each is closed by a measurement, not a guess.

- **Diagonal or per-coordinate precision on a data edge** — finding 10. Squared error already
  allocates capacity in proportion to deviation, and whitening would undo it.
- **Per-node gradient normalisation** — finding 11. It cannot rebalance two edges inside one
  node, which is the actual problem.
- **Learned scalar precision in the generative chain** — finding 9. Maximum-likelihood
  variance compares edges with different destinations, so it reports how hard each
  destination is rather than anything useful.

## Longer term

- Edges in both directions inside the energy.
- Per-edge loss schemes beyond squared error and Gaussian.

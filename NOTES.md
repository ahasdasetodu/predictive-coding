# Predictive-coding experiments: notes

Numeric claims live in `experiments/findings.yaml` and are valid only when evidence points at a result directory or a named test. This file keeps the narrative.

## Goal

Experiment with predictive-coding (PC) networks on MNIST and make new architectures and training regimes cheap to try. States are nodes; predictions are directed edges that can point either way.

Long-term ideas: edges in both directions inside the energy, a different loss or weight per edge, and possibly linking PC to "games train adaptation" experiments.

The old MAPMAKER framing is gone and no longer applies. The project rules are `.cursor/rules/clean-code.mdc` and `.cursor/rules/experiment-findings.mdc`.

## Code layout

- `pc_core/graph.py`
  - `States`: dict of named tensors.
  - `GaussianEnergy`: an `Edge.loss` with a learned scalar `log_var`, soft-floored by a softplus at `min_log_var` (a hard clamp would zero the gradient and strand a bottomed-out edge). Energy is `0.5 * ||r||^2 / sigma^2 + 0.5 * numel * log sigma^2`; the normaliser is what stops the variance running to infinity. `PCGraph` registers loss modules so their parameters reach the optimiser, and `log_variances()` reports them.
  - `Edge`: `src` (a node name or a tuple of names, concatenated), `dst`, `module`, `loss`, `weight`. Default loss `squared_error` = half squared error summed over features and batch, so each sample's settling is independent of batch size.
  - `PCGraph`: `nodes`, `energy_edges` (settled and learned on), `init_edges` (only propose starting states); reachable order fills `proposal_edges` then `readout_edges`; `edge_energies`, `energy`, `propagate` (never overwrites clamped nodes).
- `pc_core/models.py`: `mlp_graph(hidden_dims, init="forward" | "leapfrog" | None, output_activation=nn.Identity, learn_precision=False, min_log_var=-4.0, image_weight=1.0)`. Also `hub_graph` and `consensus_graph(share_init=True|False)`.
  - Generative chain label -> h1 -> ... -> image.
  - Edges into hidden nodes: Linear + Tanh. Edges into label or image: Linear + `output_activation` (identity by default). GELU cannot reach normalised background pixels at about -0.42; Tanh can.
  - `forward` steps up one node at a time; `leapfrog` (`_leapfrog_edges`) jumps two nodes up, then fills the skipped node from both neighbours. `skip2` (`_skip2_edges`) learns a bottom-up edge from every node to the node two above it. The hidden node beside the image is the one place a skip cannot land, and it is filled by the generative edge that already predicts it.
- `pc_core/regimes.py`
  - `PCConfig`: infer 20, eval 30, lr 0.1, weight lr 1e-3 by default.
  - `PCRegime`: SGD settling on states with a cosine schedule, Adam weight step on loss / batch size, evaluation with image clamped and argmax of the label. Hooks: `init_states(clamped)` (zeros) and `extra_loss(settled)` (zero).
  - `ForwardInitRegime`: init from `propagate`. `extra_loss` is cross-entropy of a second init pass that sees only the image; the clamped label is the target, not a source. Edges that the label prediction does not read get no gradient from it. Findings 5 to 15 used the older loss, which trained every init edge to copy the settled state. `ByproductRegime` uses the same image-only cross-entropy when the batch has an image, and keeps the copy loss on message-only batches.
  - Logs per sample: `init_energy`, `final_energy`, `E[src->dst]` per edge, `extra_loss`, learning rates. Hooks `observe_init(states)` (inspect the pre-settling states) and `extra_metrics()` (add log values).
  - `BalancedRegime`: forward init plus an `EdgeBalancer`, logging `w[edge]` per edge and `imbalance`.
- `pc_core/balance.py`: `EdgeBalancer` measures the per-sample gradient norm each energy edge imposes on every free node it touches, then steps the weights in log space towards equal norms at each node and renormalises against the loudest edge that touches a clamped node (finding 14). `BalanceConfig(interval=20, step=0.3, momentum=0.5, anchor_data=True)`. Costs one extra backward pass per edge, once every `interval` batches.
- `pc_core/trainer.py`: generic loop.
- `main.py`: `--regime zero|forward|balanced`, `--init forward|leapfrog`, `--hidden ...`, `--epochs`, `--infer-steps` (default 1), `--eval-steps` (default 1), `--seed`, `--learn-precision`, `--min-log-var`, `--image-weight`, and the `PCConfig` learning rates (`--infer-lr`, `--eval-lr`, `--weight-lr`, `--min-lr`). Findings 8 to 15 used one training settling step, which is still the default, and are not directly comparable to findings 5 to 7, which used 10 and 30.
- `pc_core/experiment/`: a YAML spec under `experiments/specs/` runs through the same training path as `main.py` (`uv run python -m pc_core.experiment spec.yaml`) and writes `experiments/results/<run-id>/`. Specs carry a `kind`: `mnist` (default, `mlp_graph`), `clue-bits` (the per-type bits table), or `byproduct` (hub or consensus graph, which also writes `evaluation.yaml`, `gain.png`, `imagine-*.png`, and `thoughts.txt` into the result directory). `graph.topology` defaults to `hub`. Each kind owns its spec and run path; `run_spec` writes the result directory.
- `pc_core/messages.py`: clue grammar, sampler, one-hot encoding, bits, Bayes posterior, and image degradation. `pc_core/reference.py`: the small cross-entropy reference classifier (confusion matrix for `maybe`, ceiling posterior, scorer for imagined images). `pc_core/byproduct.py`: `ByproductRegime`. `pc_core/byproduct_eval.py`: the per-type read-out and plots.
- `tests/test_training.py`: `ForwardInitRegimeTests.test_forward_init_starts_at_lower_energy_than_zeros` fails and predates the precision work. Hidden edges are `nn.Tanh`; the same check passes with GELU, which is the likely cause.
- `scratch_init_diagnostics.py` (throwaway): per-node gap between init and settled state (teacher-forced and at evaluation) plus accuracy, across depths. `--inits forward forward-matched leapfrog`, `--deep-jumps`, `--infer-steps`, `--seed`. `forward-matched` gives forward edges a hidden layer sized to leapfrog's init parameter count.

## Findings

All single seed. Findings 2 to 7 are one epoch; findings 8 to 15 train for several epochs, as each finding states. Treat a few points as noise. Accuracies are before evaluation settling unless stated. Finding 13's depth table, seed 0, has result directories listed in `experiments/findings.yaml`. The other accuracies in this section do not.

1. **Energy scale.** Hundreds per sample is expected: 784 pixels with variance about 1, so predicting zeros scores about 392. The old `F.mse_loss` averaging made state gradients tiny and settling did almost nothing.
2. **Optimiser.** Adam on the weights is what made the old code reach about 70%. SGD gave 30%; Adam gave 74-76% with forward init.
3. **The old state predictor.** At evaluation its top-down input came from a zero label, so it was constant; the predictor was image-only and the label barely moved. Effectively a feedforward classifier.
4. **Long evaluation settling erodes the label.** One hidden layer: 0 / 5 / 10 / 30 steps gave 69 / 73 / 70 / 27%. The free label drifts to explain the image; training never practises a free label. Milder in deeper networks and with more training settling. Unsolved.
5. **Leapfrog vs forward, 10 training settling steps:** [128] 78.5 vs 68.6%, [256,128] 80.4 vs 76.8%, [256,128,64] 71.4 vs 72.5%, [256,256,128,64] 52.6 vs 9.7% (forward collapsed).
6. **With 30 training settling steps** forward catches up and overtakes on deep networks: [256,128,64] 62.1 vs 68.4%, [256,256,128,64] 64.0 vs 68.0%. Trend: the fewer settling steps, the bigger leapfrog's lead. A single step continues the trend (user's run). Likely reason: with little settling, init quality is most of the result, and leapfrog's two-sided fills guess better. With one step, init gaps are partly self-referential, so judge by accuracy.
7. **Diagnostics on deeper networks.**
   - Errors piling up along the chain: not supported. Middle-node gaps stay small; trouble sits at h1 and the label.
   - Jump capacity: strongly supported. Two-layer jumps give 86.2% on [256,128] (best so far) and fix four layers (52.6 -> 79.2%).
   - Train/eval label mismatch (training fill sees the true label): real but small (gap 0.024 vs 0.051).
   - Training settling budget: 10 steps made four-layer forward collapse; 30 steps fixed it. Possibly seed-dependent.
8. **Edge balance explains most of finding 4, and one scalar is worth as much as the best architecture change.** `--image-weight` scales the energy of the edge into the image. Four epochs, `[128]`, forward init, `infer_steps=1`, best validation accuracy:

   | image weight | 1 eval step | 30 eval steps |
   | --- | --- | --- |
   | 1.0 | 84.3 | 79.5 |
   | 0.3 | 85.9 | 81.5 |
   | 0.1 | **86.0** | 82.4 |
   | 0.03 | 85.5 | **83.0** |
   | 0.01 | 84.2 | 82.6 |

   The cost of long evaluation settling falls from 4.9 points to 2.5, so label drift is mostly the 784 pixels outvoting the label during settling rather than anything about the free label itself. 86.0% also matches the best result from two-layer jumps (86.2%), reached here by changing one float. The two are not yet combined.
9. **Learned scalar precision makes accuracy worse, and the reason is instructive.** `--learn-precision` on every energy edge: 82.7% after epoch 0, falling to 74.9% over ten epochs, with the energy at -442 per sample and still dropping. It moves the image edge the opposite way to finding 8. Per dimension the image edge is the *reliable* one, residual about 0.05 per pixel because most pixels are background, so it learns `sigma^2` about 0.27 and gets amplified about fourfold. `label->h1` has a residual of about 1.45 per unit, because one-hot to 128-dim is genuinely one-to-many, so it stays near `sigma^2 = 1` and is suppressed.

   The lesson: maximum-likelihood precision answers "how well does this edge predict its own destination", which is a generative question. Two edges with different destinations are not comparable, so their learned variances mostly report how intrinsically hard each destination is, which has nothing to do with classification. Precision is only well-posed as "which source tells me more" when the edges share a destination. Falling energy is not a warning sign by itself here: the `log sigma^2` term pulls toward variance collapse, and this run claimed `min_log_var = -4` was never reached. The term is not unbounded below: softplus holds `log_var` at `min_log_var`.

   Correction, not a re-run: image `sigma^2` about 0.27 together with an energy of -442 is incompatible with `GaussianEnergy`. At a matched variance the per-sample energy is `0.5 N (1 + log v)`, about -57 for a `[128]` chain with image variance 0.27 and hidden variance 1. The accuracies in this finding were not re-measured.
10. **Squared error already allocates capacity in proportion to deviation, which is why per-coordinate precision is the wrong idea here.** Two separate questions were being conflated: *gradient weighting* (how loudly an error shouts during settling) and *capacity allocation* (how much of a state's budget goes to explaining a coordinate). Precision answers the first; plain squared error already answers the second correctly. Measured on a trained `[128]` graph, using the per-pixel weight norm of the `h1->image` edge as capacity: correlation with within-class std **0.974**, with the Fisher ratio 0.786; the 677 varying pixels get mean capacity 1.431 against 0.435 for the 107 near-constant ones.

   Minimising squared error in the data's own units puts the residual, and so the weights, where the variance is; constant pixels are absorbed by the bias and consume nothing. This is the property that makes least squares find principal components. Diagonal precision is whitening, and whitening destroys it by declaring every pixel equally important to explain. On MNIST that is doubly wrong, since high variance is where the information is: correlation of log within-class variance with log Fisher ratio is **+0.961**, so precision anti-correlates with discriminative value.

   This also explains why a per-edge scalar is the right shape of fix. It is uniform across all 784 pixels, so it leaves allocation alone and only changes image-versus-label balance. Rule of thumb: uniform within an edge, scaled per edge.
11. **Per-node gradient normalisation does not address the balance problem.** `dE/dh1` is a sum of a term from the label and a term from the image, both in `h1`'s space. Dividing the total by its own norm leaves the direction unchanged, so whichever term dominated still dominates. Per-node normalisation equalises how fast different nodes move, not how two edges compete inside one node. Rebalancing within a node requires scaling the terms separately, which is an intervention on the energy (or on each term before summing).
12. **Equal say per edge is the principle, and it predicts the accuracy optimum.** Norm of each incoming gradient term at `h1`, measured at the forward-init state (which is what the single settling step acts on), on `[128]` after four epochs:

    | trained weight | ratio image term / label term | accuracy, 1 eval step |
    | --- | --- | --- |
    | 1.0 | 3.45 | 84.3 |
    | 0.3 | 1.29 | 85.9 |
    | 0.1 | **1.00** | **86.0** |
    | 0.03 | 0.89 | 85.5 |

    Accuracy peaks exactly where the two terms have equal norm. Label-clamped and label-free measurements agree closely, so this is not a train/eval mismatch.

    The measurement is self-referential: the network rescales its weights to roughly match the imposed weight, so a model trained at 1.0 reports an equalising weight of 0.29, not 0.1. A single run cannot give the answer. The map from imposed to measured weight crosses the diagonal between 0.1 and 0.23 (0.3 returns 0.23; 0.1 returns 0.101), which brackets the flat top of the accuracy curve, so a running-norm ratio updated during training should self-calibrate into the optimum.
13. **Auto-calibrated edge weights beat hand tuning, and the weight is not identifiable but the balance is.** `--regime balanced` measures each edge's gradient contribution at the free nodes it touches and steps the weights towards equal norms. Starting from `--image-weight 1.0` and from `0.01`, ten epochs on `[128]`, the `imbalance` metric (worst node's max/min norm ratio) falls to 1.00 in both cases and accuracy lands at 86.5%, above the hand-tuned 86.0%. But the weights end at 0.122 and 0.054 respectively. Same balance, same accuracy, different weight.

    So the weight has no preferred value: the network rescales its own parameters, and only the gradient-norm ratio is invariant. That also means the right hyperparameter was never the weight. Forward init, eight epochs, best validation accuracy:

    | hidden | forward | balanced |
    | --- | --- | --- |
    | [128] | 84.0 | 86.6 |
    | [256, 128] | 86.2 | **87.2** |
    | [256, 128, 64] | 76.5 | 84.2 |

    The seed-0 reproduction of this table is the `finding-13-*` entries in `experiments/findings.yaml`. The 86.5% pair of image-weight starts, and the weights 0.122 and 0.054, still have no result directory. Balanced wins at every depth in that table with forward init, by most at three hidden layers, which fits imbalance compounding along the chain.
14. **The scale must be pinned to the data edges, not the loudest edge.** First version of `EdgeBalancer` renormalised so the largest weight was one. On `[256, 128, 64, 32]` that produced `label->h1` 0.027, `h1->h2` 1.000, `h2->h3` 0.238, `h3->h4` 0.361, `h4->image` 0.007: both edges touching real data starved while an interior edge held the maximum, so the energy became almost entirely internal consistency. The mechanism is the precision trap from finding 9 arriving from the other side. An edge between two free nodes can have its residual settled towards zero, so it measures as quiet, and the balancer then quietens the data edges to match it. One interior node has no purely internal edge and cannot show this; four hidden layers have three of them.

    `BalanceConfig.anchor_data` (default on) normalises on the largest weight among edges with a clamped endpoint instead. Worth 69.7 -> 75.7% at four hidden layers, five epochs, and unchanged at depth one where every edge touches data anyway.
15. **Balanced and leapfrog do not combine.** `[256, 128, 64, 32]`, 20 epochs, best validation accuracy:

    | init | forward | balanced |
    | --- | --- | --- |
    | forward | 59.5 (collapsing to 24.7) | 70.5 |
    | leapfrog | **84.3** | 76.6 |

    Balanced rescues forward init at this depth, including the collapse of finding 5, and it still loses to plain leapfrog by 14 points. Adding balancing on top of leapfrog costs 7.8 points. A likely reading: at `eval_steps=1` accuracy is mostly init quality (finding 6), and the init edges are trained to predict the settled states, so the balancer changes their targets. Leapfrog depends most on init quality and so suffers most when those targets move. Untested: whether balanced plus leapfrog recovers with a larger settling budget. The same question at one to three hidden layers, under the finding-13 budget, is the `leapfrog-*` and `balanced-leapfrog-*` entries in `experiments/findings.yaml`.
16. **Parameter confound.** Both schemes have the same number of init edges, but leapfrog has more init parameters because its fills read concatenated sources, including the 784-wide image: 1.08x at [128], 2.5x at [256,128], 2.1x at [256,128,64], 1.8x at [256,256,128,64]. The one-layer win (about 10 points at 1.08x) is the clean evidence that structure helps; deeper wins are confounded until compared against `forward-matched`. Note `forward-matched` also adds depth per edge, not only width.

## Literature

- Forward init trained on settled states is hybrid predictive coding: Tschantz, Millidge, Seth, Buckley, "Hybrid predictive coding: Inferring, fast and slow", PLOS Computational Biology 2023.
- "Faster Predictive Coding Networks via Better Initialization", arXiv 2601.20895 (2026): compares init schemes, notes forward init is unclear outside layer-by-layer stacks.
- "Error Highways: Scaling Predictive Coding to Very Deep Networks", arXiv 2606.22744 (2026): output-error feedback into the free energy at every inference step, not forward skip connections.
- Nothing published found like leapfrog in a quick search. Fill-from-both-sides resembles ladder VAEs and multigrid solvers.

## Language as a byproduct: hub, stage 1

Classification is the only objective and a clue is only ever an input, so whatever the model learns about words it learns because they help it classify. No next-token objective anywhere. Stage 1 is built and measured; the numbers below are the `clue-*` and `hub-*` entries in `experiments/findings.yaml`.

- **Topology: a hub.** One latent `z` predicts every leaf: `z -> h_img -> image`, `z -> msg`, `z -> label` are the energy edges (`hub_graph` in `pc_core/models.py`, built by `graph_from_edges` from edge strings so other wirings are a spec change). Every leaf proposes `z` through a learned init edge (`image -> h_img -> z`, `msg -> z`, `label -> z`); several proposals are averaged. Every energy edge doubles as a read-off init edge, so a leaf that is not clamped starts at its own prediction and pulls on nothing. `PCGraph(init_order="reachable")` runs whichever init edges the clamped nodes can reach; `mlp_graph` keeps the fixed order the confirmed runs depend on.
- **Shared destination does not set the balance.** Both error terms living at `z` makes them comparable; the balance is still set by residual size and Jacobians. `ByproductRegime` runs the `EdgeBalancer` only on batches with image, clue, and label all clamped and logs the weighted gradient-norm ratio at `z`. It ends at 1.00 on all three seeds (`hub-ratio-z`), so the per-type gaps below are not one modality shouting louder.
- **Messages** are one-hot tokens per position (22 tokens, 5 positions, `pc_core/messages.py`). Grammar: `is k`, `even`, `odd`, `less than k`, `more than k`, `k or j`, `between a and b`, `maybe k`, with `not` on everything except `maybe`: 248 statements, 15 types (`clue-grammar-size`). The sampler draws parameters without looking at the label where it can and prepends `not` when the statement is false, so a clue's likelihood is the same for every digit it allows and "zero the ruled-out digits and renormalise" is the exact Bayes posterior.
- **Bits, measured not guessed** (`clue-bits-uniform`, `clue-bits-table`): on the training prior `is k` 3.32, `k or j` 2.32, `maybe k` 1.29, parity 1.00, `between` 0.99, `less/more than` 0.79, `not between` 0.75, `not k or j` 0.32, `not k` 0.15. `maybe k` names the label 60% of the time, otherwise a digit the clean reference classifier confuses with it (4/9, 3/2/8/7, 7/9, 5/3 are the measured confusions).
- **Degradation.** A random 14x14 patch set to background takes the hub from 86.91% clean to 70.59% image-only (`hub-clean-image-only`, `hub-occlude-14-image-only-s0..s2`; seeds agree within 0.7). Noise sigma 2 gave 76.37 and was not used.
- **The model listens** (`hub-listens-is`, `hub-gain-by-type`): image plus true `is k` 97.9%, shuffled `is k` 43.2%, image only 70.6%. Gains: `is` +27, parity +5 to +9, `or` +6, `maybe` +2.5, `not between` +2, `less/more` about 0, `between` −7, `not k or j` −26. Shuffled `k or j` and `maybe k` cost 22 and 27 points. The gain is not monotone in bits: a fixed-weight clue edge pulls `z` toward the mean of the allowed digits as insistently as `is k` pins it, which is exactly the case for input-dependent variance on `z -> msg`.
- **Clue only tracks Bayes for named digits and parity, not for negation or order** (`hub-clue-only-tracks-bayes`, `hub-negation-not-learned`): `is` 100 vs 100, `or` 51.0 vs 51.5, `maybe` 59.7 vs 60.0, `even` 19.4 vs 20.2; but `not k or j` 0.0 vs 13.9 on every seed, `not k` 7.0 vs 12.4, `more than k` 8.8 vs 21.0. The linear clue edge has learned which digits a clue names; `not` does not flip them.
- **Thinking from an image** (`hub-describes-images`): the read-off clue, decoded to the nearest valid statement, is true 78% of the time under occlusion (87% clean), but it is `not even` or `not odd` for 9946 of 9984 images. Squared error on one-hot tokens regresses to the mean message, whose nearest statement starts with the commonest token. The ranked list behind each image (`thoughts.txt`) is more informative: `maybe k` and `is k` with the right `k` sit third to fifth.
- **Imagining from a clue** (`hub-imagines-from-clues`, `imagine-*.png`): `is k` and label-clamped prototypes are recognisable blurred digits (the reference recognises 0, 2, 3, 6, 7 and misses 1, 4, 5, 8, 9); vague clues give the mean of the allowed digits, as a point latent with a deterministic decoder must. Committing to the read-off label before imagining sharpens `between a and b` from 81.6% to 93.2% in-set.
- **Finding 4 is small here but not gone** (`hub-eval-step-curve`): 70.59 / 70.49 / 70.41 / 70.09% at 1 / 5 / 10 / 30 evaluation steps. The label is a leaf read off `z`, which removes most of the drift, not all of it.
- **Later stages.** Input-dependent variance on `z -> msg` (and `z -> h_img`) so `even` does not pull like `is 6`; depth or a different token loss in the clue pathway so `not` works; equations (`2 + 4` with an image of a 6, holding out pairs such as `3 + 4`); blob-count images on their own leaf with the hypothesis that blobs force an ordered notion of quantity, measurable as whether distance between class centroids in `z` tracks numeric difference.

## Language as a byproduct: consensus, stage 2

The hub is one latent that predicts every leaf. Stage 2 is the design that session underspecified: each modality has a bidirectional tower into its own latent, only clamped latents vote for a shared `s`, and `s` predicts the other two. Training is still the hub's mixed-presence protocol. Builder `consensus_graph` in `pc_core/models.py`; spec `experiments/specs/consensus-occlude-14.yaml`. Numbers below are the `consensus-*` entries in `experiments/findings.yaml`.

- **Topology.** Towers `image <-> h_img <-> z_img`, `msg <-> h_msg <-> z_msg`, `label <-> h_lab <-> z_lab`. Votes `z_* -> s` when that data node is clamped; broadcasts `s -> z_*` always. Default `share_init=True`: bottom-up energy modules are the proposal init. `share_init=False` gives those hops own modules so image-only cross-entropy does not share weights with energy. Top-down energy modules are the read-off. Tanh only into `h_*`, identity into data and latents, so the image-to-label init path has two nonlinearities like the hub.
- **Image-only accuracy is high because settling is not doing it** (`consensus-occlude-14-image-only`, `consensus-balancer-collapse`, `consensus-eval-step-curve`). Best validation 88.19% under the same 14x14 occlusion, above the hub's 70.59%, but by batch 100 of epoch 0 the balancer has set vote and broadcast weights to zero (imbalance 10^12, finding 14's interior-edge trap on a graph with many free-to-free edges). After that, `gap[s]` is zero, energy is zero, and the eval-step curve is flat at 88.19. What remains is the image-only init cross-entropy through the shared encoder modules.
- **The clue never reaches `s`** (`consensus-does-not-listen`, `consensus-clue-only-chance`). Image plus true `is k` 87.85%, shuffled 87.78%, image only 88.19%. Clue-only `is k` is 11.35% (chance); `not k or j` is 11.39% against Bayes 13.83%, so the extra `h_msg` depth did not get to test negation. Imagined `is k` images land in-set 8.91% of the time (`consensus-imagines-from-clues`).
- **Thinking is more diverse and less true** (`consensus-describes-images`). Decoded statements are true 61.00% of the time (hub 78.27%), but they are not almost all `not even`/`not odd`. The ranked list in `thoughts.txt` mixes `not between`, `not is`, `maybe`, and some `is k`.
- **Balancer off restores the decoder and a real clue path** (`consensus-nobalance-*`, spec `experiments/specs/consensus-occlude-14-nobalance.yaml`). Image-only 78.99%, below the collapsed 88.19 and above the hub's 70.59. Image plus true `is k` 85.30%, shuffled 78.24% (`consensus-nobalance-listens-is`): a 6.3-point listen, quieter than the hub's 27 because the image vote at `s` is 1.64 times the clue vote (`consensus-nobalance-ratio-s`). Clue-only `is k` is 100% and `k or j` / `maybe` / parity track Bayes; `not k or j` is still 0% (`consensus-nobalance-clue-only`), so `h_msg` did not fix negation. Think is true 74.00% and volunteers `between` and `is k` rather than parity. Imagined `is k` is in-set 59.52% (69.09% committed); label-clamped prototypes are recognised for every digit except 8 (`consensus-nobalance-imagines`). Energy and `gap[s]` stay nonzero. The next knob on this wiring is scale at `s`, not more settling.
- **Own-module CE init recovers classification and loses the clue** (`consensus-ceinit-*`, spec `experiments/specs/consensus-occlude-14-nobalance-ceinit.yaml`). Image-only 89.07%, matching the collapsed CE encoder, with energy and `gap[s]` still nonzero. Image plus true `is k` 88.21%, shuffled 88.46% (`consensus-ceinit-does-not-listen`); clue-only `is k` is 10.29% chance; imagined `is k` is 9.81% and label prototypes collapse to 0 (`consensus-ceinit-imagines`). The vote ratio at `s` is 1.09 (`consensus-ceinit-ratio-s`), so this is not scale: the CE encoder and the energy decoder no longer share a representation. Shared-init nobalance remains the language wiring.

## Next steps

See `TODO.md`. Keeping the list in one file stops the two drifting apart.

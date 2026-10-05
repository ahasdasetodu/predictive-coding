# Predictive-coding experiments: notes

## Goal

Experiment with predictive-coding (PC) networks on MNIST and make new architectures and training regimes cheap to try. States are nodes; predictions are directed edges that can point either way.

Long-term ideas: edges in both directions inside the energy, a different loss or weight per edge, and possibly linking PC to "games train adaptation" experiments.

The old MAPMAKER framing is gone and no longer applies. The only rule is `.cursor/rules/clean-code.mdc`.

## Code layout

- `pc_core/graph.py`
  - `States`: dict of named tensors.
  - `GaussianEnergy`: an `Edge.loss` with a learned scalar `log_var`, soft-floored by a softplus at `min_log_var` (a hard clamp would zero the gradient and strand a bottomed-out edge). Energy is `0.5 * ||r||^2 / sigma^2 + 0.5 * numel * log sigma^2`; the normaliser is what stops the variance running to infinity. `PCGraph` registers loss modules so their parameters reach the optimiser, and `log_variances()` reports them.
  - `Edge`: `src` (a node name or a tuple of names, concatenated), `dst`, `module`, `loss`, `weight`. Default loss `squared_error` = half squared error summed over features and batch, so each sample's settling is independent of batch size.
  - `PCGraph`: `nodes`, `energy_edges` (settled and learned on), `init_edges` (only propose starting states); `edge_energies`, `energy`, `propagate` (runs init edges in list order, never overwrites clamped nodes).
- `pc_core/models.py`: `mlp_graph(hidden_dims, init="forward" | "leapfrog" | None, output_activation=nn.Identity, learn_precision=False, min_log_var=-4.0, image_weight=1.0)`.
  - Generative chain label -> h1 -> ... -> image.
  - Edges into hidden nodes: Linear + GELU. Edges into label or image: Linear + `output_activation` (identity by default; GELU cannot reach normalised background pixels at about -0.42).
  - `forward` steps up one node at a time; `leapfrog` (`_leapfrog_edges`) jumps two nodes up, then fills the skipped node from both neighbours.
- `pc_core/regimes.py`
  - `PCConfig`: infer 20, eval 30, lr 0.1, weight lr 1e-3 by default.
  - `PCRegime`: SGD settling on states with a cosine schedule, Adam weight step on loss / batch size, evaluation with image clamped and argmax of the label. Hooks: `init_states(clamped)` (zeros) and `extra_loss(settled)` (zero).
  - `ForwardInitRegime`: init from `propagate`; `extra_loss` trains init edges to predict settled states (hybrid predictive coding).
  - Logs per sample: `init_energy`, `final_energy`, `E[src->dst]` per edge, `extra_loss`, learning rates. Hooks `observe_init(states)` (inspect the pre-settling states) and `extra_metrics()` (add log values).
  - `BalancedRegime`: forward init plus an `EdgeBalancer`, logging `w[edge]` per edge and `imbalance`.
- `pc_core/balance.py`: `EdgeBalancer` measures the per-sample gradient norm each energy edge imposes on every free node it touches, then steps the weights in log space towards equal norms at each node and renormalises against the loudest edge that touches a clamped node (finding 14). `BalanceConfig(interval=20, step=0.3, momentum=0.5, anchor_data=True)`. Costs one extra backward pass per edge, once every `interval` batches.
- `pc_core/trainer.py`: generic loop.
- `main.py`: `--regime zero|forward|balanced`, `--init forward|leapfrog`, `--hidden ...`, `--epochs`, `--eval-steps` (default 1), `--seed`, `--learn-precision`, `--min-log-var`, `--image-weight`. Training settling is fixed at `infer_steps=1`, so findings 8 to 15 are all one-step settling and are not directly comparable to findings 5 to 7, which used 10 and 30.
- `tests/test_training.py`: 11 tests. `ForwardInitRegimeTests.test_forward_init_starts_at_lower_energy_than_zeros` fails and predates the precision work; the hidden activation is now `nn.Tanh` where this file described GELU, which is the likely cause.
- `scratch_init_diagnostics.py` (throwaway): per-node gap between init and settled state (teacher-forced and at evaluation) plus accuracy, across depths. `--inits forward forward-matched leapfrog`, `--deep-jumps`, `--infer-steps`, `--seed`. `forward-matched` gives forward edges a hidden layer sized to leapfrog's init parameter count.

## Findings

All single seed, one epoch; treat a few points as noise. Accuracies are before evaluation settling unless stated.

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

   The lesson: maximum-likelihood precision answers "how well does this edge predict its own destination", which is a generative question. Two edges with different destinations are not comparable, so their learned variances mostly report how intrinsically hard each destination is, which has nothing to do with classification. Precision is only well-posed as "which source tells me more" when the edges share a destination. Falling energy is not a warning sign by itself here: the `log sigma^2` term is unbounded below, so this is the variance-collapse direction, and `min_log_var = -4` was never reached.
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

    87.2% is the best result so far. Balanced wins at every depth with forward init, by most at three hidden layers, which fits imbalance compounding along the chain.
14. **The scale must be pinned to the data edges, not the loudest edge.** First version of `EdgeBalancer` renormalised so the largest weight was one. On `[256, 128, 64, 32]` that produced `label->h1` 0.027, `h1->h2` 1.000, `h2->h3` 0.238, `h3->h4` 0.361, `h4->image` 0.007: both edges touching real data starved while an interior edge held the maximum, so the energy became almost entirely internal consistency. The mechanism is the precision trap from finding 9 arriving from the other side. An edge between two free nodes can have its residual settled towards zero, so it measures as quiet, and the balancer then quietens the data edges to match it. One interior node has no purely internal edge and cannot show this; four hidden layers have three of them.

    `BalanceConfig.anchor_data` (default on) normalises on the largest weight among edges with a clamped endpoint instead. Worth 69.7 -> 75.7% at four hidden layers, five epochs, and unchanged at depth one where every edge touches data anyway.
15. **Balanced and leapfrog do not combine.** `[256, 128, 64, 32]`, 20 epochs, best validation accuracy:

    | init | forward | balanced |
    | --- | --- | --- |
    | forward | 59.5 (collapsing to 24.7) | 70.5 |
    | leapfrog | **84.3** | 76.6 |

    Balanced rescues forward init at this depth, including the collapse of finding 5, and it still loses to plain leapfrog by 14 points. Adding balancing on top of leapfrog costs 7.8 points. A likely reading: at `eval_steps=1` accuracy is mostly init quality (finding 6), and the init edges are trained to predict the settled states, so the balancer changes their targets. Leapfrog depends most on init quality and so suffers most when those targets move. Untested: whether balanced plus leapfrog recovers with a larger settling budget.
16. **Parameter confound.** Both schemes have the same number of init edges, but leapfrog has more init parameters because its fills read concatenated sources, including the 784-wide image: 1.08x at [128], 2.5x at [256,128], 2.1x at [256,128,64], 1.8x at [256,256,128,64]. The one-layer win (about 10 points at 1.08x) is the clean evidence that structure helps; deeper wins are confounded until compared against `forward-matched`. Note `forward-matched` also adds depth per edge, not only width.

## Literature

- Forward init trained on settled states is hybrid predictive coding: Tschantz, Millidge, Seth, Buckley, "Hybrid predictive coding: Inferring, fast and slow", PLOS Computational Biology 2023.
- "Faster Predictive Coding Networks via Better Initialization", arXiv 2601.20895 (2026): compares init schemes, notes forward init is unclear outside layer-by-layer stacks.
- "Error Highways: Scaling Predictive Coding to Very Deep Networks", arXiv 2606.22744 (2026): skip connections during settling.
- Nothing published found like leapfrog in a quick search. Fill-from-both-sides resembles ladder VAEs and multigrid solvers.

## Planned: language as a byproduct

Make classification the objective and language an input pathway, so the model learns what a phrase means only because the phrase helps it classify. No next-token objective anywhere.

- **Topology.** Both inputs predict a shared latent: `image -> h_img -> z`, `msg -> h_msg -> z`, `z -> label`. Symmetric, and both error terms live in `z`'s space, so neither modality wins on scale. Settled `z` is a precision-weighted combination of what the pixels say and what the clue says, which is evidence combination.
- **Messages** are one-hot tokens per position, not pretrained embeddings: a pretrained embedding would import outside knowledge of language and could not be decoded back into words. Grammar with an optional `not` prefix on any statement: `is k`, `even`, `odd`, `less than k`, `more than k`, `maybe k`, `k or j`, `between a and b`. Vocabulary about 19 tokens.
- **Clue strength varies enormously**, in bits: `is 6` 3.3, `6 or 9` 2.3, `between 4 and 7` about 2, `even` 1, `more than 6` 1 to 1.5, `not 4` 0.15. Pooling clue types would hide everything, so sample types weighted across that range and always report per type. The headline result is clue information against accuracy gain, with the Bayes-optimal gain as the ceiling.
- **Controls.** No clue (`msg` free), shuffled clues from another image (accuracy must *drop* below no-clue, proving the model listens), clue only (`image` free). A `--degrade noise|occlude` flag makes clues matter by removing information from the image.
- **`maybe k`** is the label about 60% of the time, otherwise a commonly confused digit (4/9, 3/5/8, 1/7). It therefore means "weak evidence", not "this image looks ambiguous", since nothing ties it to the pixels.
- **Precision belongs here, not in the generative chain** (finding 9). `msg->z` and `image->z` share a destination, so their variances are directly comparable and genuinely measure which modality is more informative. An input-dependent variance on `msg->z` should learn that `is 6` is low-variance and `even` or `maybe 4` high-variance, which is exactly the calibrated trust a vague clue needs. Without it, `even` would pull `z` towards the average of five digits as insistently as `is 6` pins it.
- **The latent will collapse to class identity.** Clues derive from the label and nothing asks `z` to remember stroke style, so expect about ten clusters. Accepted for stage 1: text-to-image then generates a class prototype, which is still a useful check. If style is wanted later, split into `z_shared` plus an image-only `z_style`.
- **Generation needs a decoder per pathway.** Inverting `image -> z` by settling alone finds adversarial pixels, since no term requires the image to look like a digit. Instead add reverse init edges `z -> image` and `z -> msg`, trained like the existing init edges to predict settled states, then refine by settling. Decode text two ways: by settling, and by scoring all few-hundred valid messages against `||z - g(m)||^2`, which gives the exact best clue plus a ranked list as a view into what the model thinks about a borderline image. `propagate` will need to run whichever init edges have sources available rather than a fixed order.
- **Expected side effect.** Finding 4 should disappear by construction: here the label is a leaf predicted by `z`, so a free label reads out `z` and pushes back on nothing, instead of being a source dragged around to explain the image.
- **Later stages.** Equations (`2 + 4` paired with an image of a 6) are just a clue type that needs computing, so `h_msg` needs depth; hold out pairs such as `3 + 4` to test systematic generalisation. Then blob-count images on their own node `z -> h_count -> count_image`, with the hypothesis that handwritten digits give `z` arbitrary class identities while blobs force an ordered notion of quantity. Measurable: whether distance between class centroids in `z` tracks numeric difference, and whether held-out addition improves.

## Next steps

See `TODO.md`. Keeping the list in one file stops the two drifting apart.

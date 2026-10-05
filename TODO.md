# What still needs doing

Working list. `NOTES.md` holds the findings this refers to; nothing here repeats a result,
it only says what to do next and why. Items are roughly in priority order within sections.

## Where things stand

Best configuration so far, 87.2% validation on MNIST:

```bash
uv run main.py --regime balanced --hidden 256 128 --epochs 8
```

Every number in `NOTES.md` findings 8 to 15 is a single seed with one training settling
step (`infer_steps=1`, hardcoded in `main.py`) and best-across-epochs validation accuracy.
Treat all of them as provisional until repeated.

## Broken or stale

- `tests/test_training.py::ForwardInitRegimeTests::test_forward_init_starts_at_lower_energy_than_zeros`
  fails and predates the precision and balancing work. `mlp_graph` uses `nn.Tanh` for hidden
  activations where `NOTES.md` described GELU, which is the likely cause. Decide which was
  intended; the fix is one line either way, but the choice affects every result.
- `scratch_init_diagnostics.py` predates `--regime balanced` and the precision work, so its
  comparisons no longer cover the current options. Either update it or retire it.
- `main.py` fixes `infer_steps=1` with no flag. Make it an argument before running anything
  that depends on the settling budget.

## Balancing: open questions

- **Everything is one seed.** Re-run the depth comparison (finding 13) and the deep-chain
  comparison (finding 15) on at least three seeds before trusting either.
- **Why balanced and leapfrog conflict** (finding 15). Two hypotheses worth separating: that
  a larger settling budget recovers the combination, and that the init edges suffer because
  balancing moves the settled states they are trained to predict. The second is testable by
  training the init edges against settled states from a fixed-weight run.
- **The balancer has not converged at four hidden layers.** `imbalance` was still 1.22 and
  falling after 20 epochs. Check whether `interval` and `step` need to scale with the number
  of edges, and whether a genuinely converged depth-four state is any good.
- **Does balanced help leapfrog at all?** Only tested at four hidden layers, where it hurts.
  Try depths one to three.
- **Label drift under balanced.** Run the 0/5/10/30 eval-step curve (finding 4) with
  `--regime balanced` to see how much of the remaining 2.5-point cost it removes.
- **Evaluation uses weights calibrated with the label clamped.** The measured norms barely
  differ at one hidden layer (finding 12); confirm that holds on deeper graphs.
- **Combine with two-layer jumps.** The `--deep-jumps` path in the scratch script and
  `--regime balanced` are the two best interventions found and have never been run together.

## Build: language as a byproduct

The design is written out in `NOTES.md` under "Planned: language as a byproduct". Stage 1
in build order:

1. **`pc_core/messages.py`**: vocabulary and grammar, a sampler that draws a clue type with
   configurable weights and renders it as one-hot tokens per position, the information content
   in bits per clue type, and image degradation (`noise`, `occlude`) so clues have something
   to add.
2. **`PCGraph.propagate`**: run whichever init edges have their sources available instead of
   a fixed list order. Required before anything else, because the clamped set now varies per
   batch and generation runs init edges in both directions.
3. **`multimodal_graph()` in `pc_core/models.py`**: energy edges `image -> h_img -> z`,
   `msg -> h_msg -> z`, `z -> label`, plus reverse init edges `z -> image` and `z -> msg` for
   generation.
4. **A regime with random clamping.** Each batch, sample which modalities are clamped (image
   only, message only, both); a missing modality is simply a free node. This is also the
   fix for finding 4, since training would then practise a free label.
5. **Evaluation that can support the claim.** Accuracy per clue type, never pooled, plus the
   three controls: no clue, shuffled clue from another image (accuracy must drop below
   no-clue), and clue only. The headline output is clue information in bits against accuracy
   gain, with the Bayes-optimal gain as the ceiling line.
6. **Input-dependent variance on `msg -> z` and `image -> z`.** This is the one place
   precision is well-posed, because the two edges share a destination (finding 9). Needed so
   a vague clue like `even` does not pull `z` as insistently as `is 6`.
7. **Message decoding, two ways.** By settling a free `msg` node, and by scoring every valid
   message against `||z - g(m)||^2`. The second gives an exact ranked list, which is both a
   check on the first and the view into what the model thinks about a borderline image.

Note that balancing matters here rather than being a nicety: `msg -> z` and `image -> z`
compete at the same node, and the whole experiment measures how much the message contributes,
so an untuned balance between those two edges would silently set the headline result.

Later stages, once stage 1 reads out sensibly:

- **Equations.** `2 + 4` paired with an image of a 6. Needs depth in `h_msg`. The real test is
  holding out pairs such as `3 + 4` and checking whether 7 is still inferred.
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

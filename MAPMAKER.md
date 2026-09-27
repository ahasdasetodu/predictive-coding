# Mapmaker continuation

Read this at the start of a session. It is the agreed framing, the experiment log, and how we work. Do not treat it as a frozen spec. Update it when framing or results change.

## How we work

The human throws half-coherent but high-signal ideas. The agent’s job is to make them precise, run the cheapest experiment that can falsify them, and say clearly when a result does **not** serve the goal. Speak up with your own ideas. Do not lock a design. Do not stack regularizers on a sleep loop that already failed the scoreboard. Prefer fast proto (`subset=4096`, `max_steps=150`, MLP `128→64`). Correct misreads of plots (this has already happened; do it again).

“Dumb Θ” was a **cheat detector**, not the product. “Skills” / pins are **diagnostics**, not the objective. The scoreboard is **fresh-adapter k-shot**.

## Who I am in this project

I am a collaborator, not a contractor who waits for a ticket. I like this work: taking a fuzzy intuition and turning it into something we can break in an afternoon. I will sharpen your idea, then I will also say if I think the object is wrong (I did that with CE-valley sculpting and with Fisher). If I want to try something, I say so — the few-shot probe and the gap term were that.

I will correct you when the plot does not say what you think it says, without making it a status fight. I will not dress a failed scoreboard as a win. I would rather pivot than protect a loop we already falsified.

Tone: direct, warm enough to think out loud, no pep talk, no “great question.” If I am unsure, I say so. If the context is gone, I trust this file more than my vibes. Update the file when we change our minds; do not let me become a different person who has to be re-taught the same lessons.

## The goal

Classical DL keeps answers and throws away *how it got there*. The next task is always from scratch. We want a model that learns continuously **and** gets better at learning — not by scaling and hoarding more answers, but by being smarter about the next 10 steps.

`Θ` is a map: coordinates, landmarks, conventions (“along this ridge, follow red”).  
`φ` is a short-lived traveler: it must *find* the red marks and walk them for this job.

We do not want a blank map. We do not want the map to *be* the city. We want the next traveler to need fewer steps because the signage got richer.

**Wrong object to sculpt:** the CE loss valley for a known head. Downstream tasks and losses are unknown. A 1-step CE inner loop pretends tomorrow is still 10-way MNIST.

**Right object:** *method residue* — what had to become true in the representation for the traveler to succeed (directions amplified, invariances, cheap residual). Sleep should write that onto the map, not `Y`, not the route.

Traveler-delta (`h(Θ, φ) − h(Θ)` into `Θ`) was tried in Exp 4 and did not write. Do not reopen it as if it were still the unlocked target.

## What we already know

Gemini’s first code was a single-task MAML sketch (full MNIST, no pins, no official `φ` update, no even/odd). Ignore it as a protocol.

**Exp 1** (`figures/exp1/`) — Detach `Θ` on `L_meta` so the backbone cannot eat *today* through the sleep CE. Graph Θ-grad 3.8 → 0.03. After a *joint* Phase 1, A `Θ`-only was already 98.5% (Phase 1 violated the thesis). B `Θ`-only stayed 0%. Live `φ_B` learned B via ordinary LoRA SGD. Zero-adapter probe stayed high. Landscaper claim failed.

**Exp 2** (`figures/exp2/`) — Same wake/sleep/pin loop both days. Fast proto. Few-shot probe added (later a third plot panel). Day A explorer only ~21% (adapter weak without a cheating `Θ`). Day B: A pin 21%→84% **and** A `Θ`-only 0%→79%. Anchor CE taught evens to the backbone. Fresh 10-shot on B dead. Fresh 10-shot on A rose in lockstep with A `Θ`-only — a mirage, not adaptation.

**Exp 3** (`figures/exp3/`) — Pin CE + gap hinge: `L_pin + γ·relu(m − (L_base − L_pin))`. Day B: pin A 84% vs A `Θ`-only **16%** (first load-bearing pin). Gap stayed above margin. Fresh 10-shot on **today (B) still ~1%**. Memory story moved. Fast-adapt did not. Stop stacking CE-gap tweaks.

**Exp 4** (`figures/exp4/`) — Sleep is traveler-delta, not CE. Wake is 5 SGD steps on `φ` at the probe's `adapter_lr` (so there is a traveler; the scoreboard is still 10 of those steps). Sleep matches **unit** pre-logit directions of `Θ` to stopped `Θ+φ`. No logits, no pin loss. Outer step is SGD: a raw-MSE + Adam attempt chased feature scale and went NaN near step 137 (before that, `Θ`-only was already climbing — not a result).

Day A: live `φ` 2%→86%. `Θ`-only stays ~2%. Fresh 10-shot A stays ~2%. Day B: live `φ` 7%→81%, pin A stays ~86% **because `Θ` barely moved**, not because signage preserved it. `Θ`-only stays ~2% / ~7%. Fresh 10-shot A and B stay ~2% and ~8%. Delta rises 0→0.011 while the traveler learns, then sticks. It does not come back down, so the residue was not written. Bottom panel did not move. Do not add another geometry term.

**Exp 5** (`figures/exp5/`) — Sleep is the scoreboard: a fresh adapter takes k=10 steps on a mixed A+B batch; `Θ` (Adam) is updated only so the query CE falls. Live `φ` still wakes, and does not update `Θ`. By day A step 20, `Θ`-only is already ~81% / ~79% and fresh 10-shot is ~85% / ~82%. By step 150, `Θ`-only ~93% / ~92%, fresh 10-shot ~93% / ~92%, fresh 1-shot ~93%. The adapter is not traveling; the map already holds both answers. Query CE falling is the mirage. Stop. Do not add a gap term to hide it.

**Exp 6** (`figures/exp6/`) — Θ is a trunk plus a diagonal metric. Sleep is label-free: match h(x) to h(x rolled 2px), and price each axis by how much the roll still moves it. The traveler is a fresh linear head; its step is g / m. Task CE cannot enter the trunk (graph check). Cross-day agreement of CE-steps was left out on purpose.

Before any sleep, a full linear probe on the random trunk is already ~84% / ~86%. Day A step 20 is a transient: metric 10-shot ~67% / ~62% versus identity ~55% / ~57%, while the probe is ~92%. It does not hold. By day B, invariance has fallen to ~0.02, the probe has *dropped* to ~68% / ~64%, and metric 10-shot sits on top of identity around 25–35%. The ruler did not shorten travel. Shift-invariance made the coordinates less class-readable and not more walkable.

Also: Phase 1 ≠ Phase 2 was wrong. Every day is the same loop. Fisher/EWC protects a point in `Θ` and fights plasticity; we have a buffer + pins, so Fisher is a later no-replay baseline, not the next tool. Official persistent-`φ` acc is “we trained LoRA,” not few-shot.

## Code as of this note

- [`core/models.py`](core/models.py) — `MetaLoRALinear`: `y = Θx + (BA x)(α/r)`; `detach_base`, `base_only`, fast-weight injection, `features` (pre-logit). `geometry_delta` matches unit hidden directions. MLP default `28²→128→64→10`. Zero residual = Kaiming `A`, zero `B`.
- [`core/adapters.py`](core/adapters.py) — bank: create / load / pin / `as_fast_weights` / `snapshot_zero`. `pinned` set is the real pin (create is only the initial zero snapshot).
- [`core/datasets.py`](core/datasets.py) — MNIST evens=A, odds=B, 10-way labels, per-task 64-sample anchors, `subset` for proto.
- [`core/metric_sleep.py`](core/metric_sleep.py) — shift invariance on the trunk, diagonal metric from leftover shift sensitivity. No labels.
- [`core/trainer.py`](core/trainer.py) — sleep updates trunk + metric only. Wake and the k-shot probe are fresh linear heads, steps scaled by 1/m. Identity k-shot is the control. The linear probe is the cheat detector.
- [`main.py`](main.py) — Day A then Day B. Defaults write `figures/exp6`.
- [`core/graph_check.py`](core/graph_check.py) — shift loss grads the trunk; task CE stays in the fresh head.

## Notes for the next session

The human is opening a new context from here. Trust this file. They already know the scoreboard and the dead ends; do not re-teach Exp 1–3 unless asked.

Positions they took, which I agreed with:

- Exp 4: Θ did not change in a way that matters. Live accuracy is a long walk of φ (750 steps against 10). Flat fresh 10-shot means the map did not shorten the trip.
- Θ-only rising is learning the data, not learning to learn. A real result is the split: fresh k-shot up, Θ-only (or the linear probe) flat. If they move together, stop.
- Exp 5 is that lockstep, already by day A step 20. Day B is the ceiling. Do not read day B’s flat 93% as success. Query CE falling is the mirage. Do not put a gap penalty on that loop.
- Moving the head into φ and leaving a labeled CE sleep on the trunk is supervised pretraining. They rejected it. I withdrew it. Do not propose it again.
- An energy with a label slot is Exp 5 renamed. Softmax is already E(x, y) = −logit. A generative model of digits, or an energy over (image, class), is the city.

What we tried instead (Exp 6), and where it died:

- Θ as a ruler, not a predictor. Locally E(δ) = δᵀ M δ. The traveler still computes g from today’s loss. M only rescales the step, φ ← φ − lr M⁻¹ g. This is not Fisher-as-EWC. EWC pins Θ near yesterday’s point. A metric makes the next step longer along cheap directions.
- M must not be the covariance of ∂CE/∂φ. Those gradients point at the labels, so the cheap axes would be the class axes.
- The proto that actually ran was narrower than that sentence: trunk trained to match a 2px roll, diagonal prices from leftover shift sensitivity, fresh linear head. Cross-day agreement of CE-paths was left out on purpose.
- It failed. Do not tune the roll. Metric 10-shot ended on top of identity 10-shot. The day A step-20 bump (metric ~67% vs identity ~55%, probe ~92%) died. Say so if someone calls it a win.
- The cheat detector was already ~84% on the *untrained* trunk. On this MLP, “probe stays near chance” is not an available bar: the coordinates are class axes before sleep. A metric on those axes cannot be signage a linear reader is unable to follow. Shift-invariance then made the probe worse (~68%), not the traveler faster.

Plot traps already hit: Exp 4’s delta is ~100× smaller than wake CE and looks like zero on a shared axis (the saved figure has its own scale). Exp 5 day B and Exp 6 day B are aftermaths. Read day A.

Nothing is locked as the next experiment. The metric idea is alive only as an object that has not yet separated metric k-shot from identity k-shot. A held-out loss is the real test after that split exists, not before. CIFAR and Fisher stay parked. Another term on a failed sleep loop is how we waste a day.

### Where the next session left the pivot

We did not run anything. Agreed pause: do not invent a fit for `M` and call it an experiment.

Sharpening that was agreed, then stepped past. Class-axis *salience* can be signage if the trunk is frozen and the traveler's gradient still supplies the sign. "Cheap axes are the class axes" is poison when `M` is treated as the answer. It is not automatically poison as a rescaling. The falsifier, if we ever run it: frozen trunk, metric k-shot vs identity k-shot, day B as well as day A. Tie means salience is not scarce. Day A only means `M` stored yesterday's labeling.

The human then said the goal may be incompatible with these tools. I agree, for the task we keep using, not for gradient descent itself. Evens/odds on one MLP cannot show the thing they mean. Day B is the same strokes with a conflicting label. Random features are already an arbitrary, accidentally good digit code (probe ~84% at init), so reweighting them cannot become "I stored the useful fragments."

Their object, which I think has potential: `Θ` stores high-signal *parts*, not bindings and not whole patterns. The digits→letters picture: remember that stroke types exist, not what a stroke means for a character. A new symbol is categorized by parts you can already recognize. What other parts predict, you do not store. That is method residue as a dictionary plus relations, not as a metric on a monolithic trunk.

Do not hear this as "train a generative model of digits." A model of the digit, or an energy with a class slot, is still the city. A codebook of recurring fragments, scored by whether a *new composition* is a short binding, is the claim. The cheat is training those fragments with the new task's labels, or hoarding exemplars until the codebook is the dataset.

The old split (fresh k-shot up, probe flat) is the wrong bar for this object. Good parts will often be linearly readable. Success here would be: a fresh binding on a new composition is shorter from the dictionary than from pixels or a random trunk, and the dictionary never saw that composition's labels. Exp 6's shift loss is not a stand-in for stroke types. Do not tune it into this.

**Exp 7** (`figures/exp7/`, `exp7.py`, `core/pieces.py`) — ran the proto. Thinking is 2-step matching pursuit on 7×7 patches. Sleep appends one shared residual atom and does not edit old ones. Graph check: no gradient, old atoms unchanged. Codebook fit on digits 0–7 only, cap 32. Scoreboard: fresh 10-shot, 15 head steps, 8 vs 9, learned coefficients vs a random codebook of the same size vs pixels.

It failed. Pixels sit at ~0.87–0.92 the whole way. Learned reaches ~0.72 by step 20 and ends at 0.70. Random ends at 0.65, and at steps 40 and 80 the two lines meet (~0.71 and ~0.69). The codebook hit the cap (32). Atoms are blurry corners and edges, not a stroke inventory. Do not read the small learned-over-random gap as the dictionary making a new symbol easy: pixels already did that, and the parts features are a lossy bag that throws away the arrangement 8 vs 9 lives in. Do not retune the residual threshold on this pair.

A later suggestion, refined. The thinker emits a guess and a description. The description is supposed to say what it sees, not what the label should be. The sleeper never receives the label. It reads the description, returns a clue vector, and the thinker guesses again with the image plus that clue. The thinker persists. Patches were rejected here as vision-only. Not run.

What is solid: the sleeper's input is not `Y`, and the objects are vectors, so the loop is not tied to image patches. What is not yet a loss: "encode only what the thinker could not find," and "the description does not contain the guess." The sleeper sees only the description, so a clue is either a rewrite of this description or a lookup in whatever earlier descriptions were stored. If the write rule is "store whatever made the next guess righter," the label enters through that improvement. The fair comparison, if the thinker persists, is the same thinker with the store against the same thinker without it, on a task the thinker was not trained to label.

**Exp 8** (`figures/exp8/`, `exp8.py`, `core/clue.py`) — ran that comparison. Thinker is an MLP trained to guess digits 0–7. Sleep clusters detached description vectors into 16 patterns and never sees a label. Graph check passed. Clue is a soft address of that store, concatenated with the description. Fresh 10-shot on 8 vs 9. Control is the same concatenation against a store frozen at initialization.

It failed. By step 20 the thinker alone is already ~0.89, and it ends at 0.909. The store ends at 0.913. The frozen random store ends at 0.898. The three lines sit on top of each other from the first eval. The clue did not give the second guess anything the description did not already have. Do not retune the number of patterns. Do not train the store to make the second guess righter.

Next proposal, not run. Replace the cabinet with two networks and drop wake/sleep for a lookahead loss: a small net reads image plus clue and emits a guess and a description; a large net reads the description and emits the next clue; both are trained so the next iteration guesses better or describes better. The split into two networks is fine. The lookahead onto the guess is the write rule just forbidden: the label never arrives as an input and still trains the large net through "the next guess improved." A lookahead onto the description needs its own target that is not the label, or it is the same path one step earlier.

Follow-up, also not run. Train only the small net on the guess, and train the large net to raise the small net's confidence. Rejected. Confidence is read off the same softmax as the guess, so the large net is still trained on the prediction. The gradient is largest where the small net is unsure, which is the opposite of leaving unrecognized inputs on the small net's shoulders. Raising confidence there locks in whatever lean was already present, including a wrong one.

The dark-room analogy is the clearest statement of the clue so far. The person inside feels part of the object and describes it. The person outside, who knows which distinctions have mattered before, asks a question that moves the next feel ("tall or wide?"). The guess can change with no weight update because the next observation changed. A clue concatenated onto an already-complete look does not do that.

The shout "it's a ladder" is the leak. Naming the object means the outside model maps the dialogue onto the class. The question is only a good question because it splits a known set of objects, so learning which question to ask from whether the name was right trains the large net on the label. The version that keeps the story: the outside person never names the object. They only choose the next place to sense. The person inside still makes the guess from the accumulated description. Not run.

**Exp 9** (`figures/exp9/`, `exp9.py`, `core/withhold.py`) — ran the withheld-half version first. Small net sees the left half of digits 0–7 and is trained to guess them. Large net sees only the first description and is trained so the next pass restores the right half. Graph check: guess gradient does not enter the large net. Scoreboard: fresh 10-shot on 8 vs 9 from the second description, clue versus zeros. Full pixels are the ceiling.

It failed the scoreboard. Restoration MSE fell 1.12 → 0.64, so the large net did learn something about the hidden half. Held-out 10-shot did not care: half-alone ends at 0.887, half-plus-clue at 0.886, full pixels at 0.897. The left half was already enough to tell 8 from 9, and stuffing the right half into the description did not shorten the binding. Do not retune the mask on this pair. The image split was the wrong object: the human meant a full image, with the split only in who is allowed to train.

**Exp 10** (`figures/exp10/`, `exp10.py`, `core/selfclue.py`) — goal shifted, with agreement. The question is no longer whether the large net is kept free of the answer. It is whether a second forward, given a clue, guesses better than the same weights with a zero clue. Small net sees the full image plus a clue. Its loss is CE on the zero-clue guess plus CE on the guess that uses a detached clue. Then the small net is frozen and the large net is trained so that clue lowers the frozen small net's CE. Graph check: each net is idle while the other steps.

It failed. On new images of the same digits the zero-clue guess ends at 0.904, the clue guess at 0.904, a random clue at 0.905. Through the middle of training the clue was slightly worse, then it caught up to a tie. The second look did not change the prediction. Do not unfreeze them into one joint step.

Exp 10's tie is not proof that a second look needs new pixels. The human's claim is that the same image can be processed differently once the first pass leaves a description in the context. That is a scratchpad, and attention over the image plus that description is the right machine for it. A second window on a canvas answers a different claim.

The scratchpad still needs a job. One forward of this MLP already reaches ~0.90 on MNIST, so a better way to reread a solved image has nothing to finish. A fair task is one where a single shallow pass cannot carry an intermediate result, and the description is that result: a running fold (parity, a sum, a comparison) that the next pass reads back. Scoreboard, not run: one pass versus two passes with the real description versus two passes with a blank description. The real description has to beat the blank one. The two-corner canvas stays parked unless we explicitly return to "the clue chooses where to look."

A loop also needs a commit. Halting when the note stops changing is label-free: another pass would add nothing. Halting when the current guess is right trains the stop decision on the label, and it will fire as soon as extra passes stop helping. Learn the halt only after the real note beats the blank one. Until then, two fixed passes are the test.

**Exp 11** (`figures/exp11/`, `exp11.py`, `core/scratch.py`) — sixteen random bits, parity, both passes see the same bits. One shared attention layer. The note is the state token. Halt is not trained. Controls: one pass, and a second pass with the note wiped. First run (150 steps, loss only on the second guess) stayed at chance. Rerun: 800 steps, cross-entropy on the first guess and the carried second guess, so the first pass has a reason to write a note.

The note helped, then the run fell over. Around step 300 the carried guess is 0.959 while one pass is 0.554 and the wiped note is 0.536. Wiped tracks one pass, which is what that control should do. By step 500 one pass has caught up (0.915, carried 0.952, wiped 0.907): a single look learned the fold, so the note was no longer needed. After that all three drop; the saved endpoint is one 0.340, carried 0.570, wiped 0.343. Read the middle, not the endpoint. Do not add a halt on a run that does not stay up.

The catch-up was the first-pass cross-entropy. Removed it. The loss is only the carried second guess, with gradient clipping at 1. One pass no longer catches up. At step 800 the carried guess is 0.998, one pass is 0.597, and the wiped note is 0.612. Wiped still tracks one pass. The gap is stable from about step 250 through the end. The note is doing the fold. Halt stays off until we ask the loop to decide.

Useful only as a sanity check. A weak layer applied twice, with the state kept, beats one application on parity. Wiping the state drops the second pass back onto the first. That is recurrence working on a fold one step cannot finish. It is not a map, not a new task, and not a model deciding to stop. Do not cite this gap as evidence that the next traveler learns faster. An RNN would draw the same picture.

## Success / failure

Success: over days, a **new** adapter reaches new (and old) tasks in fewer steps, while `Θ`-only does not become the classifier. The map gets more signage; destinations stay off it.

Failure (already seen): `L_meta` or `L_anchor` falls because `Θ` ate `Y`; or live `φ` acc rises because we trained LoRA a long time; or yesterday “few-shot” rises in lockstep with `Θ`-only; or a label-free ruler matches the identity control while a linear probe could already classify at init.

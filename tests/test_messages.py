import math
import unittest

import torch

from pc_core.messages import (
    BACKGROUND,
    DIGITS,
    KINDS,
    MAX_LEN,
    MESSAGE_DIM,
    VOCAB,
    Clue,
    ClueSampler,
    Degradation,
    SamplerConfig,
    bayes_posterior,
    clue_bits,
    clues_of_kind,
    decode,
    encode,
    enumerate_clues,
    kind_available,
    kind_information,
    nearest_clue,
    uniform_confusion,
    uniform_prior,
)


def _find(text: str) -> Clue:
    """The enumerated clue with this wording."""
    matches = [clue for clue in enumerate_clues() if clue.text == text]
    if len(matches) != 1:
        raise AssertionError(f"{text!r} matched {len(matches)} clues")
    return matches[0]


class GrammarTests(unittest.TestCase):
    """Lock the size and non-triviality of the clue grammar."""

    def test_grammar_has_248_statements(self) -> None:
        """Every family with its negation, minus always-true and always-false statements, minus not-maybe."""
        clues = enumerate_clues()
        self.assertEqual(len(clues), 248)
        self.assertEqual(len({clue.text for clue in clues}), 248)
        self.assertEqual({clue.kind for clue in clues}, set(KINDS))

    def test_no_statement_is_trivial(self) -> None:
        """Truth-valued statements allow some digits and rule out some."""
        for clue in enumerate_clues():
            if clue.kind == "maybe":
                continue
            self.assertTrue(0 < len(clue.allowed) < 10, clue.text)

    def test_vocabulary_and_length(self) -> None:
        """22 tokens, five positions, 110 message dimensions."""
        self.assertEqual(len(VOCAB), 22)
        self.assertEqual(MAX_LEN, 5)
        self.assertEqual(MESSAGE_DIM, 110)
        self.assertEqual(max(len(clue.tokens) for clue in enumerate_clues()), MAX_LEN)


class SamplerTests(unittest.TestCase):
    """Every sampled clue is true for its label."""

    def test_family_samples_hold_for_their_label(self) -> None:
        """Across many draws and every label, no statement is false."""
        sampler = ClueSampler(SamplerConfig(), uniform_confusion(), seed=3)
        labels = [digit for digit in DIGITS for _ in range(200)]
        for clue, label in zip(sampler.sample(labels), labels):
            self.assertTrue(clue.holds(label), f"{clue.text} for {label}")

    def test_kind_samples_hold_and_match_kind(self) -> None:
        """Drawing by type returns a true statement of that type whenever one exists."""
        sampler = ClueSampler(SamplerConfig(), uniform_confusion(), seed=3)
        for kind in KINDS:
            for label in DIGITS:
                if not kind_available(kind, label):
                    with self.assertRaises(ValueError):
                        sampler.sample_kind(kind, label)
                    continue
                clue = sampler.sample_kind(kind, label)
                self.assertEqual(clue.kind, kind)
                self.assertTrue(clue.holds(label))

    def test_maybe_names_the_label_about_six_in_ten(self) -> None:
        """`maybe k` is the label with the configured probability, otherwise a confused digit."""
        confusion = torch.zeros(10, 10)
        confusion[4, 9] = 1.0
        for digit in DIGITS:
            if digit != 4:
                confusion[digit, (digit + 1) % 10] = 1.0
        sampler = ClueSampler(SamplerConfig(), confusion, seed=1)
        draws = [int(sampler.sample_family("maybe", 4).tokens[1]) for _ in range(4000)]
        share = draws.count(4) / len(draws)
        self.assertAlmostEqual(share, 0.6, delta=0.03)
        self.assertEqual(set(draws), {4, 9})


class EncodingTests(unittest.TestCase):
    """One-hot encoding round-trips through decoding."""

    def test_round_trip_every_clue(self) -> None:
        """Encoding then decoding returns the tokens of all 248 statements."""
        clues = enumerate_clues()
        vectors = encode(clues)
        self.assertEqual(vectors.shape, (248, MESSAGE_DIM))
        self.assertTrue(torch.equal(vectors.reshape(248, MAX_LEN, len(VOCAB)).sum(dim=2), torch.ones(248, MAX_LEN)))
        for clue, vector in zip(clues, vectors):
            self.assertEqual(decode(vector), clue.tokens)

    def test_nearest_clue_recovers_a_noisy_code(self) -> None:
        """A perturbed code decodes to the statement it came from."""
        clues = enumerate_clues()
        torch.manual_seed(0)
        noisy = encode(clues[:20]) + 0.1 * torch.randn(20, MESSAGE_DIM)
        self.assertEqual(nearest_clue(noisy, clues), list(clues[:20]))


class BitsTests(unittest.TestCase):
    """Information under a uniform prior matches the stated values."""

    def test_uniform_prior_bits(self) -> None:
        """is k 3.32, k or j 2.32, between 4 and 7 1.32, more than 6 1.74, even 1.00, not k 0.152."""
        prior = uniform_prior()
        expected = {
            "is 6": 3.32,
            "6 or 9": 2.32,
            "between 4 and 7": 1.32,
            "more than 6": 1.74,
            "even": 1.00,
            "not is 4": 0.152,
        }
        for text, bits in expected.items():
            self.assertAlmostEqual(clue_bits(_find(text), prior), bits, places=2, msg=text)

    def test_kind_bits_under_uniform_prior(self) -> None:
        """Per-type bits equal the set formula averaged over the type's statements."""
        prior = uniform_prior()
        config = SamplerConfig()
        confusion = uniform_confusion()
        self.assertAlmostEqual(kind_information("is", prior, config, confusion).bits, math.log2(10), places=4)
        self.assertAlmostEqual(kind_information("even", prior, config, confusion).bits, 1.0, places=4)
        self.assertAlmostEqual(kind_information("or", prior, config, confusion).bits, math.log2(5), places=4)
        self.assertAlmostEqual(kind_information("not is", prior, config, confusion).bits, math.log2(10 / 9), places=4)
        # Each `less than k` is weighted by how many digits it allows.
        weights = torch.tensor([k / 10 for k in range(1, 10)])
        expected = math.log2(10) - float((weights / weights.sum() * torch.tensor([math.log2(k) for k in range(1, 10)])).sum())
        self.assertAlmostEqual(kind_information("less than", prior, config, confusion).bits, expected, places=4)

    def test_bayes_accuracy_and_maybe(self) -> None:
        """Clue-only Bayes accuracy is the mean of the posterior maximum; maybe lands at 0.6 with uniform confusion."""
        prior = uniform_prior()
        config = SamplerConfig()
        confusion = uniform_confusion()
        self.assertAlmostEqual(kind_information("is", prior, config, confusion).bayes_accuracy, 1.0, places=6)
        self.assertAlmostEqual(kind_information("or", prior, config, confusion).bayes_accuracy, 0.5, places=6)
        maybe = kind_information("maybe", prior, config, confusion)
        self.assertAlmostEqual(maybe.bayes_accuracy, 0.6, places=6)
        self.assertGreater(maybe.bits, 0.0)
        self.assertLess(maybe.bits, math.log2(10))

    def test_bayes_posterior_zeroes_ruled_out_digits(self) -> None:
        """Combining with a truth-valued clue is zero-out-and-renormalise."""
        posterior = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        combined = bayes_posterior(posterior, [_find("less than 2")], SamplerConfig(), uniform_confusion())
        self.assertTrue(torch.allclose(combined[0, :2], torch.tensor([1 / 3, 2 / 3])))
        self.assertEqual(float(combined[0, 2:].sum()), 0.0)

    def test_every_kind_has_clues(self) -> None:
        """Every reported type is populated."""
        for kind in KINDS:
            self.assertTrue(clues_of_kind(kind), kind)


class DegradationTests(unittest.TestCase):
    """Degradation changes pixels and leaves labels to the caller."""

    def test_noise_changes_pixels_at_the_stated_scale(self) -> None:
        """Gaussian noise has the requested standard deviation and returns a new tensor."""
        images = torch.zeros(64, 784)
        generator = torch.Generator().manual_seed(0)
        noisy = Degradation("noise", 0.5).apply(images, generator)
        self.assertIsNot(noisy, images)
        self.assertAlmostEqual(float(noisy.std()), 0.5, delta=0.02)
        self.assertTrue(torch.equal(images, torch.zeros(64, 784)))

    def test_occlusion_sets_one_square_to_background(self) -> None:
        """Exactly size*size pixels per image become the background value."""
        images = torch.ones(8, 784)
        generator = torch.Generator().manual_seed(0)
        occluded = Degradation("occlude", 10).apply(images, generator)
        changed = (occluded != 1.0).reshape(8, 28, 28)
        self.assertTrue(torch.equal(changed.sum(dim=(1, 2)), torch.full((8,), 100)))
        self.assertTrue(torch.allclose(occluded[changed.flatten(1)], torch.full((800,), BACKGROUND)))
        self.assertAlmostEqual(BACKGROUND, -0.4242, places=4)

    def test_none_is_identity(self) -> None:
        """No degradation returns the input unchanged."""
        images = torch.randn(4, 784)
        self.assertIs(Degradation().apply(images), images)


if __name__ == "__main__":
    unittest.main()

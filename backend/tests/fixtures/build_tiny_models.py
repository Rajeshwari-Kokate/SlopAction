"""
Build tiny, locally-constructed models so the neural code paths can be executed
without network access.

READ THIS BEFORE USING THE OUTPUT
---------------------------------
The models produced here have **randomly initialised weights**.  They exist to
prove that the tensor plumbing works - tokenisation, offset mapping, sliding
windows, log-softmax, rank computation, the analytic sampling-discrepancy
moments, batched classification, embedding pooling, the two-model Binoculars
path and every error branch around them.

Their *numeric outputs are meaningless*.  A perplexity from this model says
nothing about any text.  Nothing in the application reads these fixtures unless
the ``ASD_*`` environment variables are pointed at them explicitly, which only
the test-suite does.

Usage::

    python -m tests.fixtures.build_tiny_models            # writes tests/fixtures/models
    pytest -m smoke                                       # runs the neural smoke tests

Everything built here is real code from ``transformers`` / ``tokenizers`` /
``sentence-transformers`` - the tokenizer is genuinely trained (BPE and
WordPiece) on the bundled corpus, and the models are genuine architectures.
Only the weights are random.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"

CORPUS: List[str] = [
    "The committee met on Tuesday to review the quarterly financial results.",
    "Artificial intelligence is transforming the landscape of modern education.",
    "She walked slowly along the river, listening to the water move over stones.",
    "Furthermore, adaptive learning platforms personalise instruction for students.",
    "The function returns a dictionary containing the parsed configuration values.",
    "In conclusion, thoughtful adoption matters more than raw capability.",
    "Perplexity measures how surprised a language model is by a sequence of tokens.",
    "He said that the results were preliminary and would need replication.",
    "Rain fell all afternoon and the streets emptied out before six o'clock.",
    "Researchers report a statistically significant correlation between the variables.",
    "Please let me know if this works for you, and thanks again for the update.",
    "The system combines statistical, linguistic and semantic signals into one estimate.",
] * 24


def _write_corpus(path: Path) -> Path:
    path.write_text("\n".join(CORPUS), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# byte-level BPE tokenizer (GPT-2 family)
# --------------------------------------------------------------------------


def build_bpe_tokenizer(target: Path, vocab_size: int = 900) -> Path:
    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2TokenizerFast

    target.mkdir(parents=True, exist_ok=True)
    corpus_file = _write_corpus(target / "corpus.txt")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=[str(corpus_file)], vocab_size=vocab_size, min_frequency=1,
                    special_tokens=["<|endoftext|>"])
    # Serialise the whole backend tokenizer.  Passing vocab_file/merges_file to
    # GPT2TokenizerFast is silently ignored by current transformers releases and
    # yields an empty vocabulary, so the tokenizer.json route is the only
    # reliable one.
    tokenizer_json = target / "tokenizer.json"
    tokenizer.save(str(tokenizer_json))

    fast = GPT2TokenizerFast(
        tokenizer_file=str(tokenizer_json),
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    assert len(fast) > 200, "tiny BPE tokenizer failed to load its vocabulary"
    fast.save_pretrained(str(target))
    corpus_file.unlink(missing_ok=True)
    return target


# --------------------------------------------------------------------------
# tiny causal language models
# --------------------------------------------------------------------------


def build_causal_lm(target: Path, tokenizer_dir: Path, n_layer: int = 2,
                    n_head: int = 2, n_embd: int = 64, seed: int = 0) -> Path:
    import torch
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=256, n_ctx=256, n_embd=n_embd, n_layer=n_layer, n_head=n_head,
        bos_token_id=eos_id, eos_token_id=eos_id, pad_token_id=eos_id,
    )
    model = GPT2LMHeadModel(config)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target))
    tokenizer.save_pretrained(str(target))
    _stamp(target, "causal_lm")
    return target


# --------------------------------------------------------------------------
# tiny sequence classifier
# --------------------------------------------------------------------------


def build_sequence_classifier(target: Path, tokenizer_dir: Path,
                              labels=("Human", "ChatGPT"), seed: int = 1) -> Path:
    """Two-class by default - deliberately mirroring the public checkpoints, so
    the two-class handling path is what the tests exercise."""
    import torch
    from transformers import AutoTokenizer, GPT2Config, GPT2ForSequenceClassification

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    config = GPT2Config(
        vocab_size=len(tokenizer), n_positions=256, n_ctx=256, n_embd=64,
        n_layer=2, n_head=2, num_labels=len(labels),
        id2label={i: l for i, l in enumerate(labels)},
        label2id={l: i for i, l in enumerate(labels)},
        bos_token_id=eos_id, eos_token_id=eos_id, pad_token_id=eos_id,
    )
    model = GPT2ForSequenceClassification(config)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target))
    tokenizer.save_pretrained(str(target))
    _stamp(target, "sequence_classifier")
    return target


def build_three_class_classifier(target: Path, tokenizer_dir: Path) -> Path:
    """A three-class checkpoint, to exercise the native humanized-class path."""
    return build_sequence_classifier(
        target, tokenizer_dir,
        labels=("human", "ai", "humanized_ai"), seed=2)


# --------------------------------------------------------------------------
# tiny sentence encoder
# --------------------------------------------------------------------------


def build_sentence_encoder(target: Path, seed: int = 3) -> Path:
    import torch
    from tokenizers import BertWordPieceTokenizer
    from transformers import BertConfig, BertModel, BertTokenizerFast

    torch.manual_seed(seed)
    target.mkdir(parents=True, exist_ok=True)
    backbone = target / "backbone"
    backbone.mkdir(parents=True, exist_ok=True)

    corpus_file = _write_corpus(backbone / "corpus.txt")
    wordpiece = BertWordPieceTokenizer(lowercase=True)
    wordpiece.train(files=[str(corpus_file)], vocab_size=800, min_frequency=1,
                    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    wordpiece.save_model(str(backbone))
    tokenizer_json = backbone / "tokenizer.json"
    wordpiece.save(str(tokenizer_json))
    corpus_file.unlink(missing_ok=True)

    tokenizer = BertTokenizerFast(
        vocab_file=str(backbone / "vocab.txt"),
        tokenizer_file=str(tokenizer_json),
        do_lower_case=True)
    assert len(tokenizer) > 100, "tiny WordPiece tokenizer failed to load"
    tokenizer.save_pretrained(str(backbone))

    config = BertConfig(vocab_size=len(tokenizer), hidden_size=32,
                        num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=256)
    BertModel(config).save_pretrained(str(backbone))

    from sentence_transformers import SentenceTransformer, models

    word_embedding = models.Transformer(str(backbone), max_seq_length=128)
    pooling = models.Pooling(word_embedding.get_word_embedding_dimension(),
                             pooling_mode="mean")
    encoder = SentenceTransformer(modules=[word_embedding, pooling])
    encoder.save(str(target))
    _stamp(target, "sentence_encoder")
    return target


# --------------------------------------------------------------------------


def _stamp(target: Path, kind: str) -> None:
    (target / "TINY_FIXTURE.json").write_text(json.dumps({
        "kind": kind,
        "weights": "randomly initialised",
        "purpose": "code-path smoke testing only",
        "warning": ("Outputs from this model are meaningless. Never use it for "
                    "actual detection."),
    }, indent=2), encoding="utf-8")


def build_all(root: Path = MODELS_DIR, clean: bool = False) -> dict:
    if clean and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    tokenizer_dir = build_bpe_tokenizer(root / "tokenizer")
    paths = {
        "tokenizer": tokenizer_dir,
        "causal_lm": build_causal_lm(root / "causal_lm", tokenizer_dir, seed=0),
        # a second, structurally different LM so the Binoculars observer and
        # performer are genuinely two models over one shared vocabulary
        "causal_lm_b": build_causal_lm(root / "causal_lm_b", tokenizer_dir,
                                       n_layer=3, n_embd=64, seed=7),
        "classifier": build_sequence_classifier(root / "classifier", tokenizer_dir),
        "classifier3": build_three_class_classifier(root / "classifier3", tokenizer_dir),
        "encoder": build_sentence_encoder(root / "encoder"),
    }
    return {k: str(v) for k, v in paths.items()}


if __name__ == "__main__":  # pragma: no cover
    built = build_all(clean=True)
    print(json.dumps(built, indent=2))
    print("\nThese fixtures have RANDOM weights. Their outputs are meaningless.")

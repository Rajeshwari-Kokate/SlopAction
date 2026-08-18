"""
Simulated humanizer attacks used to build ``humanized_ai`` training samples.

Why this module exists
----------------------
The third class the detector has to recognise - ``humanized_ai`` - is machine
text that has been pushed through a paraphrase tool or lightly edited by a
person.  Almost nobody publishes labelled corpora of that, so the only way to
get training mass for it is to synthesise it from the ``pure_ai`` pool.  This
module is that synthesiser: a registry of small, deterministic, seeded text
transformations plus named pipelines that chain them.

What these attacks are NOT
--------------------------
They are *simulations*.  Each one approximates a mechanism that real tools use
- lexical substitution, sentence re-segmentation, clause movement, register
shifts, deliberate typos, homoglyph obfuscation - but a commercial paraphraser
or a full LLM rewrite is a strictly stronger attack.  Those rewrite at the
level of meaning and rebuild sentences from scratch; nothing here does.

The practical consequence, stated plainly: **a detector trained and evaluated
only on these attacks will overestimate its own robustness.**  Held-out numbers
against this module tell you the detector survives light editing.  They do not
tell you it survives Quillbot, Undetectable.ai, or "rewrite this so it doesn't
sound like AI" handed to a frontier model.  Treat results here as a floor, not
as evidence of robustness.

To measure the real thing, collect output from an actual humanizer and feed it
in through ``dataset_builder.py --external-humanized PATH``, which ingests
already-humanized text without touching it.  Whenever both are available,
report metrics on the external pool separately - never pooled with these.

Randomness
----------
Every function takes a ``random.Random`` supplied by the caller and never
touches the global ``random`` module, so a seed fully determines the output and
a dataset can be rebuilt byte-for-byte.  This is randomness in *dataset
construction*, which is legitimate and reproducible.  It has nothing to do with
detector outputs, which must be a deterministic function of the input text -
a detector that returns a different verdict on a re-run of the same document is
broken, and no code path in the application is allowed to do that.
"""

from __future__ import annotations

import random
import re
import zlib
from functools import lru_cache
from typing import Callable, Dict, List, Sequence, Tuple

# --------------------------------------------------------------------------
# lexicons
# --------------------------------------------------------------------------

#: Bundled substitution lexicon for ``paraphrase_synonym``.  Deliberately
#: conservative: entries are near-synonyms that survive most syntactic frames,
#: because a swap that breaks grammar creates a spurious "humanized" cue the
#: detector could latch onto instead of the real one.
SYNONYMS: Dict[str, List[str]] = {
    "important": ["significant", "notable", "key"],
    "significant": ["substantial", "considerable", "marked"],
    "crucial": ["vital", "essential", "critical"],
    "essential": ["necessary", "fundamental", "vital"],
    "difficult": ["hard", "challenging", "tricky"],
    "easy": ["simple", "straightforward", "painless"],
    "large": ["big", "sizeable", "substantial"],
    "small": ["little", "modest", "minor"],
    "quick": ["fast", "rapid", "swift"],
    "slow": ["sluggish", "gradual", "unhurried"],
    "good": ["decent", "solid", "sound"],
    "bad": ["poor", "weak", "flawed"],
    "many": ["numerous", "plenty of", "a lot of"],
    "few": ["not many", "a handful of", "scarce"],
    "often": ["frequently", "regularly", "commonly"],
    "rarely": ["seldom", "infrequently", "hardly ever"],
    "always": ["consistently", "invariably", "without fail"],
    "never": ["at no point", "not once", "never once"],
    "show": ["demonstrate", "reveal", "indicate"],
    "shows": ["demonstrates", "reveals", "indicates"],
    "showed": ["demonstrated", "revealed", "indicated"],
    "prove": ["establish", "confirm", "verify"],
    "proves": ["establishes", "confirms", "verifies"],
    "help": ["assist", "aid", "support"],
    "helps": ["assists", "aids", "supports"],
    "make": ["create", "produce", "build"],
    "makes": ["creates", "produces", "builds"],
    "made": ["created", "produced", "built"],
    "use": ["employ", "apply", "utilise"],
    "uses": ["employs", "applies", "utilises"],
    "used": ["employed", "applied", "drew on"],
    "using": ["employing", "applying", "drawing on"],
    "need": ["require", "call for", "demand"],
    "needs": ["requires", "calls for", "demands"],
    "give": ["provide", "offer", "supply"],
    "gives": ["provides", "offers", "supplies"],
    "get": ["obtain", "acquire", "secure"],
    "gets": ["obtains", "acquires", "secures"],
    "find": ["discover", "identify", "locate"],
    "finds": ["discovers", "identifies", "locates"],
    "found": ["discovered", "identified", "located"],
    "think": ["believe", "reckon", "consider"],
    "thinks": ["believes", "reckons", "considers"],
    "know": ["understand", "grasp", "recognise"],
    "knows": ["understands", "grasps", "recognises"],
    "say": ["state", "note", "remark"],
    "says": ["states", "notes", "remarks"],
    "said": ["stated", "noted", "remarked"],
    "tell": ["inform", "notify", "let know"],
    "start": ["begin", "commence", "kick off"],
    "starts": ["begins", "commences", "kicks off"],
    "started": ["began", "commenced", "kicked off"],
    "end": ["finish", "conclude", "close"],
    "ends": ["finishes", "concludes", "closes"],
    "increase": ["rise", "grow", "climb"],
    "increases": ["rises", "grows", "climbs"],
    "increased": ["rose", "grew", "climbed"],
    "decrease": ["fall", "drop", "decline"],
    "decreases": ["falls", "drops", "declines"],
    "decreased": ["fell", "dropped", "declined"],
    "improve": ["enhance", "strengthen", "refine"],
    "improves": ["enhances", "strengthens", "refines"],
    "improved": ["enhanced", "strengthened", "refined"],
    "reduce": ["cut", "lower", "curb"],
    "reduces": ["cuts", "lowers", "curbs"],
    "change": ["shift", "alter", "modify"],
    "changes": ["shifts", "alters", "modifies"],
    "changed": ["shifted", "altered", "modified"],
    "affect": ["influence", "shape", "impact"],
    "affects": ["influences", "shapes", "impacts"],
    "cause": ["trigger", "produce", "bring about"],
    "causes": ["triggers", "produces", "brings about"],
    "allow": ["permit", "enable", "let"],
    "allows": ["permits", "enables", "lets"],
    "require": ["need", "call for", "demand"],
    "requires": ["needs", "calls for", "demands"],
    "include": ["cover", "contain", "encompass"],
    "includes": ["covers", "contains", "encompasses"],
    "including": ["such as", "covering", "among them"],
    "provide": ["offer", "supply", "deliver"],
    "provides": ["offers", "supplies", "delivers"],
    "ensure": ["make sure", "guarantee", "see to it"],
    "ensures": ["makes sure", "guarantees", "sees to it"],
    "consider": ["weigh up", "look at", "take into account"],
    "considers": ["weighs up", "looks at", "takes into account"],
    "explain": ["clarify", "set out", "spell out"],
    "explains": ["clarifies", "sets out", "spells out"],
    "describe": ["outline", "characterise", "set out"],
    "describes": ["outlines", "characterises", "sets out"],
    "suggest": ["imply", "point to", "hint at"],
    "suggests": ["implies", "points to", "hints at"],
    "argue": ["contend", "maintain", "claim"],
    "argues": ["contends", "maintains", "claims"],
    "focus": ["concentrate", "centre", "zero in"],
    "focuses": ["concentrates", "centres", "zeroes in"],
    "develop": ["build", "grow", "cultivate"],
    "develops": ["builds", "grows", "cultivates"],
    "create": ["build", "produce", "generate"],
    "creates": ["builds", "produces", "generates"],
    "achieve": ["reach", "attain", "accomplish"],
    "achieves": ["reaches", "attains", "accomplishes"],
    "maintain": ["keep up", "sustain", "preserve"],
    "support": ["back", "underpin", "bolster"],
    "supports": ["backs", "underpins", "bolsters"],
    "address": ["tackle", "deal with", "handle"],
    "addresses": ["tackles", "deals with", "handles"],
    "highlight": ["underline", "stress", "flag"],
    "highlights": ["underlines", "stresses", "flags"],
    "emphasise": ["stress", "underline", "foreground"],
    "reflect": ["mirror", "echo", "capture"],
    "reflects": ["mirrors", "echoes", "captures"],
    "reveal": ["expose", "uncover", "lay bare"],
    "remain": ["stay", "continue to be", "persist"],
    "remains": ["stays", "continues to be", "persists"],
    "appear": ["seem", "look", "come across as"],
    "appears": ["seems", "looks", "comes across as"],
    "approach": ["method", "strategy", "way"],
    "method": ["approach", "technique", "procedure"],
    "result": ["outcome", "upshot", "consequence"],
    "results": ["outcomes", "findings", "consequences"],
    "impact": ["effect", "influence", "consequence"],
    "effect": ["impact", "consequence", "result"],
    "benefit": ["advantage", "upside", "gain"],
    "benefits": ["advantages", "upsides", "gains"],
    "problem": ["issue", "difficulty", "snag"],
    "problems": ["issues", "difficulties", "snags"],
    "issue": ["problem", "concern", "matter"],
    "challenge": ["difficulty", "hurdle", "obstacle"],
    "challenges": ["difficulties", "hurdles", "obstacles"],
    "solution": ["answer", "fix", "remedy"],
    "goal": ["aim", "objective", "target"],
    "goals": ["aims", "objectives", "targets"],
    "purpose": ["aim", "intent", "point"],
    "reason": ["rationale", "motive", "basis"],
    "reasons": ["rationales", "motives", "grounds"],
    "example": ["instance", "case", "illustration"],
    "examples": ["instances", "cases", "illustrations"],
    "factor": ["element", "driver", "consideration"],
    "factors": ["elements", "drivers", "considerations"],
    "aspect": ["facet", "dimension", "side"],
    "aspects": ["facets", "dimensions", "sides"],
    "feature": ["trait", "characteristic", "property"],
    "features": ["traits", "characteristics", "properties"],
    "process": ["procedure", "workflow", "routine"],
    "system": ["setup", "framework", "arrangement"],
    "systems": ["setups", "frameworks", "arrangements"],
    "tool": ["instrument", "utility", "device"],
    "tools": ["instruments", "utilities", "devices"],
    "area": ["field", "domain", "space"],
    "areas": ["fields", "domains", "spaces"],
    "field": ["area", "discipline", "domain"],
    "research": ["study", "investigation", "work"],
    "study": ["investigation", "analysis", "piece of work"],
    "studies": ["investigations", "analyses", "papers"],
    "evidence": ["proof", "data", "grounds"],
    "data": ["figures", "numbers", "records"],
    "information": ["detail", "material", "content"],
    "knowledge": ["understanding", "expertise", "know-how"],
    "understanding": ["grasp", "comprehension", "handle"],
    "experience": ["background", "exposure", "track record"],
    "practice": ["habit", "routine", "custom"],
    "quality": ["standard", "calibre", "grade"],
    "amount": ["quantity", "volume", "level"],
    "number": ["count", "quantity", "tally"],
    "level": ["degree", "extent", "scale"],
    "growth": ["expansion", "increase", "rise"],
    "value": ["worth", "merit", "usefulness"],
    "cost": ["price", "expense", "outlay"],
    "risk": ["danger", "exposure", "hazard"],
    "risks": ["dangers", "exposures", "hazards"],
    "opportunity": ["chance", "opening", "window"],
    "opportunities": ["chances", "openings", "prospects"],
    "role": ["part", "function", "job"],
    "way": ["manner", "route", "means"],
    "ways": ["manners", "routes", "means"],
    "time": ["period", "stretch", "spell"],
    "people": ["individuals", "folk", "members of the public"],
    "person": ["individual", "someone"],
    "world": ["globe", "planet", "wider world"],
    "modern": ["contemporary", "present-day", "current"],
    "current": ["present", "existing", "ongoing"],
    "recent": ["latest", "fresh", "new"],
    "common": ["widespread", "usual", "routine"],
    "various": ["assorted", "differing", "several"],
    "several": ["a number of", "various", "some"],
    "different": ["distinct", "dissimilar", "varied"],
    "similar": ["comparable", "alike", "much the same"],
    "clear": ["obvious", "plain", "evident"],
    "complex": ["complicated", "involved", "intricate"],
    "simple": ["basic", "plain", "uncomplicated"],
    "effective": ["successful", "workable", "productive"],
    "efficient": ["economical", "streamlined", "lean"],
    "powerful": ["strong", "potent", "formidable"],
    "useful": ["handy", "helpful", "practical"],
    "popular": ["well-liked", "widely used", "in demand"],
    "successful": ["effective", "fruitful", "thriving"],
    "necessary": ["needed", "required", "called for"],
    "possible": ["feasible", "achievable", "workable"],
    "likely": ["probable", "expected", "on the cards"],
    "certain": ["sure", "definite", "settled"],
    "specific": ["particular", "precise", "narrow"],
    "general": ["broad", "overall", "wide"],
    "overall": ["on balance", "broadly", "taken together"],
    "particularly": ["especially", "notably", "above all"],
    "especially": ["particularly", "notably", "chiefly"],
    "generally": ["broadly", "as a rule", "on the whole"],
    "typically": ["usually", "as a rule", "more often than not"],
    "usually": ["typically", "normally", "as a rule"],
    "clearly": ["plainly", "obviously", "evidently"],
    "quickly": ["rapidly", "swiftly", "in short order"],
    "carefully": ["thoroughly", "attentively", "with care"],
    "directly": ["straight", "immediately", "head-on"],
    "simply": ["merely", "just", "plainly"],
    "widely": ["broadly", "extensively", "across the board"],
    "highly": ["extremely", "very", "notably"],
    "extremely": ["highly", "exceptionally", "remarkably"],
    "very": ["deeply", "genuinely", "particularly"],
    "quite": ["fairly", "rather", "reasonably"],
    "almost": ["nearly", "close to", "just about"],
    "about": ["around", "roughly", "approximately"],
    "because": ["since", "as", "given that"],
    "although": ["though", "even if", "while"],
    "however": ["though", "even so", "that said"],
    "therefore": ["so", "as a result", "which means"],
    "additionally": ["also", "on top of that", "as well"],
}

#: Expanded form -> contracted form.  Used in both directions; the reverse map
#: is derived once at import so the two attacks cannot drift apart.
CONTRACTIONS: Dict[str, str] = {
    "do not": "don't", "does not": "doesn't", "did not": "didn't",
    "is not": "isn't", "are not": "aren't", "was not": "wasn't",
    "were not": "weren't", "has not": "hasn't", "have not": "haven't",
    "had not": "hadn't", "will not": "won't", "would not": "wouldn't",
    "should not": "shouldn't", "could not": "couldn't", "cannot": "can't",
    "can not": "can't", "must not": "mustn't", "need not": "needn't",
    "it is": "it's", "it has": "it's", "that is": "that's",
    "there is": "there's", "there has": "there's", "here is": "here's",
    "what is": "what's", "who is": "who's", "where is": "where's",
    "how is": "how's", "let us": "let's", "they are": "they're",
    "we are": "we're", "you are": "you're", "i am": "I'm",
    "i have": "I've", "i will": "I'll", "i would": "I'd",
    "they have": "they've", "we have": "we've", "you have": "you've",
    "they will": "they'll", "we will": "we'll", "you will": "you'll",
    "he will": "he'll", "she will": "she'll", "it will": "it'll",
    "they would": "they'd", "we would": "we'd", "you would": "you'd",
    "he would": "he'd", "she would": "she'd",
    "he is": "he's", "she is": "she's", "who has": "who's",
    "would have": "would've", "could have": "could've",
    "should have": "should've", "might have": "might've",
    "that will": "that'll", "there will": "there'll",
}

#: Reverse map, contracted -> expanded.  ``it's`` is genuinely ambiguous
#: (is/has); the first key wins, which is the far more common reading.
EXPANSIONS: Dict[str, str] = {}
for _long, _short in CONTRACTIONS.items():
    EXPANSIONS.setdefault(_short.lower(), _long)

#: (British, American) pairs.  Only one-to-one swaps that do not change meaning.
SPELLING_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("colour", "color"), ("colours", "colors"), ("coloured", "colored"),
    ("behaviour", "behavior"), ("behaviours", "behaviors"),
    ("favour", "favor"), ("favours", "favors"), ("favourite", "favorite"),
    ("labour", "labor"), ("honour", "honor"), ("humour", "humor"),
    ("neighbour", "neighbor"), ("neighbours", "neighbors"),
    ("rumour", "rumor"), ("endeavour", "endeavor"), ("flavour", "flavor"),
    ("organise", "organize"), ("organised", "organized"),
    ("organisation", "organization"), ("organisations", "organizations"),
    ("recognise", "recognize"), ("recognised", "recognized"),
    ("realise", "realize"), ("realised", "realized"),
    ("emphasise", "emphasize"), ("emphasised", "emphasized"),
    ("analyse", "analyze"), ("analysed", "analyzed"),
    ("summarise", "summarize"), ("summarised", "summarized"),
    ("prioritise", "prioritize"), ("prioritised", "prioritized"),
    ("utilise", "utilize"), ("utilised", "utilized"),
    ("minimise", "minimize"), ("maximise", "maximize"),
    ("optimise", "optimize"), ("optimised", "optimized"),
    ("specialise", "specialize"), ("specialised", "specialized"),
    ("centre", "center"), ("centres", "centers"), ("centred", "centered"),
    ("metre", "meter"), ("metres", "meters"), ("litre", "liter"),
    ("theatre", "theater"), ("fibre", "fiber"),
    ("defence", "defense"), ("offence", "offense"), ("licence", "license"),
    ("practise", "practice"), ("pretence", "pretense"),
    ("travelled", "traveled"), ("travelling", "traveling"),
    ("modelled", "modeled"), ("modelling", "modeling"),
    ("labelled", "labeled"), ("labelling", "labeling"),
    ("cancelled", "canceled"), ("cancelling", "canceling"),
    ("programme", "program"), ("programmes", "programs"),
    ("catalogue", "catalog"), ("dialogue", "dialog"),
    ("analogue", "analog"), ("grey", "gray"),
    ("judgement", "judgment"), ("acknowledgement", "acknowledgment"),
    ("ageing", "aging"), ("cheque", "check"), ("enrol", "enroll"),
    ("fulfil", "fulfill"), ("instalment", "installment"),
    ("skilful", "skillful"), ("wilful", "willful"),
    ("towards", "toward"), ("amongst", "among"), ("whilst", "while"),
    ("learnt", "learned"), ("spelt", "spelled"), ("burnt", "burned"),
)

#: Sentence-initial connectives that giveaway-heavy model prose leans on.
#: ``discourse_strip`` removes them, which is exactly what a human editor does
#: on a second pass and what several paraphrase tools do automatically.
DISCOURSE_MARKERS: Tuple[str, ...] = (
    "Furthermore", "Moreover", "Additionally", "In addition", "However",
    "Therefore", "Thus", "Consequently", "Nevertheless", "Nonetheless",
    "Indeed", "Overall", "In conclusion", "Ultimately", "Importantly",
    "Notably", "Similarly", "In contrast", "On the other hand",
    "As a result", "That said", "Firstly", "Secondly", "Thirdly",
    "Finally", "In summary", "To summarise", "Crucially", "Subsequently",
    "In particular", "For instance", "For example", "In essence",
)

#: Formal phrase -> informal counterpart.  ``tone_formal`` runs it backwards.
TONE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("in order to", "to"),
    ("a significant number of", "lots of"),
    ("a large number of", "loads of"),
    ("it is important to note that", "worth saying that"),
    ("it should be noted that", "note that"),
    ("it is worth noting that", "worth mentioning that"),
    ("due to the fact that", "because"),
    ("in the event that", "if"),
    ("prior to", "before"),
    ("subsequent to", "after"),
    ("with regard to", "about"),
    ("in relation to", "about"),
    ("in terms of", "when it comes to"),
    ("a variety of", "all sorts of"),
    ("numerous", "loads of"),
    ("utilise", "use"),
    ("commence", "start"),
    ("terminate", "end"),
    ("demonstrate", "show"),
    ("obtain", "get"),
    ("purchase", "buy"),
    ("assist", "help"),
    ("attempt", "try"),
    ("require", "need"),
    ("sufficient", "enough"),
    ("approximately", "roughly"),
    ("additional", "extra"),
    ("individuals", "people"),
    ("furthermore", "plus"),
    ("consequently", "so"),
    ("nevertheless", "even so"),
    ("is capable of", "can"),
    ("has the ability to", "can"),
    ("at this point in time", "right now"),
    ("in the near future", "soon"),
)

#: Informal sides that must never be run backwards by ``tone_formal``.  They
#: are function words or duplicated targets, so promoting them produces
#: nonsense ("to" -> "in order to" inside "important to note").
_UNSAFE_FORMALISATIONS = frozenset({
    "to", "if", "about", "before", "after", "because", "so", "can",
    "plus", "even so", "note that", "when it comes to",
})

#: Fillers ``shorten`` deletes and ``expand`` inserts.
FILLER_PHRASES: Tuple[str, ...] = (
    "It is important to note that", "It is worth noting that",
    "As previously mentioned", "In today's world",
    "At the end of the day", "When all is said and done",
    "For all intents and purposes", "In the grand scheme of things",
    "Broadly speaking", "To put it simply",
)

#: Redundant intensifiers ``shorten`` strips.
FILLER_WORDS: Tuple[str, ...] = (
    "very", "really", "quite", "rather", "actually", "basically",
    "essentially", "certainly", "definitely", "simply", "truly",
    "significantly", "substantially", "extremely", "incredibly",
)

#: Epistemic softeners.  Model prose hedges with adverbs; people hedge with
#: first-person asides, which is why these are phrased that way.
HEDGES: Tuple[str, ...] = (
    "I think", "in my view", "as far as I can tell", "arguably",
    "to be fair", "more or less", "if I'm honest", "from what I've seen",
    "at least in my experience", "or so it seems",
)

#: QWERTY neighbours, for realistic fat-finger typos.
_KEY_NEIGHBOURS: Dict[str, str] = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh",
    "u": "yij", "i": "uok", "o": "ipl", "p": "ol", "a": "qsz", "s": "awdx",
    "d": "sefc", "f": "drgv", "g": "fthb", "h": "gyjn", "j": "hukm",
    "k": "jil", "l": "kop", "z": "asx", "x": "zsdc", "c": "xdfv",
    "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}

#: Latin -> Cyrillic look-alikes.  Real obfuscation services do exactly this to
#: break tokenisation and character-level features while the text still reads
#: normally to a person.
HOMOGLYPHS: Dict[str, str] = {
    "a": "а", "c": "с", "e": "е", "i": "і",
    "j": "ј", "o": "о", "p": "р", "s": "ѕ",
    "x": "х", "y": "у",
    "A": "А", "B": "В", "C": "С", "E": "Е",
    "H": "Н", "K": "К", "M": "М", "O": "О",
    "P": "Р", "T": "Т", "X": "Х",
}


# --------------------------------------------------------------------------
# text plumbing
# --------------------------------------------------------------------------

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "fig", "vol", "no", "approx", "cf", "al",
}

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])[\"'’”)\]]*\s+")
_PARAGRAPH_SPLIT = re.compile(r"(\n\s*\n|\n)")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")


def split_sentences(text: str) -> List[str]:
    """Split on real sentence boundaries, guarding common abbreviations.

    A regex is the right tool here even though the application uses a proper
    segmenter: this runs over the whole AI pool at build time, must not pull in
    spaCy, and a mis-split only costs one slightly odd training sample.
    """
    if not text or not text.strip():
        return []
    sentences: List[str] = []
    start = 0
    for match in _SENTENCE_BREAK.finditer(text):
        chunk = text[start:match.start()]
        tokens = chunk.rstrip().split()
        tail = tokens[-1].lower().strip(".!?\"')]") if tokens else ""
        # "Dr." and initials such as "J." are not sentence ends
        if tail in _ABBREVIATIONS or (len(tail) == 1 and tail.isalpha()):
            continue
        if chunk.strip():
            sentences.append(chunk.strip())
        start = match.end()
    remainder = text[start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def _map_paragraphs(text: str, transform: Callable[[str], str]) -> str:
    """Apply a sentence-level transform inside each paragraph.

    Paragraph structure is itself a detector signal, so attacks that reshuffle
    sentences must not silently flatten a document into one block.
    """
    parts = _PARAGRAPH_SPLIT.split(text)
    out: List[str] = []
    for index, part in enumerate(parts):
        if index % 2 == 1 or not part.strip():
            out.append(part)
        else:
            out.append(transform(part))
    return "".join(out)


def _match_case(source: str, replacement: str) -> str:
    """Carry the original token's capitalisation onto its replacement."""
    if not source or not replacement:
        return replacement
    if source.isupper() and len(source) > 1:
        return replacement.upper()
    if source[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _capitalise_first(text: str) -> str:
    for index, char in enumerate(text):
        if char.isalpha():
            return text[:index] + char.upper() + text[index + 1:]
    return text


def _lower_first(text: str) -> str:
    for index, char in enumerate(text):
        if char.isalpha():
            # leave acronyms and the pronoun "I" alone
            word = text[index:].split(" ", 1)[0]
            if word.isupper() or word in ("I", "I'm", "I've", "I'd", "I'll"):
                return text
            return text[:index] + char.lower() + text[index + 1:]
    return text


@lru_cache(maxsize=64)
def _phrase_pattern(sources: Tuple[str, ...]) -> re.Pattern:
    """One alternation, longest phrase first, cached across records."""
    ordered = sorted(sources, key=len, reverse=True)
    return re.compile(
        r"\b(?:" + "|".join(re.escape(s) for s in ordered) + r")\b",
        re.IGNORECASE)


def _replace_phrases(text: str, mapping: Sequence[Tuple[str, str]],
                     rng: random.Random, rate: float) -> str:
    """Case-insensitive whole-phrase replacement at a probability per hit.

    Deliberately a *single* pass over the text rather than one pass per pair.
    Sequential passes let an earlier substitution's output be re-matched by a
    later rule, which turned "it is important to note that" into word salad;
    one pass means every character of the input is rewritten at most once.
    """
    lookup: Dict[str, str] = {}
    for source, target in mapping:
        if source:
            lookup.setdefault(source.lower(), target)
    if not lookup:
        return text
    pattern = _phrase_pattern(tuple(lookup))

    def _sub(match: re.Match) -> str:
        original = match.group(0)
        target = lookup.get(original.lower())
        if target is None or rng.random() > rate:
            return original
        return _match_case(original, target)

    return pattern.sub(_sub, text)


# --------------------------------------------------------------------------
# attacks
# --------------------------------------------------------------------------


def paraphrase_synonym(text: str, rng: random.Random) -> str:
    """Swap a fraction of content words for bundled near-synonyms.

    This is the weakest possible model of a paraphraser: it changes the lexicon
    without touching syntax.  Real tools re-plan the sentence, so treat any
    robustness this buys as the easy half of the problem.
    """
    if not text:
        return text
    rate = 0.28

    def _sub(match: re.Match) -> str:
        token = match.group(0)
        options = SYNONYMS.get(token.lower())
        if not options or rng.random() > rate:
            return token
        return _match_case(token, rng.choice(options))

    return _WORD.sub(_sub, text)


def contraction_insert(text: str, rng: random.Random) -> str:
    """Contract expanded forms.  Model prose under-contracts; people do not."""
    if not text:
        return text
    pairs = sorted(CONTRACTIONS.items(), key=lambda kv: -len(kv[0]))
    return _replace_phrases(text, pairs, rng, rate=0.85)


def contraction_expand(text: str, rng: random.Random) -> str:
    """Expand contractions - the formalising direction a grammar tool takes."""
    if not text:
        return text
    pairs = sorted(EXPANSIONS.items(), key=lambda kv: -len(kv[0]))
    # apostrophes may be straight or curly in the source
    widened: List[Tuple[str, str]] = []
    for short, long in pairs:
        widened.append((short, long))
        if "'" in short:
            widened.append((short.replace("'", "’"), long))
    return _replace_phrases(text, widened, rng, rate=0.85)


def sentence_split(text: str, rng: random.Random) -> str:
    """Break long coordinated sentences in two at a real clause boundary."""
    if not text:
        return text

    def _transform(block: str) -> str:
        sentences = split_sentences(block)
        if not sentences:
            return block
        out: List[str] = []
        for sentence in sentences:
            words = sentence.split()
            match = re.search(r",\s+(and|but|so|which|while|although)\s+", sentence)
            if len(words) >= 18 and match and rng.random() < 0.6:
                head = sentence[:match.start()].rstrip(" ,")
                tail = sentence[match.end():]
                connector = match.group(1)
                lead = {"and": "", "but": "But ", "so": "So ",
                        "which": "This ", "while": "Meanwhile, ",
                        "although": "That said, "}.get(connector, "")
                tail = lead + (_capitalise_first(tail) if not lead else tail)
                if not head.endswith((".", "!", "?")):
                    head += "."
                out.extend([head, _capitalise_first(tail)])
            else:
                out.append(sentence)
        return " ".join(out)

    return _map_paragraphs(text, _transform)


def sentence_merge(text: str, rng: random.Random) -> str:
    """Join adjacent short sentences, flattening the uniform rhythm models emit."""
    if not text:
        return text
    joiners = (", and ", "; ", ", but ", " - ", ", which is why ")

    def _transform(block: str) -> str:
        sentences = split_sentences(block)
        if len(sentences) < 2:
            return block
        out: List[str] = []
        index = 0
        while index < len(sentences):
            current = sentences[index]
            if (index + 1 < len(sentences) and rng.random() < 0.55
                    and len(current.split()) <= 32
                    and current.endswith(".")):
                nxt = sentences[index + 1]
                joined = current[:-1] + rng.choice(joiners) + _lower_first(nxt)
                out.append(joined)
                index += 2
                continue
            out.append(current)
            index += 1
        return " ".join(out)

    return _map_paragraphs(text, _transform)


def clause_reorder(text: str, rng: random.Random) -> str:
    """Move a leading adverbial clause to the end of its sentence.

    Front-loaded subordinate clauses are one of the more stable model habits,
    so shifting them is a cheap attack on syntactic features.
    """
    if not text:
        return text
    lead = re.compile(
        r"^(While|Although|Though|Because|Since|When|If|After|Before|"
        r"As|Given that|Despite|In order to|By)\b[^,]{4,80},\s+")

    def _transform(block: str) -> str:
        sentences = split_sentences(block)
        if not sentences:
            return block
        out: List[str] = []
        for sentence in sentences:
            match = lead.match(sentence)
            if match and rng.random() < 0.85:
                clause = match.group(0).rstrip().rstrip(",")
                body = sentence[match.end():]
                punctuation = "."
                if body and body[-1] in ".!?":
                    punctuation = body[-1]
                    body = body[:-1]
                moved = (_capitalise_first(body.rstrip())
                         + " " + _lower_first(clause) + punctuation)
                out.append(moved)
            else:
                out.append(sentence)
        return " ".join(out)

    return _map_paragraphs(text, _transform)


def punctuation_shift(text: str, rng: random.Random) -> str:
    """Rewrite the punctuation habits that survive most paraphrasers.

    Em dashes, semicolons and the serial comma are heavily model-flavoured, and
    every humanizer worth the name normalises them.
    """
    if not text:
        return text
    result = text
    result = re.sub(r"\s*[—–]\s*",
                    lambda m: rng.choice([", ", " - ", " "]), result)
    result = re.sub(r";\s+",
                    lambda m: rng.choice([". ", ", ", " and "]), result)
    result = re.sub(r"…", "...", result)
    # drop the serial comma some of the time
    result = re.sub(r",\s+(and|or)\s+",
                    lambda m: (f" {m.group(1)} " if rng.random() < 0.5
                               else m.group(0)), result)
    result = re.sub(r":\s+", lambda m: ". " if rng.random() < 0.25 else ": ",
                    result)
    # a sentence that now starts lower-case after a ". " swap looks wrong
    result = re.sub(r"([.!?]\s+)([a-z])",
                    lambda m: m.group(1) + m.group(2).upper(), result)
    return result


def typo_injection(text: str, rng: random.Random, rate: float = 0.006) -> str:
    """Introduce adjacent-key swaps, dropped characters and doubled characters.

    ``rate`` is per alphabetic character and intentionally low: a document with
    visible typo density is not what a humanizer produces, it is what a careless
    writer produces, and conflating the two would teach the detector the wrong
    thing.
    """
    if not text:
        return text
    rate = max(0.0, min(rate, 0.05))
    chars = list(text)
    out: List[str] = []
    index = 0
    while index < len(chars):
        char = chars[index]
        if char.isalpha() and rng.random() < rate:
            mode = rng.choice(("swap", "drop", "double"))
            if mode == "swap":
                neighbours = _KEY_NEIGHBOURS.get(char.lower())
                if neighbours:
                    out.append(_match_case(char, rng.choice(neighbours)))
                else:
                    out.append(char)
            elif mode == "drop":
                pass
            else:
                out.extend([char, char])
        else:
            out.append(char)
        index += 1
    return "".join(out)


def spelling_convention_swap(text: str, rng: random.Random) -> str:
    """Flip British/American spelling wholesale.

    Direction is chosen from whichever convention already dominates, because a
    document with mixed conventions is itself a suspicious artefact.
    """
    if not text:
        return text
    lowered = text.lower()
    british = sum(lowered.count(b) for b, _ in SPELLING_PAIRS)
    american = sum(lowered.count(a) for _, a in SPELLING_PAIRS)
    if british >= american:
        pairs = [(b, a) for b, a in SPELLING_PAIRS]
    else:
        pairs = [(a, b) for b, a in SPELLING_PAIRS]
    return _replace_phrases(text, pairs, rng, rate=0.9)


def tone_informal(text: str, rng: random.Random) -> str:
    """Drop the register: formal phrasing out, conversational phrasing in."""
    if not text:
        return text
    result = _replace_phrases(text, TONE_PAIRS, rng, rate=0.7)
    return contraction_insert(result, rng)


def tone_formal(text: str, rng: random.Random) -> str:
    """Raise the register - what a grammar assistant does on "improve" mode."""
    if not text:
        return text
    reversed_pairs = [(informal, formal) for formal, informal in TONE_PAIRS
                      if informal.lower() not in _UNSAFE_FORMALISATIONS]
    result = _replace_phrases(text, reversed_pairs, rng, rate=0.55)
    return contraction_expand(result, rng)


def shorten(text: str, rng: random.Random) -> str:
    """Cut filler openers, redundant intensifiers and the weakest sentence."""
    if not text:
        return text
    result = text
    for phrase in FILLER_PHRASES:
        result = re.sub(r"\b" + re.escape(phrase) + r",?\s*",
                        lambda m: "" if rng.random() < 0.8 else m.group(0),
                        result, flags=re.IGNORECASE)
    for word in FILLER_WORDS:
        result = re.sub(r"\b" + re.escape(word) + r"\s+",
                        lambda m: "" if rng.random() < 0.55 else m.group(0),
                        result, flags=re.IGNORECASE)

    def _transform(block: str) -> str:
        sentences = split_sentences(block)
        if len(sentences) < 4:
            return block
        # the shortest sentence carries the least information
        victim = min(range(len(sentences)), key=lambda i: len(sentences[i].split()))
        if rng.random() < 0.7:
            sentences.pop(victim)
        return " ".join(sentences)

    result = _map_paragraphs(result, _transform)
    return _capitalise_first(re.sub(r"\s{2,}", " ", result).strip())


def expand(text: str, rng: random.Random) -> str:
    """Pad with filler openers and hedged asides - the "lengthen" button."""
    if not text:
        return text
    asides = (", which is fair enough,", ", at least on the face of it,",
              ", broadly speaking,", ", for what it's worth,")

    def _transform(block: str) -> str:
        sentences = split_sentences(block)
        if not sentences:
            return block
        out: List[str] = []
        for sentence in sentences:
            if rng.random() < 0.34:
                opener = rng.choice(FILLER_PHRASES)
                # "... that, this has" is ungrammatical; complementisers take
                # no comma before the clause they introduce
                joiner = " " if opener.lower().endswith("that") else ", "
                sentence = f"{opener}{joiner}{_lower_first(sentence)}"
            elif rng.random() < 0.30 and len(sentence.split()) > 8:
                words = sentence.split()
                cut = rng.randrange(3, min(len(words) - 2, 12))
                aside = rng.choice(asides).strip(" ,")
                words[cut - 1] = words[cut - 1].rstrip(",") + ","
                words.insert(cut, aside + ",")
                sentence = " ".join(words)
            out.append(sentence)
        return " ".join(out)

    return _map_paragraphs(text, _transform)


def hedge_insert(text: str, rng: random.Random) -> str:
    """Add first-person epistemic hedges.

    Assertive, unhedged claims are a model tell; the standard human edit is to
    soften them with an aside rather than an adverb.
    """
    if not text:
        return text

    def _transform(block: str) -> str:
        sentences = split_sentences(block)
        if not sentences:
            return block
        out: List[str] = []
        for sentence in sentences:
            if rng.random() < 0.28 and len(sentence.split()) > 6:
                hedge = rng.choice(HEDGES)
                if rng.random() < 0.5:
                    sentence = f"{_capitalise_first(hedge)}, {_lower_first(sentence)}"
                else:
                    body = sentence.rstrip()
                    punctuation = body[-1] if body[-1:] in ".!?" else ""
                    body = body[:-1] if punctuation else body
                    sentence = f"{body}, {hedge}{punctuation or '.'}"
            out.append(sentence)
        return " ".join(out)

    return _map_paragraphs(text, _transform)


def discourse_strip(text: str, rng: random.Random) -> str:
    """Remove sentence-initial discourse connectives.

    "Furthermore," at the head of a paragraph is close to a signature of
    generated prose, and it is the first thing an editor deletes.
    """
    if not text:
        return text
    pattern = re.compile(
        r"(?<![\w])(" + "|".join(re.escape(m) for m in DISCOURSE_MARKERS)
        + r")\s*,\s+", re.IGNORECASE)

    def _transform(block: str) -> str:
        sentences = split_sentences(block)
        if not sentences:
            return block
        out: List[str] = []
        for sentence in sentences:
            match = pattern.match(sentence)
            if match and rng.random() < 0.8:
                sentence = _capitalise_first(sentence[match.end():])
            out.append(sentence)
        return " ".join(out)

    return _map_paragraphs(text, _transform)


def quote_style_swap(text: str, rng: random.Random) -> str:
    """Toggle between straight and typographic quotes and apostrophes.

    Curly punctuation is a fingerprint of the tool that produced the text, not
    of its author, which is precisely why obfuscators normalise it.
    """
    if not text:
        return text
    curly = sum(text.count(c) for c in "‘’“”")
    straight = text.count('"') + text.count("'")
    if curly >= straight:
        table = {"‘": "'", "’": "'", "“": '"', "”": '"'}
        return "".join(table.get(c, c) for c in text)
    out: List[str] = []
    open_double = True
    for index, char in enumerate(text):
        if char == '"':
            out.append("“" if open_double else "”")
            open_double = not open_double
        elif char == "'":
            previous = text[index - 1] if index else " "
            following = text[index + 1] if index + 1 < len(text) else " "
            if previous.isalnum() and following.isalnum():
                out.append("’")          # apostrophe inside a word
            elif previous.isalnum():
                out.append("’")          # possessive
            else:
                out.append("‘")
        else:
            out.append(char)
    return "".join(out)


def whitespace_noise(text: str, rng: random.Random) -> str:
    """Introduce the ragged spacing that hand-edited documents accumulate."""
    if not text:
        return text
    result = re.sub(r"([.!?]) ",
                    lambda m: m.group(1) + ("  " if rng.random() < 0.35 else " "),
                    text)
    result = re.sub(r" ",
                    lambda m: " " if rng.random() < 0.004 else " ",
                    result)
    result = re.sub(r"\n",
                    lambda m: " \n" if rng.random() < 0.15 else "\n", result)
    return result


def homoglyph_swap(text: str, rng: random.Random) -> str:
    """Substitute a handful of Latin characters with Cyrillic look-alikes.

    Kept sparse on purpose - a couple of characters per few hundred is what the
    obfuscation services actually emit, because saturating the text makes it
    fail a copy-paste sanity check.  This is the one attack here that does not
    model editing at all; it models evasion tooling.
    """
    if not text:
        return text
    positions = [i for i, c in enumerate(text) if c in HOMOGLYPHS]
    if not positions:
        return text
    budget = max(1, min(len(text) // 200, 8))
    chosen = rng.sample(positions, min(budget, len(positions)))
    chars = list(text)
    for index in chosen:
        chars[index] = HOMOGLYPHS[chars[index]]
    return "".join(chars)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

ATTACKS: Dict[str, Callable[[str, random.Random], str]] = {
    "paraphrase_synonym": paraphrase_synonym,
    "contraction_insert": contraction_insert,
    "contraction_expand": contraction_expand,
    "sentence_split": sentence_split,
    "sentence_merge": sentence_merge,
    "clause_reorder": clause_reorder,
    "punctuation_shift": punctuation_shift,
    "typo_injection": typo_injection,
    "spelling_convention_swap": spelling_convention_swap,
    "tone_informal": tone_informal,
    "tone_formal": tone_formal,
    "shorten": shorten,
    "expand": expand,
    "hedge_insert": hedge_insert,
    "discourse_strip": discourse_strip,
    "quote_style_swap": quote_style_swap,
    "whitespace_noise": whitespace_noise,
    "homoglyph_swap": homoglyph_swap,
}

#: Named chains, ordered roughly by how much they disturb the text.  The names
#: are what lands in ``Record.humanizer``, so held-out evaluation by pipeline is
#: possible: train on ``light_edit`` and ``grammar_pass``, test on
#: ``aggressive_rewrite``, and you learn something about transfer.
PIPELINES: Dict[str, Tuple[str, ...]] = {
    "light_edit": (
        "discourse_strip", "contraction_insert", "punctuation_shift",
    ),
    "grammar_pass": (
        "contraction_expand", "tone_formal", "spelling_convention_swap",
        "punctuation_shift",
    ),
    "paraphrase_tool": (
        "paraphrase_synonym", "sentence_split", "clause_reorder",
        "discourse_strip", "punctuation_shift",
    ),
    "student_edit": (
        "contraction_insert", "tone_informal", "sentence_merge",
        "hedge_insert", "shorten", "typo_injection", "whitespace_noise",
    ),
    "aggressive_rewrite": (
        "paraphrase_synonym", "discourse_strip", "sentence_split",
        "sentence_merge", "clause_reorder", "tone_informal",
        "hedge_insert", "expand", "punctuation_shift",
    ),
    "obfuscation_tool": (
        "quote_style_swap", "homoglyph_swap", "whitespace_noise",
        "punctuation_shift",
    ),
}


def _derive_seed(seed: int, label: str) -> int:
    """Fold a name into a seed so each attack draws an independent stream.

    ``zlib.crc32`` rather than ``hash`` because the built-in is salted per
    process and would silently destroy reproducibility across runs.
    """
    return (int(seed) * 1_000_003 + zlib.crc32(label.encode("utf-8"))) % (2 ** 32)


def apply_attack(text: str, attack_name: str, seed: int = 0) -> str:
    """Run one registered attack.  Returns the input unchanged on empty text."""
    attack = ATTACKS.get(attack_name)
    if attack is None:
        raise KeyError(
            f"unknown attack '{attack_name}'. Available: {sorted(ATTACKS)}")
    if not isinstance(text, str) or not text.strip():
        return text if isinstance(text, str) else ""
    rng = random.Random(_derive_seed(seed, attack_name))
    try:
        result = attack(text, rng)
    except Exception:
        # A malformed sample must never abort a dataset build; a no-op sample
        # is preferable to a lost one, and the caller drops unchanged output.
        return text
    return result if isinstance(result, str) and result.strip() else text


def apply_pipeline(text: str, pipeline_name: str,
                   seed: int = 0) -> Tuple[str, List[str]]:
    """Run a named chain, returning the text and the attacks that bit.

    Only attacks that actually altered the text are reported, so
    ``Record.attack_type`` describes what happened rather than what was
    attempted - otherwise per-attack error analysis is meaningless.
    """
    chain = PIPELINES.get(pipeline_name)
    if chain is None:
        raise KeyError(
            f"unknown pipeline '{pipeline_name}'. Available: {sorted(PIPELINES)}")
    if not isinstance(text, str) or not text.strip():
        return (text if isinstance(text, str) else ""), []
    current = text
    applied: List[str] = []
    for step, attack_name in enumerate(chain):
        step_seed = _derive_seed(seed, f"{pipeline_name}:{step}")
        candidate = apply_attack(current, attack_name, step_seed)
        if candidate != current:
            applied.append(attack_name)
            current = candidate
    return current, applied

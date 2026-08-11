"""Language detection and per-language rule sets for style analysis.

Several style metrics are only meaningful against the grammar of the language
the text is written in. Passive voice is found by looking for auxiliary verbs;
hedging and formality are looked up in closed word lists; an apostrophe means a
stylistic contraction in English but a mandatory elision in French. Applying the
English tables to French or Spanish text does not merely lose signal — it
reports confident, wrong numbers: the passive rate is diluted toward zero
because no English auxiliary ever appears, while ``l'enfant`` and ``d'accord``
inflate the contraction rate.

This module supplies two things:

* :func:`detect_language` — a dependency-free detector that scores text against
  per-language closed-class function words and returns the best match, or
  :data:`UNKNOWN_LANGUAGE` when no language is clearly indicated.
* :func:`rules_for` — the :class:`LanguageRules` table used by the analyzer to
  compute language-sensitive metrics.

Detection is deliberately stopword-based rather than model-based: it needs no
dependency, no network, no model file, and is fully deterministic, which keeps
the project's local-only and hermetic-test guarantees intact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Language code used when no language is clearly indicated by the text.
UNKNOWN_LANGUAGE: str = "unknown"

#: Language whose rules are applied when the detected language has no table.
DEFAULT_LANGUAGE: str = "en"

#: Minimum share of tokens that must be recognized function words of the
#: winning language before detection is trusted. Below this the text is
#: predominantly proper nouns, numbers, or code, and is reported as unknown.
MIN_FUNCTION_WORD_SHARE: float = 0.08

#: Minimum tokens required before detection is attempted at all. Short
#: fragments carry too few function words to separate close Romance languages.
MIN_TOKENS_FOR_DETECTION: int = 12

#: How far ahead of the runner-up the winning language must score. Spanish,
#: Portuguese and Italian share much of their closed-class vocabulary, so a
#: narrow win is not evidence.
MIN_WINNER_MARGIN: float = 1.25

#: Word token used by the detector: Unicode letters/digits with optional
#: internal apostrophe, so accented text tokenizes correctly.
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


@dataclass(frozen=True)
class LanguageRules:
    """Closed-class word tables used by the language-sensitive metrics.

    Attributes:
        code: ISO 639-1 code this table describes.
        function_words: Common closed-class words, used for detection scoring.
        be_verbs: Auxiliary verbs that can introduce a passive construction.
        participle_suffixes: Regex matching a past-participle-looking token.
        irregular_participles: Participles the suffix regex cannot catch.
        first_person: First-person pronouns for the formality markers.
        second_person: Second-person pronouns for the formality markers.
        hedging: Hedging/softening words for the formality markers.
        mandatory_elisions: Elided forms that are compulsory grammar rather
            than a stylistic choice, and so must not count as contractions.
    """

    code: str
    function_words: frozenset[str]
    be_verbs: frozenset[str]
    participle_suffixes: re.Pattern[str]
    irregular_participles: frozenset[str] = frozenset()
    first_person: frozenset[str] = frozenset()
    second_person: frozenset[str] = frozenset()
    hedging: frozenset[str] = frozenset()
    mandatory_elisions: frozenset[str] = field(default_factory=frozenset)


_ENGLISH = LanguageRules(
    code="en",
    function_words=frozenset(
        {
            "the", "and", "that", "have", "for", "not", "with", "you", "this",
            "but", "his", "her", "they", "she", "will", "would", "there",
            "their", "what", "about", "which", "when", "make", "like", "just",
            "him", "know", "take", "into", "your", "some", "could", "them",
            "than", "then", "now", "only", "come", "its", "over", "also",
            "back", "after", "use", "how", "our", "well", "way", "even",
            "because", "any", "these", "give", "most", "should", "before",
        }
    ),
    be_verbs=frozenset({"am", "is", "are", "was", "were", "be", "been", "being"}),
    participle_suffixes=re.compile(r"^\w+(ed|en)$", re.IGNORECASE),
    irregular_participles=frozenset(
        {
            "done", "made", "said", "gone", "known", "shown", "found", "held",
            "kept", "left", "lost", "meant", "met", "paid", "put", "read",
            "run", "sent", "set", "sold", "spent", "told", "thought", "won",
        }
    ),
    first_person=frozenset(
        {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
    ),
    second_person=frozenset({"you", "your", "yours", "yourself", "yourselves"}),
    hedging=frozenset(
        {
            "maybe", "perhaps", "possibly", "probably", "likely", "seems",
            "seem", "appears", "appear", "somewhat", "rather", "fairly",
            "arguably", "presumably", "apparently", "roughly", "approximately",
            "kind", "sort", "guess", "suppose", "think", "believe", "might",
            "could", "may", "tend", "generally", "usually", "often",
        }
    ),
)

_FRENCH = LanguageRules(
    code="fr",
    function_words=frozenset(
        {
            "le", "la", "les", "de", "des", "du", "un", "une", "et", "que",
            "qui", "dans", "pour", "pas", "sur", "avec", "plus", "par", "ce",
            "cette", "ces", "mais", "nous", "vous", "ils", "elle", "elles",
            "son", "sa", "ses", "leur", "leurs", "au", "aux", "est", "sont",
            "être", "avoir", "faire", "comme", "tout", "tous", "bien", "aussi",
            "encore", "déjà", "quand", "parce", "donc", "alors", "avant",
            "après", "chez", "sans", "sous", "entre", "je", "moi", "toi",
        }
    ),
    be_verbs=frozenset(
        {
            "est", "sont", "était", "étaient", "être", "été", "suis", "es",
            "sommes", "êtes", "sera", "seront", "serait", "soit",
        }
    ),
    participle_suffixes=re.compile(r"^\w+(é|és|ée|ées|i|is|ie|ies|u|us|ue|ues)$", re.IGNORECASE),
    irregular_participles=frozenset(
        {"fait", "faits", "dit", "dits", "mis", "pris", "écrit", "ouvert", "mort", "né"}
    ),
    first_person=frozenset({"je", "me", "moi", "mon", "ma", "mes", "nous", "notre", "nos"}),
    second_person=frozenset({"tu", "te", "toi", "ton", "ta", "tes", "vous", "votre", "vos"}),
    hedging=frozenset(
        {
            "peut-être", "probablement", "sans", "doute", "semble", "paraît",
            "plutôt", "assez", "apparemment", "environ", "quelque", "certains",
            "pense", "crois", "suppose", "généralement", "souvent", "parfois",
        }
    ),
    mandatory_elisions=frozenset({"l", "d", "j", "n", "s", "c", "m", "t", "qu"}),
)

_SPANISH = LanguageRules(
    code="es",
    function_words=frozenset(
        {
            "el", "los", "las", "de", "del", "un", "una", "unos", "unas", "y",
            "que", "en", "por", "para", "con", "no", "se", "su", "sus", "al",
            "lo", "como", "más", "pero", "sus", "le", "ya", "o", "este",
            "esta", "estos", "estas", "sí", "porque", "esta", "entre", "cuando",
            "muy", "sin", "sobre", "también", "hasta", "donde", "quien",
            "desde", "todos", "todo", "nos", "nosotros", "ellos", "ellas",
            "es", "son", "está", "están", "hay", "ser", "estar", "tener",
        }
    ),
    be_verbs=frozenset(
        {
            "es", "son", "era", "eran", "fue", "fueron", "ser", "sido",
            "soy", "eres", "somos", "está", "están", "estaba", "estaban",
        }
    ),
    participle_suffixes=re.compile(r"^\w+(ado|ados|ada|adas|ido|idos|ida|idas)$", re.IGNORECASE),
    irregular_participles=frozenset(
        {"hecho", "dicho", "puesto", "visto", "escrito", "abierto", "muerto", "vuelto"}
    ),
    first_person=frozenset(
        {"yo", "me", "mi", "mis", "mío", "nosotros", "nos", "nuestro", "nuestra"}
    ),
    second_person=frozenset(
        {"tú", "te", "tu", "tus", "usted", "ustedes", "vosotros", "vuestro"}
    ),
    hedging=frozenset(
        {
            "quizá", "quizás", "tal", "vez", "probablemente", "posiblemente",
            "parece", "aparentemente", "bastante", "algo", "creo", "pienso",
            "supongo", "generalmente", "normalmente", "suele", "casi",
        }
    ),
)

_PORTUGUESE = LanguageRules(
    code="pt",
    function_words=frozenset(
        {
            "o", "os", "as", "de", "da", "do", "das", "dos", "um", "uma", "e",
            "que", "em", "por", "para", "com", "não", "se", "seu", "sua", "ao",
            "como", "mais", "mas", "já", "ou", "este", "esta", "isso", "porque",
            "entre", "quando", "muito", "sem", "sobre", "também", "até", "onde",
            "desde", "todos", "nós", "eles", "elas", "é", "são", "está",
            "estão", "há", "ser", "estar", "ter", "pelo", "pela", "nesse",
        }
    ),
    be_verbs=frozenset(
        {
            "é", "são", "era", "eram", "foi", "foram", "ser", "sido",
            "sou", "somos", "está", "estão", "estava", "estavam",
        }
    ),
    participle_suffixes=re.compile(r"^\w+(ado|ados|ada|adas|ido|idos|ida|idas)$", re.IGNORECASE),
    irregular_participles=frozenset(
        {"feito", "dito", "posto", "visto", "escrito", "aberto", "morto", "vindo"}
    ),
    first_person=frozenset({"eu", "me", "mim", "meu", "minha", "nós", "nosso", "nossa"}),
    second_person=frozenset({"tu", "te", "ti", "teu", "tua", "você", "vocês", "vosso"}),
    hedging=frozenset(
        {
            "talvez", "provavelmente", "possivelmente", "parece",
            "aparentemente", "bastante", "algo", "acho", "penso", "suponho",
            "geralmente", "normalmente", "quase", "costuma",
        }
    ),
)

_ITALIAN = LanguageRules(
    code="it",
    function_words=frozenset(
        {
            "il", "lo", "la", "i", "gli", "le", "di", "del", "della", "dei",
            "un", "una", "e", "che", "in", "per", "con", "non", "si", "suo",
            "sua", "al", "come", "più", "ma", "già", "o", "questo", "questa",
            "perché", "tra", "quando", "molto", "senza", "anche", "fino",
            "dove", "tutti", "tutto", "noi", "loro", "è", "sono", "era",
            "essere", "avere", "fare", "alla", "dalla", "nella", "prima",
        }
    ),
    be_verbs=frozenset(
        {
            "è", "sono", "era", "erano", "fu", "furono", "essere", "stato",
            "sei", "siamo", "siete", "sarà", "saranno",
        }
    ),
    participle_suffixes=re.compile(r"^\w+(ato|ati|ata|ate|ito|iti|ita|ite|uto|uti|uta|ute)$", re.IGNORECASE),
    irregular_participles=frozenset(
        {"fatto", "detto", "messo", "visto", "scritto", "aperto", "morto", "preso"}
    ),
    first_person=frozenset({"io", "mi", "me", "mio", "mia", "miei", "noi", "nostro", "nostra"}),
    second_person=frozenset({"tu", "ti", "te", "tuo", "tua", "voi", "vostro", "vostra"}),
    hedging=frozenset(
        {
            "forse", "probabilmente", "possibilmente", "sembra",
            "apparentemente", "abbastanza", "credo", "penso", "suppongo",
            "generalmente", "normalmente", "quasi", "spesso",
        }
    ),
    mandatory_elisions=frozenset({"l", "d", "un", "dell", "all", "nell", "sull", "dall"}),
)

#: Every language with a complete rule table.
_RULES: dict[str, LanguageRules] = {
    r.code: r for r in (_ENGLISH, _FRENCH, _SPANISH, _PORTUGUESE, _ITALIAN)
}

#: Human-readable names, used when a generated style guide names a language.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
}

#: Language codes this module can detect and analyze.
SUPPORTED_LANGUAGES: frozenset[str] = frozenset(_RULES)


def rules_for(code: str) -> LanguageRules:
    """Return the rule table for ``code``, falling back to the default language.

    Args:
        code: ISO 639-1 code, or :data:`UNKNOWN_LANGUAGE`.

    Returns:
        The matching :class:`LanguageRules`, or the default-language table when
        ``code`` has none.
    """
    return _RULES.get(code, _RULES[DEFAULT_LANGUAGE])


def detect_language(text: str) -> str:
    """Return the ISO 639-1 code of the language ``text`` is written in.

    Scores the text against each language's closed-class function words and
    returns the best match. Closed-class words are used because they are the
    highest-frequency, least topic-dependent signal available without a model.

    Args:
        text: Raw document text.

    Returns:
        A code in :data:`SUPPORTED_LANGUAGES`, or :data:`UNKNOWN_LANGUAGE` when
        the text is too short, carries too little function-word signal (a list
        of names and dates, a code block, a table of numbers), or does not
        favour one language clearly enough over the runner-up.
    """
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    if len(tokens) < MIN_TOKENS_FOR_DETECTION:
        return UNKNOWN_LANGUAGE

    scores = {
        code: sum(1 for t in tokens if t in rules.function_words)
        for code, rules in _RULES.items()
    }
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_score = ranked[0]
    if best_score / len(tokens) < MIN_FUNCTION_WORD_SHARE:
        return UNKNOWN_LANGUAGE

    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0
    if runner_up_score and best_score < runner_up_score * MIN_WINNER_MARGIN:
        return UNKNOWN_LANGUAGE
    return best

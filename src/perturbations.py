import random
import re
from typing import Callable
from .config import RANDOM_SEED

random.seed(RANDOM_SEED)

EMOJIS_POS = ["😀", "😄", "😊", "😍", "🤩", "🔥", "👍"]
EMOJIS_NEG = ["😢", "😭", "😡", "🤬", "👎"]
HASHTAGS = ["#love", "#fail", "#lol", "#omg", "#mood"]
SLANG_MAP = {
    "really": "rly",
    "very": "sooo",
    "right now": "rn",
    "because": "cuz",
    "great": "fire",
    "good": "lit",
}
CODESWITCH_WORDS = ["很好玩", "一般般", "有点糟糕", "挺不错", "有点贵"]


def _random_indices(n_tokens: int, intensity: float):
    k = max(1, int(n_tokens * intensity))
    return set(random.sample(range(n_tokens), min(k, n_tokens)))


def add_emoji_noise(text: str, intensity: float) -> str:
    tokens = text.split()
    idxs = _random_indices(len(tokens), intensity)
    new_tokens = []
    for i, tok in enumerate(tokens):
        new_tokens.append(tok)
        if i in idxs:
            emo = random.choice(EMOJIS_POS + EMOJIS_NEG)
            new_tokens.append(emo)
    return " ".join(new_tokens)


def add_hashtag_noise(text: str, intensity: float) -> str:
    tokens = text.split()
    idxs = _random_indices(len(tokens), intensity)
    new_tokens = []
    for i, tok in enumerate(tokens):
        new_tokens.append(tok)
        if i in idxs:
            new_tokens.append(random.choice(HASHTAGS))
    return " ".join(new_tokens)


def add_repetition_noise(text: str, intensity: float) -> str:
    def stretch(word: str) -> str:
        if len(word) <= 3:
            return word + random.choice(["!", "!!", "!!!"])
        i = random.randint(1, len(word) - 2)
        return word[:i] + word[i] * random.randint(2, 4) + word[i + 1:] + random.choice(["", "!", "!!"])

    tokens = text.split()
    idxs = _random_indices(len(tokens), intensity)
    new_tokens = []
    for i, tok in enumerate(tokens):
        if i in idxs:
            new_tokens.append(stretch(tok))
        else:
            new_tokens.append(tok)
    return " ".join(new_tokens)


def add_slang_noise(text: str, intensity: float) -> str:
    tokens = text.split()
    idxs = _random_indices(len(tokens), intensity)
    new_tokens = []
    for i, tok in enumerate(tokens):
        low = tok.lower()
        replaced = False
        for k, v in SLANG_MAP.items():
            if low == k:
                new_tokens.append(v)
                replaced = True
                break
        if not replaced:
            new_tokens.append(tok)
    return " ".join(new_tokens)


def add_spelling_noise(text: str, intensity: float) -> str:
    def corrupt(word: str) -> str:
        if len(word) <= 3:
            return word
        i = random.randint(0, len(word) - 2)
        j = i + 1
        chars = list(word)
        chars[i], chars[j] = chars[j], chars[i]
        return "".join(chars)

    tokens = text.split()
    idxs = _random_indices(len(tokens), intensity)
    new_tokens = []
    for i, tok in enumerate(tokens):
        if i in idxs and tok.isalpha():
            new_tokens.append(corrupt(tok))
        else:
            new_tokens.append(tok)
    return " ".join(new_tokens)


def add_codeswitch_noise(text: str, intensity: float) -> str:
    tokens = text.split()
    idxs = _random_indices(len(tokens), intensity)
    new_tokens = []
    for i, tok in enumerate(tokens):
        new_tokens.append(tok)
        if i in idxs:
            new_tokens.append(random.choice(CODESWITCH_WORDS))
    return " ".join(new_tokens)


NOISE_FUNCS: dict[str, Callable[[str, float], str]] = {
    "emoji": add_emoji_noise,
    "spelling": add_spelling_noise,
    "slang": add_slang_noise,
    "hashtag": add_hashtag_noise,
    "repetition": add_repetition_noise,
    "codeswitch": add_codeswitch_noise,
}

"""Content-free hardening transform: the trust boundary between raw input and
everything KeySpark stores. Raw typed characters must never reach Kafka, the
parquet archive, or any feature table. This module maps a raw key identity to two
content-free tokens and nothing else:

  key_class : a coarse category (letter / digit / punct / space / correction /
              enter / tab / nav / modifier / function / other). Carries the
              structural role of the key, never which letter it was. Preserves the
              word count (space), correction count (backspace/delete), and the
              key-class mix features.
  key_token : a minute-rotating keyed hash of the (case-folded) key. Within one
              wall-clock minute the same physical key maps to the same token, so
              distinct-token counting inside a one-minute window reproduces
              key_diversity exactly. Across minutes the salt input changes, so
              tokens are not linkable window-to-window. The key is a random secret
              generated per process and never persisted, so tokens are
              irreversible: recovering content would need to brute-force the key
              alphabet against an unknown secret, and only within a single minute
              of tokens (too little material for frequency analysis).

Pairing key_down to key_up (for dwell) is deliberately NOT done here. The agent
emits atomic key_down / key_up events and the featurizer pairs them on key_token,
so the live pipeline and the reprocessed archive pair over the SAME
post-transport event stream and therefore share one missingness profile. Pairing
inside the agent would see the pre-loss stream and break that parity.
"""

from __future__ import annotations

import hashlib
import os

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
TOKEN_BYTES = 6          # tune: key_token length in bytes (12 hex chars)
KEY_EVENT_TYPES = ("key_down", "key_up")

# pynput special-key reprs ("Key.<name>") mapped to a structural class. Anything
# not listed and not a single printable character falls through to "other".
_SPECIAL_CLASS = {
    "space": "space",
    "backspace": "correction",
    "delete": "correction",
    "enter": "enter",
    "tab": "tab",
    "esc": "nav",
    "left": "nav", "right": "nav", "up": "nav", "down": "nav",
    "home": "nav", "end": "nav", "page_up": "nav", "page_down": "nav",
    "cmd": "modifier", "cmd_l": "modifier", "cmd_r": "modifier",
    "ctrl": "modifier", "ctrl_l": "modifier", "ctrl_r": "modifier",
    "alt": "modifier", "alt_l": "modifier", "alt_r": "modifier", "alt_gr": "modifier",
    "shift": "modifier", "shift_l": "modifier", "shift_r": "modifier",
    "caps_lock": "modifier", "num_lock": "modifier", "scroll_lock": "modifier",
}

# The stable, ordered vocabulary of key classes. Fixed order so downstream
# one-hot / mix-fraction features have a stable column layout.
KEY_CLASSES = (
    "letter", "digit", "punct", "space", "correction",
    "enter", "tab", "nav", "modifier", "function", "other",
)


# --------------------------------------------------------------------------
# Key classification and tokenization
# --------------------------------------------------------------------------
def classify_key(raw_key: str | None) -> str:
    """Map a raw key id (a printable character or a pynput "Key.<name>" repr) to
    one of KEY_CLASSES. Content-free: 'a' and 'z' both return "letter".
    """
    if raw_key is None:
        return "other"
    k = str(raw_key)
    if k.startswith("Key."):
        name = k[4:]
        if name in _SPECIAL_CLASS:
            return _SPECIAL_CLASS[name]
        if len(name) >= 2 and name[0] == "f" and name[1:].isdigit():
            return "function"
        return "other"
    if len(k) == 1:
        if k.isalpha():
            return "letter"
        if k.isdigit():
            return "digit"
        if k.isspace():
            return "space"
        if k.isprintable():
            return "punct"
        return "other"
    return "other"


def _canonical_key(raw_key: str | None) -> str:
    """The token pre-image: case-fold single printable characters so a Shift-down
    'A' and its release 'a' (pynput reports the shifted then unshifted form) tokenize
    identically and pair. Special keys pass through as their repr.
    """
    k = "" if raw_key is None else str(raw_key)
    if len(k) == 1:
        return k.lower()
    return k


def new_secret() -> bytes:
    """A fresh 16-byte token secret. Generate once per process / reprocessing run;
    never persist it (that is what makes tokens irreversible).
    """
    return os.urandom(16)


def key_token(raw_key: str | None, ts: float, secret: bytes) -> str:
    """Minute-rotating keyed hash of the case-folded key. Stable within a
    wall-clock minute, unlinkable across minutes, irreversible without ``secret``.
    """
    minute = int(ts // 60)
    msg = f"{minute}:{_canonical_key(raw_key)}".encode()
    return hashlib.blake2b(msg, key=secret, digest_size=TOKEN_BYTES).hexdigest()


# --------------------------------------------------------------------------
# Event / frame hardening
# --------------------------------------------------------------------------
def harden_event(ev: dict, secret: bytes) -> dict:
    """Return a content-free copy of one raw event. Key events lose ``key`` and
    gain ``key_class`` + ``key_token``; move/click/scroll pass through unchanged
    (they never carried content). The input dict is not mutated.
    """
    if ev.get("type") not in KEY_EVENT_TYPES:
        return dict(ev)
    out = {k: v for k, v in ev.items() if k != "key"}
    out["key_class"] = classify_key(ev.get("key"))
    out["key_token"] = key_token(ev.get("key"), float(ev["ts"]), secret)
    return out


def harden_frame(df, secret: bytes | None = None):
    """Reprocess a raw-event pandas frame into hardened form: drop the ``key``
    column, add ``key_class`` and ``key_token`` for key events (null elsewhere).
    Uses a fresh random secret unless one is supplied. Returns a new frame; the
    raw ``key`` column is never returned.
    """
    import numpy as np
    import pandas as pd

    if secret is None:
        secret = new_secret()
    out = df.copy()
    is_key = out["type"].isin(KEY_EVENT_TYPES).to_numpy() if "type" in out else np.zeros(len(out), bool)
    raw = out["key"] if "key" in out.columns else pd.Series([None] * len(out), index=out.index)
    ts = out["ts"] if "ts" in out.columns else pd.to_datetime(out["event_time"]).astype("int64") / 1e9

    key_class = np.where(is_key, [classify_key(k) for k in raw], None)
    key_tok = np.where(
        is_key,
        [key_token(k, float(t), secret) for k, t in zip(raw, ts)],
        None,
    )
    out["key_class"] = key_class
    out["key_token"] = key_tok
    if "key" in out.columns:
        out = out.drop(columns=["key"])
    return out

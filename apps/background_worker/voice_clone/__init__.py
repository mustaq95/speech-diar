"""Voice cloning: register a new speaker on the TryHamsa TTS pod from one
reference clip, so `hamsa-tts` can then synthesize with it by name.

NOT a fourth entry in `TTS_ENGINES`, and deliberately so. Cloning produces no
audio and has no `delivery`, no voice list and nothing to measure against the
other engines; what it produces is a *speaker name*, which the existing
`hamsa-tts` engine uses like any of the pod's built-in voices. Registering it
as an engine would put a row in the TTS comparison that can never render a
clip.

Same runner/adapter split as `../tts/` and `../transcription/`: `runner.py`
executes the two pod calls and returns their NATIVE output untouched;
`adapter.py` is the only code allowed to understand those shapes. No registry
and no provider abstraction -- there is exactly one pod that can do this, and
`../tts/__init__.py` already explains why three single-use engines did not earn
a base class either.
"""

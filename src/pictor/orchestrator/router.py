"""Turn a conversation into a Plan: answer in text, generate, or edit.

No LLM in the loop: the user's text is the prompt (Qwen-Image-2.1 understands English best),
a few inline tokens are parsed (seed, steps, aspect words, "transparent"), and the mode is
decided by what is attached: images → edit, none → generate, a question → help.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Literal

from PIL import Image

from pictor.backends.base import Capabilities, GenerationRequest
from pictor.config import Settings
from pictor.orchestrator.conversation import Turn, last_user, latest_images
from pictor.orchestrator.replies import t
from pictor.util.images import ASPECT_PRESETS, clamp_size, size_for_ratio, size_like

Action = Literal["chat", "generate", "edit"]

RGBA_PROMPT = "This is an RGBA image with transparency. {prompt}. The image has alpha channel and the background is transparent."

_QUESTION = re.compile(
    r"^\s*(what can you|what do you|who are you|how do i|how does this|help|hello|hi|hey|thanks|thank you|"
    r"qu'est-ce que tu|que sais-tu|comment ça marche|aide|bonjour|salut|merci)\b",
    re.I,
)
_EDIT = re.compile(
    r"\b(the previous|previous image|this image|the image|that image|last image|it again|"
    r"la pr[ée]c[ée]dente|cette image|l'image)\b",
    re.I,
)
_TRANSPARENT = re.compile(r"\b(transparent|transparency|no background|rgba|sans fond|transparence)\b", re.I)
_SEED = re.compile(r"\bseed\s*[:=]?\s*(\d+)\b", re.I)
_STEPS = re.compile(r"\b(\d{1,3})\s*(steps?|[ée]tapes?)\b", re.I)
_RATIO_WORDS = [
    (re.compile(r"\b(\d{1,2})\s*[:x]\s*(\d{1,2})\b"), None),
    (re.compile(r"\b(portrait|vertical|story|phone wallpaper)\b", re.I), "9:16"),
    (re.compile(r"\b(landscape|horizontal|wide|widescreen|banner|cinematic|paysage)\b", re.I), "16:9"),
    (re.compile(r"\b(square|carr[ée])\b", re.I), "1:1"),
    (re.compile(r"\b(poster|affiche)\b", re.I), "3:4"),
]
_SIZE_2K = re.compile(r"\b(2k|2048|high resolution|hi-?res|haute r[ée]solution)\b", re.I)


@dataclass
class Plan:
    action: Action
    reply: str = ""  # chat answer or intro line before a job
    request: GenerationRequest | None = None
    notes: list[str] = field(default_factory=list)


class Router:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def plan(self, turns: list[Turn], caps: Capabilities, overrides: dict | None = None,
                   backend_name: str = "", device: str = "") -> Plan:
        overrides = dict(overrides or {})
        user = last_user(turns)
        if user is None:
            return Plan(action="chat", reply=self._help(caps, backend_name, device))

        text = user.text.strip()
        attached = list(user.images)
        forced = overrides.get("mode") if overrides.get("mode") in ("chat", "generate", "edit") else None

        # 1. Action -----------------------------------------------------------------------
        action: Action = forced or self._heuristic_action(text, attached, turns, user)
        if action == "chat":
            reply = t("image_only") if (attached and not text) else self._help(caps, backend_name, device)
            return Plan(action="chat", reply=reply)

        # 2. Reference images ---------------------------------------------------------------
        refs: list[Image.Image] = attached
        if action == "edit" and not refs:
            refs = latest_images(turns, before=user, limit=caps.max_reference_images)
        if action == "edit" and not refs:
            action = "generate"  # nothing to edit: draw it instead
        if action == "generate" and refs:
            action = "edit"  # an attached image always conditions the output
        refs = refs[: caps.max_reference_images]

        # 3. Prompt -----------------------------------------------------------------------------
        prompt = self._clean_prompt(text) or "high quality image"
        notes: list[str] = []
        transparent = bool(overrides.get("transparent", bool(_TRANSPARENT.search(text))))
        if transparent and not caps.rgba:
            notes.append("rgba_not_produced")
            transparent = False
        if transparent:
            prompt = RGBA_PROMPT.format(prompt=prompt.rstrip(". "))

        # 4. Parameters -------------------------------------------------------------------------
        width, height = self._size(text, overrides, refs if action == "edit" else [], caps)
        steps = int(overrides.get("steps") or self._steps_from_text(text) or self.settings.default_steps)
        steps = max(1, min(steps, self.settings.max_steps))
        m = _SEED.search(text)
        seed = int(overrides["seed"]) if overrides.get("seed") is not None else (int(m.group(1)) if m else random.randint(0, 2**31 - 1))
        guidance = float(overrides.get("guidance") or self.settings.default_guidance)
        negative = overrides.get("negative_prompt") or None

        request = GenerationRequest(
            prompt=prompt, width=width, height=height, steps=steps, seed=seed, guidance=guidance,
            negative_prompt=negative, references=refs if action == "edit" else [], transparent=transparent,
        )
        intro_key = "intro_edit" if action == "edit" else "intro_generate"
        reply = t(
            intro_key,
            prompt=self._short(prompt), w=width, h=height, steps=steps, seed=seed,
            n=len(request.references), s="s" if len(request.references) > 1 else "",
        )
        return Plan(action=action, reply=reply, request=request, notes=notes)

    # -- helpers ---------------------------------------------------------------------------
    @staticmethod
    def _heuristic_action(text: str, attached: list, turns: list[Turn], user: Turn) -> Action:
        if not text:
            return "chat"  # empty message, with or without an image: ask what to do
        if attached:
            return "edit"
        if _EDIT.search(text) and latest_images(turns, before=user, limit=1):
            return "edit"
        if _QUESTION.match(text) and len(text) < 120:
            return "chat"
        return "generate"

    def _help(self, caps: Capabilities, backend: str, device: str) -> str:
        return t("help", n=caps.max_reference_images, backend=backend or "?", device=device or "?")

    def _size(self, text: str, overrides: dict, refs: list, caps: Capabilities) -> tuple[int, int]:
        base = max(self.settings.default_width, self.settings.default_height)
        if overrides.get("base_size"):
            base = min(int(overrides["base_size"]), caps.max_side)
        elif _SIZE_2K.search(text):
            base = min(2048, caps.max_side)
        if overrides.get("width") and overrides.get("height"):
            return clamp_size(int(overrides["width"]), int(overrides["height"]), caps.max_side)
        ratio = overrides.get("aspect_ratio") or self._ratio_from_text(text)
        if ratio:
            try:
                return size_for_ratio(ratio, base=base, max_side=caps.max_side)
            except (ValueError, ZeroDivisionError):
                pass
        if refs:
            return size_like(refs[-1], base=base, max_side=caps.max_side)
        if base != max(self.settings.default_width, self.settings.default_height):
            return clamp_size(base, base, caps.max_side)
        return clamp_size(self.settings.default_width, self.settings.default_height, caps.max_side)

    @staticmethod
    def _ratio_from_text(text: str) -> str | None:
        for rx, preset in _RATIO_WORDS:
            m = rx.search(text)
            if not m:
                continue
            if preset:
                return preset
            a, b = int(m.group(1)), int(m.group(2))
            if a and b and a <= 32 and b <= 32:
                key = f"{a}:{b}"
                return key if key in ASPECT_PRESETS else key
        return None

    @staticmethod
    def _steps_from_text(text: str) -> int | None:
        m = _STEPS.search(text)
        return int(m.group(1)) if m else None

    @staticmethod
    def _clean_prompt(text: str) -> str:
        s = _SEED.sub("", text)
        s = _STEPS.sub("", s)
        return re.sub(r"\s{2,}", " ", s).strip(" ,.;")

    @staticmethod
    def _short(prompt: str, n: int = 160) -> str:
        p = prompt.replace("\n", " ")
        return p if len(p) <= n else p[: n - 1] + "…"

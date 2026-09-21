import pytest
from PIL import Image

from pictor.backends.base import Capabilities
from pictor.config import Settings
from pictor.orchestrator.conversation import Turn
from pictor.orchestrator.router import Router

CAPS = Capabilities()


@pytest.fixture
def router():
    return Router(Settings(_env_file=None))


def img(w=64, h=32):
    return Image.new("RGB", (w, h), "blue")


async def test_question_is_chat(router):
    plan = await router.plan([Turn("user", "What can you do?")], CAPS, backend_name="torch", device="mps")
    assert plan.action == "chat" and "Pictor" in plan.reply


async def test_prompt_is_passed_as_written_with_inline_options(router):
    plan = await router.plan([Turn("user", "a red fox in the snow, 16:9, seed 7, 12 steps")], CAPS)
    assert plan.action == "generate"
    r = plan.request
    assert r.seed == 7 and r.steps == 12 and r.width > r.height
    assert r.prompt == "a red fox in the snow, 16:9"


async def test_attached_image_is_an_edit_keeping_its_ratio(router):
    plan = await router.plan([Turn("user", "make the sky orange", images=[img()])], CAPS)
    assert plan.action == "edit"
    assert len(plan.request.references) == 1
    assert plan.request.width > plan.request.height


async def test_previous_image_is_reused_for_edits(router):
    turns = [
        Turn("user", "a cat"),
        Turn("assistant", "Done.", images=[img(32, 64)]),
        Turn("user", "change the background of this image to a beach"),
    ]
    plan = await router.plan(turns, CAPS)
    assert plan.action == "edit" and len(plan.request.references) == 1
    assert plan.request.height > plan.request.width


async def test_multiple_references_are_kept_in_order(router):
    refs = [img(10, 10), img(20, 10), img(30, 10)]
    plan = await router.plan([Turn("user", "put the first two in the third scene", images=refs)], CAPS)
    assert [r.width for r in plan.request.references] == [10, 20, 30]


async def test_image_alone_asks_what_to_do(router):
    plan = await router.plan([Turn("user", "", images=[img()])], CAPS)
    assert plan.action == "chat" and "What should I do" in plan.reply


async def test_transparent_wraps_the_prompt(router):
    plan = await router.plan([Turn("user", "a fox logo, transparent background")], CAPS)
    assert plan.request.transparent and plan.request.prompt.startswith("This is an RGBA image")


async def test_overrides_win(router):
    plan = await router.plan([Turn("user", "hello")], CAPS, {"mode": "generate", "width": 128, "height": 256, "steps": 2, "seed": 1})
    assert plan.action == "generate"
    assert (plan.request.width, plan.request.height, plan.request.steps, plan.request.seed) == (128, 256, 2, 1)

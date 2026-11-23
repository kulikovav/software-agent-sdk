import asyncio
from collections.abc import Callable, Coroutine, Iterable
from dataclasses import dataclass
from typing import Any

from litellm.types.utils import ModelResponseStream


# Type alias for stream chunks
LLMStreamChunk = ModelResponseStream

TokenCallbackType = Callable[[LLMStreamChunk], None]
AsyncTokenCallbackType = Callable[[LLMStreamChunk], Coroutine[Any, Any, None]]


@dataclass(slots=True)
class _StreamedToolCall:
    """Tracks which call owns one raw stream index.

    ``index`` is the value its fragments are rewritten to.
    """

    index: int
    id: str = ""
    name: str = ""
    has_arguments: bool = False


def _take_index(used: set[int], raw_index: int) -> int:
    """Return an index no call in this choice has claimed yet.

    A restarted call takes the next index above every used one, so it cannot
    collide with an index a later call legitimately claims.
    """
    if raw_index not in used:
        used.add(raw_index)
        return raw_index
    index = max(used) + 1
    used.add(index)
    return index


def _restarts_call(owner: _StreamedToolCall, call_id: str, name: str) -> bool:
    """Whether a fragment starts a new call on an index another call owns.

    A gateway that reuses one index marks the new call with a fresh id. Some
    omit the id, so a different name on an index whose call already has
    arguments marks the restart.
    """
    if call_id and owner.id:
        return call_id != owner.id
    if not name or not owner.name:
        return False
    return owner.has_arguments and name != owner.name


def _normalize_tool_call_indices(chunks: Iterable[Any]) -> None:
    """Give tool calls that restart one stream index a distinct index.

    Some gateways stream every tool call of a turn under index 0, each with a
    new id and (often) its complete arguments in one delta.
    ``litellm.stream_chunk_builder`` groups fragments by index alone, so the
    second call's arguments are appended to the first: the joined string then
    fails to parse as JSON ("Extra data: line 1 column ...") and the tool never
    runs. Renumbering the restarted call keeps the builder's groups apart.

    Rewrites each fragment's ``index`` in place: the chunk objects are shared,
    so a consumer that keeps the chunks sees the rewritten indices. The first
    call keeps its raw index, so nothing changes for a well-formed stream,
    where fragments share an index only within one call.
    """
    owners: dict[tuple[int, int], _StreamedToolCall] = {}
    owners_by_id: dict[tuple[int, str], _StreamedToolCall] = {}
    used_indices: dict[int, set[int]] = {}

    for chunk in chunks:
        for choice in chunk.choices or ():
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            # A ModelResponse rewraps StreamingChoices, so delta can be a dict.
            tool_calls = (
                delta.get("tool_calls") if isinstance(delta, dict) else delta.tool_calls
            )
            choice_index = choice.index or 0
            for fragment in tool_calls or ():
                if not fragment:
                    continue
                raw_index = fragment.get("index")
                if raw_index is None:
                    raw_index = 0
                function = fragment.get("function") or {}
                call_id = str(fragment.get("id") or "").strip()
                name = str(function.get("name") or "").strip()
                arguments = function.get("arguments")

                # ponytail: a new call that reuses its predecessor's id is
                # indistinguishable from a continuation; sniff the JSON only
                # if a provider starts doing that.
                known_owner = (
                    owners_by_id.get((choice_index, call_id)) if call_id else None
                )
                if known_owner is not None:
                    fragment["index"] = known_owner.index
                    if arguments:
                        known_owner.has_arguments = True
                    continue

                owner = owners.get((choice_index, raw_index))
                if owner is None or _restarts_call(owner, call_id, name):
                    index = _take_index(
                        used_indices.setdefault(choice_index, set()), raw_index
                    )
                    owner = _StreamedToolCall(index=index)
                    owners[(choice_index, raw_index)] = owner
                    if call_id:
                        owners_by_id[(choice_index, call_id)] = owner
                if call_id:
                    owner.id = call_id
                if name:
                    owner.name = name
                if arguments:
                    owner.has_arguments = True
                fragment["index"] = owner.index


# Accepts either sync or async token callbacks for async methods.
AnyTokenCallbackType = TokenCallbackType | AsyncTokenCallbackType


async def _invoke_token_callback(
    cb: AnyTokenCallbackType, chunk: LLMStreamChunk
) -> None:
    """Invoke a token callback, awaiting if it returns a coroutine.

    Handles both plain coroutine functions **and** callable objects
    whose ``__call__`` is async (``asyncio.iscoroutinefunction`` does
    not detect the latter).
    """
    result = cb(chunk)  # type: ignore[arg-type]
    if asyncio.iscoroutine(result):
        await result

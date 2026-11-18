"""Model-facing same-tree messaging tools: ``agent_message`` and ``list_agents``.

The handlers are deliberately thin: identity, authorization, state and delivery live
in ``tools.delegate_tool_messages``. The toolset is additive and never grants spawn
or cross-tree authority.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from tools.registry import registry
from tools.delegate_tool_messages import list_tree_agents, send_agent_message

AGENT_MESSAGE_TOOL_NAME = "agent_message"
LIST_AGENTS_TOOL_NAME = "list_agents"


def agent_message(target: Optional[str] = None, message: Optional[str] = None,
                  message_id: Optional[str] = None, parent_agent: Any = None) -> str:
    """Send one addressed message to another agent in the caller's tree."""
    return json.dumps(
        send_agent_message(parent_agent, target, message, message_id=message_id),
        ensure_ascii=False,
    )


def list_agents(parent_agent: Any = None) -> str:
    """List the caller's root and tree members with stable ids and statuses."""
    return json.dumps(list_tree_agents(parent_agent), ensure_ascii=False)


AGENT_MESSAGE_SCHEMA = {
    "name": AGENT_MESSAGE_TOOL_NAME,
    "description": (
        "Send a message to another agent in YOUR delegation tree (the root or a sibling/descendant). "
        "The recipient sees your agent id and the exact message in a new user turn. A running "
        "recipient receives it at its next step boundary; a finished continuable child is resumed "
        "on its own conversation. Same tree only: self, unknown ids, other trees/profiles and "
        "cancelled/closed targets are refused. `accepted` means the message was admitted, not "
        "that it was processed or made durable. Use list_agents to find ids. This is not a "
        "scheduler: the recipient acts only when it is woken by this message or its own work."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Recipient agent id from list_agents (or 'lead' for the root).",
            },
            "message": {
                "type": "string",
                "description": "The message body. Compose it for the other agent; include the concrete ask.",
            },
            "message_id": {
                "type": "string",
                "description": (
                    "Optional immutable idempotency key (letters, digits, . _ : -; max 128). "
                    "Reusing an id with the same sender/recipient/body returns the recorded state; "
                    "reusing it with different content is refused."
                ),
            },
        },
        "required": ["target", "message"],
    },
}

LIST_AGENTS_SCHEMA = {
    "name": LIST_AGENTS_TOOL_NAME,
    "description": (
        "List your entire delegation tree: the root ('lead') and every live or retained child in "
        "the same root and profile, including yourself. Ids are stable subagent ids; use them with "
        "agent_message. Status 'running' means a turn is executing; 'inactive' does not say the "
        "task succeeded or failed."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}


registry.register(
    name=AGENT_MESSAGE_TOOL_NAME,
    toolset="agent_messaging",
    schema=AGENT_MESSAGE_SCHEMA,
    handler=lambda args, **kw: agent_message(
        target=args.get("target"), message=args.get("message"), message_id=args.get("message_id"),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=lambda: True,
    emoji="✉️",
)

registry.register(
    name=LIST_AGENTS_TOOL_NAME,
    toolset="agent_messaging",
    schema=LIST_AGENTS_SCHEMA,
    handler=lambda args, **kw: list_agents(parent_agent=kw.get("parent_agent")),
    check_fn=lambda: True,
    emoji="👥",
)

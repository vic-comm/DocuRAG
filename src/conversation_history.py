"""
ConversationSummaryBufferMessageHistory:
  - Keeps the most recent `k` messages in full
  - Older messages are summarised by the LLM into a single SystemMessage
    prepended to the buffer, so the model always has context without
    exceeding the context window.
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConversationSummaryBufferMessageHistory(BaseChatMessageHistory, BaseModel):
    """
    Hybrid memory: full buffer for recent messages + LLM summary for older ones.

    Args:
        llm:  The language model used to generate summaries.
        k:    Maximum number of recent messages to keep in full.
              Must be even (pairs of human + AI turns). Defaults to 6.
    """

    model_config = {"arbitrary_types_allowed": True}

    messages: list[BaseMessage] = Field(default_factory=list)
    llm: ChatGroq = Field(default_factory=ChatGroq)
    k: int = Field(default=6)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """
        Append new messages to the history.
        If the buffer grows beyond `k`, the oldest messages are summarised
        and replaced with a single SystemMessage.
        """
        existing_summary: Optional[SystemMessage] = None

        # 1. Pull out any existing summary so we update it, not duplicate it
        if self.messages and isinstance(self.messages[0], SystemMessage):
            existing_summary = self.messages.pop(0)
            logger.debug("Found existing summary — will merge.")

        # 2. Append incoming messages
        self.messages.extend(messages)

        # 3. No trimming needed — re-attach the summary and return
        if len(self.messages) <= self.k:
            if existing_summary:
                self.messages = [existing_summary] + self.messages
            return

        # 4. Trim: separate overflow messages from the recent buffer
        overflow_count = len(self.messages) - self.k
        old_messages = self.messages[:overflow_count]
        self.messages = self.messages[overflow_count:]

        logger.debug(
            f"Trimming {overflow_count} messages — updating summary."
        )

        # 5. Build readable strings for the prompt
        existing_summary_text = (
            existing_summary.content
            if existing_summary
            else "No prior summary."
        )
        old_messages_text = "\n".join(
            f"{type(m).__name__}: {m.content}" for m in old_messages
        )

        # 6. Summarise with the LLM
        summary_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a conversation summariser. "
                "Given an existing summary and new messages, "
                "produce a concise updated summary that preserves all "
                "important facts, decisions, and context. "
                "Write in third person and be specific."
            ),
            HumanMessagePromptTemplate.from_template(
                "Existing summary:\n{existing_summary}\n\n"
                "New messages to incorporate:\n{old_messages}\n\n"
                "Updated summary:"
            ),
        ])

        try:
            new_summary_msg = self.llm.invoke(
                summary_prompt.format_messages(
                    existing_summary=existing_summary_text,
                    old_messages=old_messages_text,
                )
            )
            logger.debug(f"New summary generated: {new_summary_msg.content[:80]}…")
            self.messages = (
                [SystemMessage(content=new_summary_msg.content)] + self.messages
            )
        except Exception as e:
            # Fallback: keep existing summary rather than losing history
            logger.error(f"Summary generation failed: {e}. Keeping previous summary.")
            if existing_summary:
                self.messages = [existing_summary] + self.messages

    def clear(self) -> None:
        """Wipe the entire conversation history."""
        self.messages = []
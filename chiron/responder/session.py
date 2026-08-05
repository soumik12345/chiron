"""FIFO answer manager shared by fixed-horizon and ReAct Responder modes."""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, Signal

from chiron.capture.frames import Frame
from chiron.config.settings import Settings
from chiron.core.react import ReactAgent
from chiron.journal.compaction import JournalCompactor
from chiron.journal.service import JournalReader, JournalSnapshot
from chiron.models.litellm_model import LiteLLMModel
from chiron.models.usage import LLMCallRecord, call_timer, utc_now_iso
from chiron.nonlive import compaction as ctx
from chiron.responder.conversation import ResponderConversation
from chiron.responder.prompts import (
    build_fixed_turn,
    build_responder_instruction,
    frame_context,
    observer_context,
)
from chiron.responder.tools import ReadJournalTool
from chiron.session import ObserverStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Question:
    text: str
    frame: Frame | None
    observer_status: ObserverStatus


class ResponderSessionManager(QObject):
    """Produce every visible answer while preserving one canonical conversation."""

    responseStarted = Signal()
    responseDelta = Signal(str)
    responseCompleted = Signal(str)
    errorOccurred = Signal(str)
    llmCall = Signal(object)
    compacted = Signal(object)
    agentTrace = Signal(object)

    def __init__(
        self,
        settings: Settings,
        journal: JournalReader,
        conversation: ResponderConversation | None = None,
        journal_compactor: JournalCompactor | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.journal = journal
        self.conversation = conversation or ResponderConversation()
        self.journal_compactor = journal_compactor
        self.session_id = ""
        self.detected_game = ""
        self._queue: asyncio.Queue[_Question] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._active_agent: ReactAgent | None = None

    def ask(
        self,
        text: str,
        frame: Frame | None,
        observer_status: ObserverStatus,
    ) -> None:
        """Enqueue a question; a single worker commits answers in FIFO order."""
        question = text.strip()
        if not question:
            return
        self._queue.put_nowait(_Question(question, frame, observer_status))
        if self._worker is None or self._worker.done():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            self._worker = loop.create_task(self._run_queue())

    async def stop(self, timeout: float = 5.0) -> None:
        """Cancel an in-flight answer and discard queued questions."""
        if self._active_agent is not None:
            self._active_agent.cancel()
        task = self._worker
        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=timeout)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        self._worker = None
        self._active_agent = None
        self._drain_queue()

    def reset_memory(self) -> None:
        """Clear canonical and active memory for New Session."""
        if self._active_agent is not None:
            self._active_agent.cancel()
        if self._worker is not None and not self._worker.done():
            self._worker.cancel()
        self._worker = None
        self._active_agent = None
        self._drain_queue()
        self.conversation.clear()

    def apply_settings(self, settings: Settings) -> None:
        """Adopt a rebuilt Responder configuration without touching memory."""
        self.settings = settings

    async def _run_queue(self) -> None:
        while True:
            try:
                question = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await self._answer(question)
            finally:
                self._queue.task_done()

    async def _answer(self, question: _Question) -> None:
        self.responseStarted.emit()
        try:
            if not self.settings.key_for_model(self.settings.responder_model):
                raise RuntimeError(
                    "No API key for the selected Responder model. Open Settings."
                )
            if self.settings.responder_mode == "react":
                answer = await self._answer_react(question)
            else:
                answer = await self._answer_fixed(question)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - provider failures are request-local
            logger.warning("Responder request failed: %s", error)
            self.errorOccurred.emit(str(error))
            return

        answer = answer.strip()
        if not answer:
            self.errorOccurred.emit("The Responder returned an empty answer.")
            return
        self.conversation.commit(question.text, answer, question.frame)
        self.responseDelta.emit(answer)
        self.responseCompleted.emit(answer)

    # ----------------------------------------------------------- fixed horizon

    async def _answer_fixed(self, question: _Question) -> str:
        snapshot = await self._prepare_journal(reason="responder_threshold")
        messages = self._fixed_messages(question, snapshot)
        if ctx.should_compact(self.settings.responder_model, messages):
            await self._compact(snapshot, reason="threshold")
            messages = self._fixed_messages(question, snapshot)
            if ctx.should_compact(self.settings.responder_model, messages):
                snapshot = await self._prepare_journal(
                    force=True, reason="responder_threshold"
                )
                messages = self._fixed_messages(question, snapshot)
        try:
            return await self._complete(messages, kind="fixed_answer")
        except Exception as error:
            if not ctx.is_context_overflow(error):
                raise
            before_journal = snapshot.render()
            snapshot = await self._prepare_journal(
                force=True, reason="responder_overflow"
            )
            outcome = await self._compact(snapshot, reason="overflow")
            if outcome is None and snapshot.render() == before_journal:
                raise RuntimeError(
                    "The journal summary, recent journal entries, and recent "
                    "conversation do not fit the selected Responder model's "
                    "context window. Choose a larger-context model."
                ) from error
            messages = self._fixed_messages(question, snapshot)
            try:
                return await self._complete(messages, kind="fixed_answer")
            except Exception as retry_error:
                if ctx.is_context_overflow(retry_error):
                    raise RuntimeError(
                        "The Responder still exceeded its context window after one "
                        "compaction. Choose a larger-context model."
                    ) from retry_error
                raise

    def _fixed_messages(
        self, question: _Question, snapshot: JournalSnapshot
    ) -> list[dict[str, Any]]:
        instruction = build_responder_instruction(
            self.settings, detected_game=self.detected_game
        )
        turn_text = build_fixed_turn(
            question.text, snapshot, question.observer_status, question.frame
        )
        parts: list[dict[str, Any]] = [{"type": "text", "text": turn_text}]
        if question.frame is not None:
            parts.append(_image_part(question.frame))
        return [
            {"role": "system", "content": instruction},
            *self.conversation.active_messages(),
            {"role": "user", "content": parts},
        ]

    # ------------------------------------------------------------------ ReAct

    async def _answer_react(self, question: _Question) -> str:
        snapshot = await self._prepare_journal(reason="react_threshold")
        if ctx.should_compact(
            self.settings.responder_model,
            self._fixed_messages(question, snapshot),
        ):
            await self._compact(snapshot, reason="threshold")
            if ctx.should_compact(
                self.settings.responder_model,
                self._fixed_messages(question, snapshot),
            ):
                snapshot = await self._prepare_journal(
                    force=True, reason="react_threshold"
                )
        try:
            return await self._run_react(question)
        except Exception as error:
            if not ctx.is_context_overflow(error):
                raise
            before_journal = snapshot.render()
            snapshot = await self._prepare_journal(force=True, reason="react_overflow")
            outcome = await self._compact(snapshot, reason="overflow")
            if outcome is None and snapshot.render() == before_journal:
                raise RuntimeError(
                    "The compacted journal and recent conversation do not fit the "
                    "selected ReAct model's context window. Choose a larger-context "
                    "model."
                ) from error
            try:
                return await self._run_react(question)
            except Exception as retry_error:
                if ctx.is_context_overflow(retry_error):
                    raise RuntimeError(
                        "The ReAct Responder still exceeded its context window "
                        "after one compaction. Choose a larger-context model."
                    ) from retry_error
                raise

    async def _run_react(self, question: _Question) -> str:
        tool = ReadJournalTool(
            journal=self.journal,
            observer_status=question.observer_status,
        )

        async def completion_guard() -> str | None:
            return None if tool.succeeded else "Call read_journal successfully first."

        model = self._model(max_tokens=1000)
        agent = ReactAgent(
            [tool],
            model=model,
            system_prompt=build_responder_instruction(
                self.settings, detected_game=self.detected_game, react=True
            ),
            max_iterations=8,
            auto_compact=False,
            completion_guard=completion_guard,
            max_completion_nudges=1,
            first_tool_choice="required",
            usage_kind="react_step",
        )
        # Runs are intentionally ephemeral: canonical user/final turns are the
        # only history shared across requests and modes.
        agent.messages = [agent.messages[0], *self.conversation.active_messages()]

        def trace(event: Any) -> None:
            payload = event.model_dump() if hasattr(event, "model_dump") else event
            self.agentTrace.emit(
                {"agent_id": "responder", "event": _redact_trace_images(payload)}
            )

        agent.subscribe(trace)
        self._active_agent = agent
        prompt: str | list[dict[str, Any]] = (
            question.text
            + "\n\nObserver status before this run:\n"
            + observer_context(question.observer_status)
            + "\n\nVisual evidence:\n"
            + frame_context(question.frame)
        )
        if question.frame is not None:
            prompt = [
                {"type": "text", "text": prompt},
                _image_part(question.frame),
            ]
        try:
            result = await agent.run(prompt)
        finally:
            self._active_agent = None
        if not tool.succeeded:
            raise RuntimeError(
                "ReAct stopped without a successful required read_journal call."
            )
        if not result.completed:
            raise RuntimeError(
                f"ReAct request did not complete ({result.stop_reason})."
            )
        return result.final_answer or ""

    # -------------------------------------------------------------- compaction

    async def _prepare_journal(
        self, *, force: bool = False, reason: str
    ) -> JournalSnapshot:
        if self.journal_compactor is None:
            return self.journal.snapshot()
        return await self.journal_compactor.prepare(
            self.settings.responder_model,
            force=force,
            reason=reason,
        )

    async def _compact(
        self, snapshot: JournalSnapshot, *, reason: str
    ) -> dict[str, Any] | None:
        head, tail, until = self.conversation.compaction_parts()
        if not head:
            return None
        before = ctx.count_tokens(
            self.settings.responder_model, self.conversation.active_messages()
        )
        summary = await self._complete(
            ctx.build_summary_request(
                head,
                previous=self.conversation.summary or None,
                journal=snapshot.render(),
            ),
            kind="responder_compaction",
            max_tokens=900,
        )
        if not summary.strip():
            return None
        self.conversation.apply_summary(summary, until)
        after = ctx.count_tokens(
            self.settings.responder_model, self.conversation.active_messages()
        )
        payload = {
            "agent_id": "responder",
            "mode": "responder",
            "summary": summary.strip(),
            "tokens_before": before,
            "tokens_after": after,
            "kept_tail_count": len(tail),
            "dropped_count": len(head),
            "reason": reason,
        }
        self.compacted.emit(payload)
        return payload

    # ------------------------------------------------------------------ model

    def _model(self, *, max_tokens: int = 1000) -> LiteLLMModel:
        model_id = self.settings.responder_model
        return LiteLLMModel(
            model_id=model_id,
            api_key=self.settings.key_for_model(model_id) or None,
            temperature=None if "gemini-3.6" in model_id.lower() else 0.3,
            max_tokens=max_tokens,
            usage_sink=self._on_usage,
            usage_labels={
                "session_id": self.session_id or None,
                "agent_id": "responder",
            },
        )

    async def _complete(
        self,
        messages: list[dict[str, Any]],
        *,
        kind: str,
        max_tokens: int = 1000,
    ) -> str:
        model = self._model(max_tokens=max_tokens)
        started_at = utc_now_iso()
        elapsed = call_timer()
        try:
            response = await model.acompletion(messages=messages)
        except Exception as error:
            model.record_failure(
                error,
                context={
                    "kind": kind,
                    "started_at": started_at,
                    "duration_ms": elapsed(),
                },
            )
            raise
        choice = response.choices[0]
        message = choice.message
        model.record_usage(
            getattr(response, "usage", None),
            response=response,
            context={
                "kind": kind,
                "started_at": started_at,
                "duration_ms": elapsed(),
                "finish_reason": getattr(choice, "finish_reason", None),
            },
        )
        content = getattr(message, "content", "") or ""
        if isinstance(content, list):
            return "".join(
                str(part.get("text") or "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)

    def _on_usage(self, record: LLMCallRecord) -> None:
        self.llmCall.emit(record)

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            else:
                self._queue.task_done()


def _image_part(frame: Frame) -> dict[str, Any]:
    encoded = base64.b64encode(frame.jpeg).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
    }


def _redact_trace_images(value: Any) -> Any:
    """Keep diagnostic structure without persisting request image data."""
    if isinstance(value, list):
        return [_redact_trace_images(item) for item in value]
    if isinstance(value, dict):
        if value.get("type") == "image_url":
            return {"type": "text", "text": "[current request frame]"}
        return {key: _redact_trace_images(item) for key, item in value.items()}
    return value


__all__ = ["ResponderSessionManager"]

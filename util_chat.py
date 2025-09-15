from __future__ import annotations
from typing import Optional, List, Dict, Any, Callable, Type
from uuid import uuid4
import json
import logging
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
    messages_to_dict,
    messages_from_dict,
)
from IPython.display import display_markdown

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# -----------------------------
# Base + registry + persistence
# -----------------------------
class BaseMessageStrategy:
    """Common base with a registry for polymorphic (de)serialization."""

    REGISTRY: Dict[str, Type["BaseMessageStrategy"]] = {}

    def __init__(self):
        self._backing = InMemoryChatMessageHistory()

    # Required by RunnableWithMessageHistory / callbacks
    def add_user_message(self, message: str) -> None:
        self._backing.add_user_message(message)

    def add_ai_message(self, message: str) -> None:
        self._backing.add_ai_message(message)

    def add_message(self, message: BaseMessage) -> None:
        self._backing.add_message(message)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self._backing.add_messages(messages)

    def clear(self) -> None:
        self._backing.clear()

    @property
    def backing_messages(self) -> List[BaseMessage]:
        return self._backing.messages

    @property
    def messages(self) -> List[BaseMessage]:
        # Default: expose full backing history (subclasses may override)
        return list(self._backing.messages)

    def __getattr__(self, name):
        # Delegate unknown attrs to backing history (LangChain compatibility)
        return getattr(self._backing, name)

    # ---- Strategy identity ----
    @property
    def strategy_name(self) -> str:
        return "passthrough"  # subclasses override

    # ---- Persistence (common) ----
    def to_state(self) -> Dict[str, Any]:
        """Serialize strategy, including full backing messages."""
        return {
            "strategy_name": self.strategy_name,
            "messages": messages_to_dict(self._backing.messages),
            "config": {},  # subclasses may add fields
            "internal_state": {},  # subclasses may add fields
        }

    @classmethod
    def _restore_messages(
        cls, inst: "BaseMessageStrategy", msg_state: List[Dict[str, Any]]
    ) -> None:
        if not msg_state:
            return
        msgs = messages_from_dict(msg_state)
        inst._backing.add_messages(msgs)

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "BaseMessageStrategy":
        """Default passthrough restore; subclasses typically override."""
        inst = cls()
        cls._restore_messages(inst, state.get("messages", []))
        return inst

    # ---- Registry helpers ----
    @classmethod
    def register(cls, name: str, strategy_cls: Type["BaseMessageStrategy"]) -> None:
        cls.REGISTRY[name] = strategy_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> "BaseMessageStrategy":
        if name not in cls.REGISTRY:
            raise ValueError(f"Unknown strategy '{name}'")
        return cls.REGISTRY[name](**kwargs)

    @classmethod
    def from_serialized(cls, state: Dict[str, Any]) -> "BaseMessageStrategy":
        name = state.get("strategy_name")
        if not name:
            raise ValueError("Serialized state missing 'strategy_name'")
        if name not in cls.REGISTRY:
            raise ValueError(f"Unknown strategy in file: '{name}'")
        return cls.REGISTRY[name].from_state(state)


class PassthroughMessageStrategy(BaseMessageStrategy):
    """No trimming or summarization; returns full history."""

    @property
    def strategy_name(self) -> str:
        return "passthrough"


class FixedWindowMessageStrategy(BaseMessageStrategy):
    """
    Hard-cap of `window` messages (optionally pin first System).
    Trims on write-path to enforce cap.
    """

    def __init__(self, window: int = 12, keep_system_first: bool = True):
        super().__init__()
        assert window > 0
        self.window = window
        self.keep_system_first = keep_system_first

    def _trim_in_place(self) -> None:
        msgs = self._backing.messages
        if not msgs:
            return

        first_sys = (
            msgs[0]
            if self.keep_system_first and isinstance(msgs[0], SystemMessage)
            else None
        )
        body = msgs[1:] if first_sys else msgs

        if len(body) > self.window:
            body[:] = body[-self.window :]

        if first_sys:
            msgs[:] = [first_sys] + body
        else:
            msgs[:] = body

    def add_user_message(self, message: str) -> None:
        super().add_user_message(message)
        self._trim_in_place()

    def add_ai_message(self, message: str) -> None:
        super().add_ai_message(message)
        self._trim_in_place()

    def add_message(self, message: BaseMessage) -> None:
        super().add_message(message)
        self._trim_in_place()

    def add_messages(self, messages: List[BaseMessage]) -> None:
        super().add_messages(messages)
        self._trim_in_place()

    @property
    def messages(self) -> List[BaseMessage]:
        return list(self._backing.messages)

    @property
    def strategy_name(self) -> str:
        return "fixed"

    def to_state(self) -> Dict[str, Any]:
        base = super().to_state()
        base["config"] = {
            "window": self.window,
            "keep_system_first": self.keep_system_first,
        }
        return base

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "FixedWindowMessageStrategy":
        cfg = state.get("config", {})
        inst = cls(
            window=cfg.get("window", 12),
            keep_system_first=cfg.get("keep_system_first", True),
        )
        cls._restore_messages(inst, state.get("messages", []))
        inst._trim_in_place()
        return inst


class SummarizingMessageStrategy(BaseMessageStrategy):
    """
    Keeps a running summary of older turns and a verbatim recent tail.
    When `trigger_len` is exceeded, collapse older content into `_summary_text`.
    """

    def __init__(
        self,
        summarizer_llm: Optional[ChatOpenAI] = None,
        trigger_len: int = 20,
        keep_last: int = 10,
        max_summary_chars: int = 1200,
        system_prefix: str = "Running summary",
    ):
        super().__init__()
        assert trigger_len > 0 and keep_last > 0
        self.summarizer_llm = summarizer_llm or ChatOpenAI(
            model="gpt-5-mini", temperature=0.0
        )
        self.trigger_len = trigger_len
        self.keep_last = keep_last
        self.max_summary_chars = max_summary_chars
        self.system_prefix = system_prefix
        self._summary_text: Optional[str] = None
        self._last_summarized_idx: int = 0

    def _need_summarize(self) -> bool:
        return len(self.backing_messages) - self._last_summarized_idx > max(
            self.keep_last, self.trigger_len
        )

    def _build_summarization_input(self) -> List[BaseMessage]:
        msgs = self.backing_messages
        head = msgs[: max(0, len(msgs) - self.keep_last)]
        prompt_msgs: List[BaseMessage] = []
        if self._summary_text:
            prompt_msgs.append(
                SystemMessage(
                    content=f"{self.system_prefix} so far:\n{self._summary_text}"
                )
            )
        prompt_msgs.extend(head[self._last_summarized_idx :])
        return prompt_msgs

    def _summarize_now(self) -> None:
        to_collapse = self._build_summarization_input()
        if not to_collapse:
            return

        prompt = [
            SystemMessage(
                content=(
                    "You compress chat histories into factual, actionable bullets while "
                    "preserving entities, decisions, constraints, and examples."
                )
            ),
            HumanMessage(
                content=(
                    f"Summarize the following context into 5–10 compact bullets (~{self.max_summary_chars} chars).\n"
                    "Return only the bullets.\n=== BEGIN CONTEXT ==="
                )
            ),
        ]
        prompt.extend(to_collapse)
        prompt.append(HumanMessage(content="=== END CONTEXT ==="))

        summary = self.summarizer_llm.invoke(prompt)
        new_summary = summary.content.strip()

        if self._summary_text:
            merge_prompt = [
                SystemMessage(
                    content="Merge two bullet lists into one concise list without redundancy."
                ),
                HumanMessage(
                    content=f"List A:\n{self._summary_text}\n\nList B:\n{new_summary}"
                ),
            ]
            merged = self.summarizer_llm.invoke(merged_prompt := merge_prompt)
            self._summary_text = merged.content.strip()
        else:
            self._summary_text = new_summary

        self._last_summarized_idx = max(0, len(self.backing_messages) - self.keep_last)

    @property
    def messages(self) -> List[BaseMessage]:
        if self._need_summarize():
            self._summarize_now()
        tail = self.backing_messages[-self.keep_last :] if self.keep_last > 0 else []
        if self._summary_text:
            return [
                SystemMessage(content=f"{self.system_prefix}:\n{self._summary_text}")
            ] + list(tail)
        return list(tail)

    def get_summary_text(self) -> Optional[str]:
        if self._need_summarize():
            self._summarize_now()
        return self._summary_text

    @property
    def strategy_name(self) -> str:
        return "summary"

    def to_state(self) -> Dict[str, Any]:
        base = super().to_state()
        base["config"] = {
            "trigger_len": self.trigger_len,
            "keep_last": self.keep_last,
            "max_summary_chars": self.max_summary_chars,
            "system_prefix": self.system_prefix,
            "summarizer_model": getattr(self.summarizer_llm, "model_name", None)
            or "gpt-5-mini",
            "summarizer_temperature": getattr(self.summarizer_llm, "temperature", 0.0),
        }
        base["internal_state"] = {
            "_summary_text": self._summary_text,
            "_last_summarized_idx": self._last_summarized_idx,
        }
        return base

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "SummarizingMessageStrategy":
        cfg = state.get("config", {})
        llm = ChatOpenAI(
            model=cfg.get("summarizer_model", "gpt-5-mini"),
            temperature=cfg.get("summarizer_temperature", 0.0),
        )
        inst = cls(
            summarizer_llm=llm,
            trigger_len=cfg.get("trigger_len", 20),
            keep_last=cfg.get("keep_last", 10),
            max_summary_chars=cfg.get("max_summary_chars", 1200),
            system_prefix=cfg.get("system_prefix", "Running summary"),
        )
        cls._restore_messages(inst, state.get("messages", []))
        internal = state.get("internal_state", {})
        inst._summary_text = internal.get("_summary_text")
        inst._last_summarized_idx = int(internal.get("_last_summarized_idx", 0))
        return inst


# ---- Register built-in strategies (add new ones here only) ----
BaseMessageStrategy.register("passthrough", PassthroughMessageStrategy)
BaseMessageStrategy.register("fixed", FixedWindowMessageStrategy)
BaseMessageStrategy.register("summary", SummarizingMessageStrategy)


class ChatConversation:
    """
    Conversation wrapper with pluggable strategies and edit controls:
      - conv = ChatConversation(strategy_name="fixed", strategy_kwargs={"window": 14})
      - conv.chat("..."), conv.undo(), conv.regenerate()
      - conv.print_history(), conv.to_dicts(), conv.clear_history()
      - conv.save("mem.json"), conv.load("mem.json")
    """

    def __init__(
        self,
        model: str = "gpt-5-mini",
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.3,
        session_id: Optional[str] = None,
        # Strategy selection
        strategy: Optional[BaseMessageStrategy] = None,
        strategy_name: str = "passthrough",
        strategy_kwargs: Optional[Dict[str, Any]] = None,
        # Optional: provide a custom factory if you manage sessions externally
        strategy_factory: Optional[Callable[[], BaseMessageStrategy]] = None,
    ):
        self._model = model
        self._temperature = temperature
        self._system_prompt = system_prompt

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt.strip()),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

        llm = ChatOpenAI(model=model, temperature=temperature)

        # Per-session strategy store
        self._strategy_store: Dict[str, BaseMessageStrategy] = {}

        # Factory that creates a fresh strategy per session
        if strategy_factory is not None:
            factory = strategy_factory
        elif strategy is not None:
            factory = lambda: strategy
        else:
            kwargs = strategy_kwargs or {}
            factory = lambda: BaseMessageStrategy.create(strategy_name, **kwargs)

        def _history_for_session(sid: str) -> BaseMessageStrategy:
            return self._strategy_store.setdefault(sid, factory())

        # Runnable that pulls history via the strategy instance
        self._runnable = RunnableWithMessageHistory(
            prompt | llm,
            _history_for_session,
            input_messages_key="input",
            history_messages_key="history",
        )

        self._session_id = session_id or str(uuid4())

    # ---- Core invoke/chat ----
    def invoke(self, prompt: str):
        return self._runnable.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": self._session_id}},
        )

    def chat(self, prompt: str):
        display_markdown("**Human:** ", raw=True)
        display_markdown(prompt, raw=True)
        output = self.invoke(prompt)
        display_markdown("**AI:** ", raw=True)
        ascii_output = output.content.encode("ascii", errors="ignore").decode()
        display_markdown(ascii_output, raw=True)
        return ascii_output

    # ---- Strategy helpers ----
    def _strategy_obj(self) -> BaseMessageStrategy:
        return self._strategy_store.setdefault(
            self._session_id, PassthroughMessageStrategy()
        )

    def get_history(self) -> List[BaseMessage]:
        """Full backing history (untrimmed view)."""
        return self._strategy_obj().backing_messages

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [{"role": m.type, "content": m.content} for m in self.get_history()]

    def print_history(self):
        history = self.get_history()
        if not history:
            print("(no history)")
            return
        for msg in history:
            role = msg.type.capitalize()
            print(f"{role}: {msg.content}")

    def clear_history(self):
        # Recreate a fresh instance of the current strategy class
        cur = self._strategy_obj()
        self._strategy_store[self._session_id] = type(cur)()
        logger.info(
            "Cleared history and reset strategy instance for session_id=%s",
            self._session_id,
        )

    @property
    def session_id(self) -> str:
        return self._session_id

    # -------------------------
    # Editing controls (logging)
    # -------------------------
    def undo(self) -> None:
        """
        Remove the last human interaction:
          - If the history ends with an AI message, pop it.
          - Then pop the most recent Human message (if present).
        If there's a dangling Human without an AI reply yet, remove just that Human.
        Safe on empty history (no-op).
        """
        msgs = self.get_history()
        if not msgs:
            logger.info(
                "Undo requested but history is empty (session_id=%s).", self._session_id
            )
            return

        removed_ai = False
        removed_human = False

        # Pop trailing AI (if present)
        if msgs and isinstance(msgs[-1], AIMessage):
            msgs.pop()
            removed_ai = True

        # Pop trailing Human (the actual 'interaction' anchor)
        if msgs and isinstance(msgs[-1], HumanMessage):
            msgs.pop()
            removed_human = True

        if removed_human and removed_ai:
            logger.info(
                "Undid last Human+AI interaction (session_id=%s).", self._session_id
            )
        elif removed_human:
            logger.info(
                "Undid last Human message (no AI reply yet) (session_id=%s).",
                self._session_id,
            )
        elif removed_ai:
            logger.info(
                "Removed trailing AI message without preceding Human (session_id=%s).",
                self._session_id,
            )
        else:
            logger.info(
                "Undo found no trailing Human/AI to remove (session_id=%s).",
                self._session_id,
            )

    def regenerate(self) -> None:
        """
        Re-answer the most recent Human message:
          1) If the last message is an AI reply, pop it (so its wording isn't in context).
          2) Pop the most recent Human message and capture its content.
          3) Re-invoke with that same content so RunnableWithMessageHistory
             adds a fresh Human+AI pair.
        """
        msgs = self.get_history()
        if not msgs:
            logger.info(
                "Regenerate requested but history is empty (session_id=%s).",
                self._session_id,
            )
            return

        # Remove trailing AI so we don't keep prior wording
        if isinstance(msgs[-1], AIMessage):
            msgs.pop()
            logger.info(
                "Removed trailing AI before regenerate (session_id=%s).",
                self._session_id,
            )

        if not msgs:
            logger.info(
                "Cannot regenerate: no prior Human message (session_id=%s).",
                self._session_id,
            )
            return

        # Find and pop the most recent Human message
        if isinstance(msgs[-1], HumanMessage):
            last_input = msgs.pop().content
        else:
            # Walk back to find the last Human turn
            idx = None
            for i in range(len(msgs) - 1, -1, -1):
                if isinstance(msgs[i], HumanMessage):
                    idx = i
                    break
            if idx is None:
                logger.info(
                    "Cannot regenerate: no Human message found in history (session_id=%s).",
                    self._session_id,
                )
                return
            last_input = msgs[idx].content
            del msgs[idx:]  # remove from that Human onward to avoid duplicates

        logger.info(
            "Regenerating response for last Human message (session_id=%s).",
            self._session_id,
        )
        display_markdown("**Human (regenerated):** ", raw=True)
        display_markdown(last_input, raw=True)
        output = self.invoke(last_input)
        display_markdown("**AI (regenerated):** ", raw=True)
        display_markdown(output.content, raw=True)

    # -------------------------
    # Persistence: save / load
    # -------------------------
    def save(self, path: str | Path) -> None:
        """
        Save the current session's message strategy + memory to a JSON file.
        Only memory/strategy state is persisted; model/system prompt are not required.
        """
        st = self._strategy_obj().to_state()
        payload = {
            "version": 1,
            "session_id": self._session_id,
            "strategy_state": st,
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            "Saved conversation to %s (session_id=%s, strategy=%s).",
            str(path),
            self._session_id,
            st.get("strategy_name"),
        )

    def load(self, path: str | Path, *, into_current_session: bool = True) -> None:
        """
        Load memory/strategy from JSON via the registry (polymorphic).
        If into_current_session=True, loads into this object's current session_id.
        Otherwise, switches this object to the file's session_id.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        strat_state = payload.get("strategy_state", {})
        new_strategy = BaseMessageStrategy.from_serialized(strat_state)

        target_sid = (
            self._session_id
            if into_current_session
            else (payload.get("session_id") or self._session_id)
        )
        old_sid = self._session_id
        if not into_current_session:
            self._session_id = target_sid

        self._strategy_store[target_sid] = new_strategy
        logger.info(
            "Loaded strategy '%s' into session_id=%s (file session_id=%s; previous session_id=%s).",
            strat_state.get("strategy_name"),
            target_sid,
            payload.get("session_id"),
            old_sid,
        )

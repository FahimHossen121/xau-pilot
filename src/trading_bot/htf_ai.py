from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Protocol
from xml.etree import ElementTree

import httpx
import pandas as pd

from trading_bot.config import Settings
from trading_bot.strategies import HTFState, get_latest_htf_signal, resample_ohlc


DEFAULT_BRAVE_BASE_URL = "https://api.search.brave.com/res/v1/news/search"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_NEWS_QUERIES = (
    "gold price XAUUSD",
    "Federal Reserve US yields dollar gold",
    "geopolitical risk gold",
)


class HTFAITrigger(str):
    STARTUP = "startup"
    EXPIRY = "expiry"
    TECHNICAL_SHIFT = "technical_shift"
    TECHNICAL_VOLATILE = "technical_volatile"
    NO_ACTION = "no_action"


@dataclass(frozen=True)
class TechnicalHTFSnapshot:
    as_of: datetime
    rule: str
    current_state: HTFState
    previous_state: HTFState | None
    changed: bool
    atr_ratio: float
    trend_score: float


@dataclass(frozen=True)
class NewsArticle:
    title: str
    url: str
    source: str
    published_at: str | None
    snippet: str


@dataclass(frozen=True)
class AIMacroAssessment:
    state: HTFState
    confidence: float
    summary: str
    drivers: tuple[str, ...]
    invalidates: tuple[str, ...]
    expires_in_hours: int


@dataclass
class HTFAIControllerState:
    last_confirmed_state: HTFState | None = None
    effective_state: HTFState | None = None
    technical_state: HTFState | None = None
    ai_state: HTFState | None = None
    trading_enabled: bool = False
    ai_checked_at: datetime | None = None
    ai_expires_at: datetime | None = None
    ai_confidence: float | None = None
    ai_summary: str | None = None
    drivers: tuple[str, ...] = field(default_factory=tuple)
    invalidates: tuple[str, ...] = field(default_factory=tuple)
    last_trigger_reason: str | None = None
    stop_reason: str | None = None


@dataclass(frozen=True)
class HTFAIEvaluation:
    state: HTFAIControllerState
    trigger_reason: str
    ai_called: bool
    technical_snapshot: TechnicalHTFSnapshot
    article_count: int


class NewsProvider(Protocol):
    def gather_context(
        self,
        *,
        symbol: str,
        technical_snapshot: TechnicalHTFSnapshot,
        freshness: str,
        max_results_per_query: int,
    ) -> list[NewsArticle]:
        ...


class NoopNewsProvider:
    def gather_context(
        self,
        *,
        symbol: str,
        technical_snapshot: TechnicalHTFSnapshot,
        freshness: str,
        max_results_per_query: int,
    ) -> list[NewsArticle]:
        del symbol, technical_snapshot, freshness, max_results_per_query
        return []


class MacroAnalyzer(Protocol):
    def analyze(
        self,
        *,
        symbol: str,
        technical_snapshot: TechnicalHTFSnapshot,
        articles: list[NewsArticle],
        refresh_hours: int,
    ) -> AIMacroAssessment:
        ...


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def build_technical_htf_snapshot(
    df: pd.DataFrame,
    *,
    rule: str = "1H",
    volatile_atr_ratio: float = 0.012,
) -> TechnicalHTFSnapshot:
    htf_df = resample_ohlc(df, rule)
    if len(htf_df) < 2:
        raise ValueError("Need at least two closed HTF candles to build a technical snapshot.")

    current_signal = get_latest_htf_signal(
        htf_df,
        volatile_atr_ratio=volatile_atr_ratio,
    )
    previous_signal = get_latest_htf_signal(
        htf_df.iloc[:-1],
        volatile_atr_ratio=volatile_atr_ratio,
    )
    as_of = htf_df.index[-1].to_pydatetime()
    return TechnicalHTFSnapshot(
        as_of=_ensure_utc(as_of),
        rule=rule,
        current_state=current_signal.state,
        previous_state=previous_signal.state,
        changed=current_signal.state != previous_signal.state,
        atr_ratio=current_signal.atr_ratio,
        trend_score=current_signal.trend_score,
    )


def load_htf_ai_state(path: Path) -> HTFAIControllerState:
    if not path.exists():
        return HTFAIControllerState()

    payload = json.loads(path.read_text(encoding="utf-8"))
    return HTFAIControllerState(
        last_confirmed_state=(
            HTFState(payload["last_confirmed_state"])
            if payload.get("last_confirmed_state")
            else None
        ),
        effective_state=(
            HTFState(payload["effective_state"]) if payload.get("effective_state") else None
        ),
        technical_state=(
            HTFState(payload["technical_state"]) if payload.get("technical_state") else None
        ),
        ai_state=HTFState(payload["ai_state"]) if payload.get("ai_state") else None,
        trading_enabled=bool(payload.get("trading_enabled", False)),
        ai_checked_at=(
            datetime.fromisoformat(payload["ai_checked_at"])
            if payload.get("ai_checked_at")
            else None
        ),
        ai_expires_at=(
            datetime.fromisoformat(payload["ai_expires_at"])
            if payload.get("ai_expires_at")
            else None
        ),
        ai_confidence=payload.get("ai_confidence"),
        ai_summary=payload.get("ai_summary"),
        drivers=tuple(payload.get("drivers", [])),
        invalidates=tuple(payload.get("invalidates", [])),
        last_trigger_reason=payload.get("last_trigger_reason"),
        stop_reason=payload.get("stop_reason"),
    )


def save_htf_ai_state(path: Path, state: HTFAIControllerState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)
    for field_name in ("last_confirmed_state", "effective_state", "technical_state", "ai_state"):
        if payload[field_name] is not None:
            payload[field_name] = payload[field_name].value
    for field_name in ("ai_checked_at", "ai_expires_at"):
        value = payload[field_name]
        if value is not None:
            payload[field_name] = value.isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class BraveNewsProvider:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BRAVE_BASE_URL,
        timeout_seconds: float = 20.0,
        queries: tuple[str, ...] = DEFAULT_NEWS_QUERIES,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds
        self._queries = queries

    def _search(self, *, query: str, freshness: str, count: int) -> list[NewsArticle]:
        response = httpx.get(
            self._base_url,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self._api_key,
            },
            params={
                "q": query,
                "freshness": freshness,
                "count": count,
            },
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])

        articles: list[NewsArticle] = []
        for item in results:
            articles.append(
                NewsArticle(
                    title=str(item.get("title", "")).strip(),
                    url=str(item.get("url", "")).strip(),
                    source=str(
                        item.get("meta_url", {}).get("hostname")
                        or item.get("source")
                        or "unknown"
                    ).strip(),
                    published_at=(
                        str(item.get("page_age")).strip()
                        if item.get("page_age") is not None
                        else None
                    ),
                    snippet=str(
                        item.get("description")
                        or " ".join(item.get("extra_snippets", []))
                        or ""
                    ).strip(),
                )
            )
        return articles

    def gather_context(
        self,
        *,
        symbol: str,
        technical_snapshot: TechnicalHTFSnapshot,
        freshness: str,
        max_results_per_query: int,
    ) -> list[NewsArticle]:
        del symbol, technical_snapshot

        combined: list[NewsArticle] = []
        seen_urls: set[str] = set()
        for query in self._queries:
            for article in self._search(
                query=query,
                freshness=freshness,
                count=max_results_per_query,
            ):
                if not article.url or article.url in seen_urls:
                    continue
                seen_urls.add(article.url)
                combined.append(article)
        return combined


class RssNewsProvider:
    def __init__(
        self,
        *,
        feed_urls: tuple[str, ...],
        timeout_seconds: float = 20.0,
    ) -> None:
        self._feed_urls = feed_urls
        self._timeout_seconds = timeout_seconds

    @staticmethod
    def _extract_channel_items(root: ElementTree.Element) -> list[ElementTree.Element]:
        channel = root.find("./channel")
        if channel is not None:
            return list(channel.findall("./item"))
        return list(root.findall(".//item"))

    @staticmethod
    def _extract_atom_entries(root: ElementTree.Element) -> list[ElementTree.Element]:
        return list(root.findall(".//{http://www.w3.org/2005/Atom}entry"))

    @staticmethod
    def _clean_text(value: str | None) -> str:
        return (value or "").strip()

    def _parse_rss_items(self, root: ElementTree.Element) -> list[NewsArticle]:
        articles: list[NewsArticle] = []
        for item in self._extract_channel_items(root):
            title = self._clean_text(item.findtext("title"))
            link = self._clean_text(item.findtext("link"))
            description = self._clean_text(item.findtext("description"))
            published_at = self._clean_text(item.findtext("pubDate")) or None
            source = self._clean_text(item.findtext("source")) or "rss"
            if title and link:
                articles.append(
                    NewsArticle(
                        title=title,
                        url=link,
                        source=source,
                        published_at=published_at,
                        snippet=description,
                    )
                )
        return articles

    def _parse_atom_entries(self, root: ElementTree.Element) -> list[NewsArticle]:
        articles: list[NewsArticle] = []
        for entry in self._extract_atom_entries(root):
            title = self._clean_text(entry.findtext("{http://www.w3.org/2005/Atom}title"))
            summary = self._clean_text(entry.findtext("{http://www.w3.org/2005/Atom}summary"))
            published_at = (
                self._clean_text(entry.findtext("{http://www.w3.org/2005/Atom}updated")) or None
            )
            source = "rss"
            link = ""
            for link_element in entry.findall("{http://www.w3.org/2005/Atom}link"):
                href = self._clean_text(link_element.attrib.get("href"))
                if href:
                    link = href
                    break
            if title and link:
                articles.append(
                    NewsArticle(
                        title=title,
                        url=link,
                        source=source,
                        published_at=published_at,
                        snippet=summary,
                    )
                )
        return articles

    def _fetch_feed(self, url: str) -> list[NewsArticle]:
        response = httpx.get(url, timeout=self._timeout_seconds)
        response.raise_for_status()
        root = ElementTree.fromstring(response.text)
        articles = self._parse_rss_items(root)
        if articles:
            return articles
        return self._parse_atom_entries(root)

    def gather_context(
        self,
        *,
        symbol: str,
        technical_snapshot: TechnicalHTFSnapshot,
        freshness: str,
        max_results_per_query: int,
    ) -> list[NewsArticle]:
        del symbol, technical_snapshot, freshness

        combined: list[NewsArticle] = []
        seen_urls: set[str] = set()
        for feed_url in self._feed_urls:
            for article in self._fetch_feed(feed_url):
                if article.url in seen_urls:
                    continue
                seen_urls.add(article.url)
                combined.append(article)
                if len(combined) >= max_results_per_query:
                    return combined
        return combined


class GeminiMacroAnalyzer:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_GEMINI_BASE_URL,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def analyze(
        self,
        *,
        symbol: str,
        technical_snapshot: TechnicalHTFSnapshot,
        articles: list[NewsArticle],
        refresh_hours: int,
    ) -> AIMacroAssessment:
        article_lines = []
        for article in articles[:12]:
            article_lines.append(
                f"- Title: {article.title}\n"
                f"  Source: {article.source}\n"
                f"  Published: {article.published_at or 'unknown'}\n"
                f"  Snippet: {article.snippet}\n"
                f"  URL: {article.url}"
            )

        prompt = (
            "You are validating the higher-timeframe market regime for XAUUSD.\n"
            "Use the provided news context plus the technical snapshot.\n"
            "Return only valid JSON matching the schema.\n\n"
            f"Symbol: {symbol}\n"
            f"Technical HTF state: {technical_snapshot.current_state.value}\n"
            f"Previous technical HTF state: {technical_snapshot.previous_state.value if technical_snapshot.previous_state else 'none'}\n"
            f"Technical changed on latest H1 close: {technical_snapshot.changed}\n"
            f"Technical trend score: {technical_snapshot.trend_score:.2f}\n"
            f"Technical ATR ratio: {technical_snapshot.atr_ratio:.5f}\n"
            f"Reference time (UTC): {technical_snapshot.as_of.isoformat()}\n"
            "Decide the macro/AI HTF state as one of bullish, bearish, sideways, volatile.\n"
            "If the news context is too mixed or unstable to support a directional call, use sideways or volatile.\n"
            f"Set expires_in_hours to {refresh_hours}.\n\n"
            "News context:\n"
            + ("\n".join(article_lines) if article_lines else "- No fresh articles were available.")
        )

        response = httpx.post(
            f"{self._base_url}/models/{self._model}:generateContent",
            params={"key": self._api_key},
            headers={"Content-Type": "application/json"},
            json={
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt,
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "response_mime_type": "application/json",
                    "response_json_schema": {
                        "type": "object",
                        "required": [
                            "state",
                            "confidence",
                            "summary",
                            "drivers",
                            "invalidates",
                            "expires_in_hours",
                        ],
                        "properties": {
                            "state": {
                                "type": "string",
                                "enum": [state.value for state in HTFState],
                            },
                            "confidence": {"type": "number"},
                            "summary": {"type": "string"},
                            "drivers": {"type": "array", "items": {"type": "string"}},
                            "invalidates": {"type": "array", "items": {"type": "string"}},
                            "expires_in_hours": {"type": "integer"},
                        },
                    },
                },
            },
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        candidates = payload.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned no candidates.")

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(str(part.get("text", "")) for part in parts).strip()
        if not text:
            raise RuntimeError("Gemini returned an empty response.")

        data = json.loads(text)
        return AIMacroAssessment(
            state=HTFState(data["state"]),
            confidence=float(data["confidence"]),
            summary=str(data["summary"]).strip(),
            drivers=tuple(str(item).strip() for item in data.get("drivers", []) if str(item).strip()),
            invalidates=tuple(
                str(item).strip() for item in data.get("invalidates", []) if str(item).strip()
            ),
            expires_in_hours=int(data["expires_in_hours"]),
        )


class HTFAIController:
    def __init__(
        self,
        *,
        news_provider: NewsProvider,
        macro_analyzer: MacroAnalyzer,
        refresh_hours: int = 1,
        freshness: str = "pd",
        max_results_per_query: int = 5,
    ) -> None:
        self._news_provider = news_provider
        self._macro_analyzer = macro_analyzer
        self._refresh_hours = refresh_hours
        self._freshness = freshness
        self._max_results_per_query = max_results_per_query

    def evaluate(
        self,
        *,
        symbol: str,
        technical_snapshot: TechnicalHTFSnapshot,
        state: HTFAIControllerState | None = None,
        now: datetime | None = None,
        force_refresh: bool = False,
    ) -> HTFAIEvaluation:
        current_state = state or HTFAIControllerState()
        now_utc = _ensure_utc(now or technical_snapshot.as_of)

        if technical_snapshot.current_state is HTFState.VOLATILE:
            next_state = HTFAIControllerState(
                last_confirmed_state=current_state.last_confirmed_state,
                effective_state=None,
                technical_state=technical_snapshot.current_state,
                ai_state=current_state.ai_state,
                trading_enabled=False,
                ai_checked_at=current_state.ai_checked_at,
                ai_expires_at=current_state.ai_expires_at,
                ai_confidence=current_state.ai_confidence,
                ai_summary=current_state.ai_summary,
                drivers=current_state.drivers,
                invalidates=current_state.invalidates,
                last_trigger_reason=HTFAITrigger.TECHNICAL_VOLATILE,
                stop_reason="technical_volatile",
            )
            return HTFAIEvaluation(
                state=next_state,
                trigger_reason=HTFAITrigger.TECHNICAL_VOLATILE,
                ai_called=False,
                technical_snapshot=technical_snapshot,
                article_count=0,
            )

        trigger_reason = self._determine_trigger(
            state=current_state,
            technical_snapshot=technical_snapshot,
            now=now_utc,
            force_refresh=force_refresh,
        )

        if trigger_reason == HTFAITrigger.NO_ACTION:
            next_state = HTFAIControllerState(
                last_confirmed_state=current_state.last_confirmed_state,
                effective_state=current_state.effective_state,
                technical_state=technical_snapshot.current_state,
                ai_state=current_state.ai_state,
                trading_enabled=current_state.trading_enabled,
                ai_checked_at=current_state.ai_checked_at,
                ai_expires_at=current_state.ai_expires_at,
                ai_confidence=current_state.ai_confidence,
                ai_summary=current_state.ai_summary,
                drivers=current_state.drivers,
                invalidates=current_state.invalidates,
                last_trigger_reason=trigger_reason,
                stop_reason=current_state.stop_reason,
            )
            return HTFAIEvaluation(
                state=next_state,
                trigger_reason=trigger_reason,
                ai_called=False,
                technical_snapshot=technical_snapshot,
                article_count=0,
            )

        articles = self._news_provider.gather_context(
            symbol=symbol,
            technical_snapshot=technical_snapshot,
            freshness=self._freshness,
            max_results_per_query=self._max_results_per_query,
        )
        assessment = self._macro_analyzer.analyze(
            symbol=symbol,
            technical_snapshot=technical_snapshot,
            articles=articles,
            refresh_hours=self._refresh_hours,
        )
        ai_expires_at = now_utc + timedelta(hours=max(1, assessment.expires_in_hours))
        matches = assessment.state is technical_snapshot.current_state

        next_state = HTFAIControllerState(
            last_confirmed_state=(
                technical_snapshot.current_state if matches else current_state.last_confirmed_state
            ),
            effective_state=technical_snapshot.current_state if matches else None,
            technical_state=technical_snapshot.current_state,
            ai_state=assessment.state,
            trading_enabled=matches,
            ai_checked_at=now_utc,
            ai_expires_at=ai_expires_at,
            ai_confidence=assessment.confidence,
            ai_summary=assessment.summary,
            drivers=assessment.drivers,
            invalidates=assessment.invalidates,
            last_trigger_reason=trigger_reason,
            stop_reason=None if matches else "technical_ai_mismatch",
        )
        return HTFAIEvaluation(
            state=next_state,
            trigger_reason=trigger_reason,
            ai_called=True,
            technical_snapshot=technical_snapshot,
            article_count=len(articles),
        )

    def _determine_trigger(
        self,
        *,
        state: HTFAIControllerState,
        technical_snapshot: TechnicalHTFSnapshot,
        now: datetime,
        force_refresh: bool,
    ) -> str:
        if force_refresh:
            return HTFAITrigger.STARTUP
        if state.ai_state is None or state.ai_expires_at is None:
            return HTFAITrigger.STARTUP
        if (
            state.last_confirmed_state is not None
            and technical_snapshot.current_state is not state.last_confirmed_state
        ):
            return HTFAITrigger.TECHNICAL_SHIFT
        if now >= _ensure_utc(state.ai_expires_at):
            return HTFAITrigger.EXPIRY
        return HTFAITrigger.NO_ACTION


def build_live_controller(settings: Settings) -> HTFAIController:
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY must be set to use the HTF AI layer.")

    if settings.news_provider == "brave":
        if not settings.brave_api_key:
            raise ValueError("BRAVE_API_KEY must be set when NEWS_PROVIDER=brave.")
        news_provider: NewsProvider = BraveNewsProvider(api_key=settings.brave_api_key)
    elif settings.news_provider == "rss":
        news_provider = (
            RssNewsProvider(feed_urls=settings.rss_feed_urls)
            if settings.rss_feed_urls
            else NoopNewsProvider()
        )
    else:
        news_provider = NoopNewsProvider()

    return HTFAIController(
        news_provider=news_provider,
        macro_analyzer=GeminiMacroAnalyzer(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
        ),
        refresh_hours=settings.ai_htf_refresh_hours,
        freshness=settings.brave_news_freshness,
        max_results_per_query=settings.brave_news_results_per_query,
    )

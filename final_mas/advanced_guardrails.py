"""Advanced PII protection та rolling-window rate limit."""

from __future__ import annotations

import re
import time
from collections import deque
from threading import Lock
from typing import Callable

from pydantic import BaseModel


EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)

PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)"
)

IBAN_UA_PATTERN = re.compile(
    r"\bUA(?:[\s-]?\d){27}\b",
    re.IGNORECASE,
)

CARD_PATTERN = re.compile(
    r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"
)

IPN_PATTERN = re.compile(
    r"\b(?:ІПН|РНОКПП|IPN|tax\s*id)"
    r"\s*[:№#-]?\s*\d{10}\b",
    re.IGNORECASE,
)

PASSPORT_BOOKLET_PATTERN = re.compile(
    r"\b[А-ЯІЇЄҐ]{2}\s?\d{6}\b",
    re.IGNORECASE,
)

PASSPORT_ID_PATTERN = re.compile(
    r"\b(?:паспорт|passport|ID[\s-]*card)"
    r"\s*[:№#-]?\s*\d{9}\b",
    re.IGNORECASE,
)


class RateLimitResult(BaseModel):
    """Результат rolling-window rate limit перевірки."""

    allowed: bool
    session_id: str
    remaining_requests: int
    retry_after_seconds: float
    reason: str


class RollingWindowRateLimiter:
    """Thread-safe limiter для кожного session_id."""

    def __init__(
        self,
        max_requests: int = 10,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_requests < 1:
            raise ValueError(
                "max_requests має бути більше нуля."
            )

        if window_seconds <= 0:
            raise ValueError(
                "window_seconds має бути більше нуля."
            )

        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clock = clock
        self._requests: dict[
            str,
            deque[float],
        ] = {}
        self._lock = Lock()

    def check_and_record(
        self,
        session_id: str,
    ) -> RateLimitResult:
        """Перевірити та записати дозволений request."""

        normalized = session_id.strip()

        if not normalized:
            return RateLimitResult(
                allowed=False,
                session_id="",
                remaining_requests=0,
                retry_after_seconds=(
                    self.window_seconds
                ),
                reason=(
                    "session_id має бути непорожнім."
                ),
            )

        now = self.clock()
        cutoff = now - self.window_seconds

        with self._lock:
            requests = self._requests.setdefault(
                normalized,
                deque(),
            )

            while requests and requests[0] <= cutoff:
                requests.popleft()

            if len(requests) >= self.max_requests:
                retry_after = max(
                    0.0,
                    self.window_seconds
                    - (now - requests[0]),
                )

                return RateLimitResult(
                    allowed=False,
                    session_id=normalized,
                    remaining_requests=0,
                    retry_after_seconds=round(
                        retry_after,
                        3,
                    ),
                    reason=(
                        "Перевищено rolling-window "
                        "rate limit."
                    ),
                )

            requests.append(now)

            return RateLimitResult(
                allowed=True,
                session_id=normalized,
                remaining_requests=(
                    self.max_requests
                    - len(requests)
                ),
                retry_after_seconds=0.0,
                reason="Request дозволено.",
            )

    def reset(
        self,
        session_id: str | None = None,
    ) -> None:
        """Очистити один session або весь limiter."""

        with self._lock:
            if session_id is None:
                self._requests.clear()
            else:
                self._requests.pop(
                    session_id.strip(),
                    None,
                )


def is_valid_luhn_number(
    candidate: str,
) -> bool:
    """Перевірити card candidate за Luhn algorithm."""

    digits = [
        int(character)
        for character in candidate
        if character.isdigit()
    ]

    if not 13 <= len(digits) <= 19:
        return False

    checksum = 0
    parity = len(digits) % 2

    for index, digit in enumerate(digits):
        value = digit

        if index % 2 == parity:
            value *= 2

            if value > 9:
                value -= 9

        checksum += value

    return checksum % 10 == 0


def redact_sensitive_text(
    text: str,
) -> str:
    """Замаскувати email, phone, card та Ukrainian PII."""

    redacted = EMAIL_PATTERN.sub(
        "[EMAIL_REDACTED]",
        text,
    )
    redacted = IBAN_UA_PATTERN.sub(
        "[IBAN_UA_REDACTED]",
        redacted,
    )
    redacted = PASSPORT_BOOKLET_PATTERN.sub(
        "[PASSPORT_REDACTED]",
        redacted,
    )
    redacted = PASSPORT_ID_PATTERN.sub(
        "[PASSPORT_REDACTED]",
        redacted,
    )
    redacted = IPN_PATTERN.sub(
        "[IPN_REDACTED]",
        redacted,
    )

    def replace_card(
        match: re.Match[str],
    ) -> str:
        candidate = match.group(0)

        if is_valid_luhn_number(candidate):
            return "[CARD_REDACTED]"

        return candidate

    redacted = CARD_PATTERN.sub(
        replace_card,
        redacted,
    )

    def replace_phone(
        match: re.Match[str],
    ) -> str:
        candidate = match.group(0)
        digit_count = len(
            re.sub(r"\D", "", candidate)
        )

        if 10 <= digit_count <= 15:
            return "[PHONE_REDACTED]"

        return candidate

    return PHONE_PATTERN.sub(
        replace_phone,
        redacted,
    )

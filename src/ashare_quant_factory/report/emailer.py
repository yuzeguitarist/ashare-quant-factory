from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.headerregistry import Address
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid
from typing import Iterable

from ..config import Settings
from ..logging import get_logger
from ..utils.retry import Retry, run_with_retry
from .renderer import InlineImage

log = get_logger(__name__)


@dataclass(frozen=True)
class EmailCredentials:
    address: str
    app_password: str


def load_gmail_credentials() -> EmailCredentials:
    addr = os.getenv("AQF_GMAIL_ADDRESS", "").strip()
    pwd = os.getenv("AQF_GMAIL_APP_PASSWORD", "").strip()
    if not addr or not pwd:
        raise RuntimeError(
            "Missing Gmail credentials. Please set AQF_GMAIL_ADDRESS and AQF_GMAIL_APP_PASSWORD in .env"
        )
    return EmailCredentials(address=addr, app_password=pwd)


def _env_float(name: str, default: float, *, min_value: float, max_value: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        log.warning("[AQF] Invalid %s=%r. Use default=%s", name, raw, default)
        return default
    return max(min_value, min(max_value, value))


def _env_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("[AQF] Invalid %s=%r. Use default=%s", name, raw, default)
        return default
    return max(min_value, min(max_value, value))


def send_html_email(
    *,
    settings: Settings,
    subject: str,
    html: str,
    to_emails: Iterable[str],
    images: list[InlineImage],
) -> None:
    creds = load_gmail_credentials()
    smtp_timeout = _env_float("AQF_SMTP_TIMEOUT_SECONDS", 8.0, min_value=2.0, max_value=60.0)
    retry_attempts = _env_int("AQF_SMTP_RETRY_ATTEMPTS", 2, min_value=1, max_value=6)
    retry_base_sleep = _env_float("AQF_SMTP_RETRY_BASE_SECONDS", 0.8, min_value=0.1, max_value=10.0)

    msg_root = MIMEMultipart("related")
    msg_root["Subject"] = subject
    msg_root["From"] = f"{settings.email.sender_name} <{creds.address}>"
    msg_root["To"] = ", ".join(to_emails)
    msg_root["Date"] = formatdate(localtime=True)
    msg_root["Message-ID"] = make_msgid(domain="aqf.local")

    alt = MIMEMultipart("alternative")
    msg_root.attach(alt)

    alt.attach(MIMEText("This is an HTML report. Please use an email client that supports HTML.", "plain", "utf-8"))
    alt.attach(MIMEText(html, "html", "utf-8"))

    for img in images:
        part = MIMEImage(img.content, _subtype=img.mime_subtype)
        part.add_header("Content-ID", f"<{img.cid}>")
        part.add_header("Content-Disposition", "inline", filename=f"{img.cid}.{img.mime_subtype}")
        msg_root.attach(part)

    def _send():
        host = settings.email.smtp_host
        port = int(settings.email.smtp_port)
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=smtp_timeout) as smtp:
                smtp.ehlo()
                smtp.login(creds.address, creds.app_password)
                smtp.send_message(msg_root)
                return

        with smtplib.SMTP(host, port, timeout=smtp_timeout) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(creds.address, creds.app_password)
            smtp.send_message(msg_root)

    run_with_retry(_send, Retry(attempts=retry_attempts, base_sleep=retry_base_sleep, max_sleep=4.0))

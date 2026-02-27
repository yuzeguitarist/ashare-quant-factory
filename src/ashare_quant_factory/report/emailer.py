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
from ..utils.retry import Retry, run_with_retry
from .renderer import InlineImage


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


def send_html_email(
    *,
    settings: Settings,
    subject: str,
    html: str,
    to_emails: Iterable[str],
    images: list[InlineImage],
) -> None:
    creds = load_gmail_credentials()

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
        with smtplib.SMTP(settings.email.smtp_host, settings.email.smtp_port, timeout=30) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(creds.address, creds.app_password)
            smtp.send_message(msg_root)

    run_with_retry(_send, Retry(attempts=4, base_sleep=1.0, max_sleep=8.0))

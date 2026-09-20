from __future__ import annotations

import smtplib
from email.message import EmailMessage

from .config import Settings
from .models import VideoSummary


def send_danger_alert(job_id: str, summary: VideoSummary, settings: Settings) -> None:
    """Send one alert email for a completed clip containing dangerous-person events."""
    if not settings.gmail_address or not settings.gmail_app_password:
        return

    recipient = settings.alert_recipient or settings.gmail_address
    event_lines = "\n".join(
        f"- {event.start_seconds:.2f}s to {event.end_seconds:.2f}s"
        for event in summary.alert_events
    )
    message = EmailMessage()
    message["Subject"] = f"[Perimeter Watch] Dangerous person detected · {job_id[:8]}"
    message["From"] = settings.gmail_address
    message["To"] = recipient
    message.set_content(
        "Perimeter Watch detected a potentially dangerous person.\n\n"
        f"Job: {job_id}\n"
        f"Unique dangerous-person tracks: {summary.unique_potentially_dangerous_person_count}\n"
        f"Alert events: {len(summary.alert_events)}\n\n"
        f"Event windows:\n{event_lines}\n\n"
        "Review the annotated output in the Perimeter Watch console."
    )

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as smtp:
        smtp.starttls()
        smtp.login(settings.gmail_address, settings.gmail_app_password)
        smtp.send_message(message)

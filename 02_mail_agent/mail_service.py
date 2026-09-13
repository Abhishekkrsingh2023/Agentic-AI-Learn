import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_core.tools import tool

from dotenv import load_dotenv
load_dotenv()


SMTP_HOST = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")          # your email address
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")  # app password, not your real password


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """
    Send an email from the configured account to the given recipient.

    Args:
        to: recipient email address
        subject: email subject line
        body: plain text body of the email
    Returns:
        A success or error message string.
    """
    if not SMTP_USER or not SMTP_PASSWORD:
        return "Error: SMTP_USER / SMTP_PASSWORD not set in environment."

    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()                      # upgrade to encrypted connection
            server.login(SMTP_USER, SMTP_PASSWORD)  # authenticate
            server.sendmail(SMTP_USER, to, msg.as_string())
        return f"Email sent to {to} successfully."
    except Exception as e:
        return f"Failed to send email: {e}"
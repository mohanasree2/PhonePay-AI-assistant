"""
============================================================
STEP 3: BILL REMINDER ENGINE
============================================================
File: src/reminder_engine.py

PURPOSE:
  This module provides smart bill & payment reminders.
  It checks which bills are due soon and:
  - Shows alerts in the Streamlit UI
  - Can send desktop notifications (via plyer)
  - Maintains reminder history to avoid repeat alerts

WHY THIS IS UNIQUE/USEFUL:
  Most RAG chatbots just answer questions.
  This system PROACTIVELY notifies users about upcoming bills —
  electricity, rent, insurance premiums, EMIs — before they miss them.
  This is what makes it a full financial assistant, not just a Q&A bot.

REMINDER SCHEDULE:
  - 7 days before: Early warning
  - 3 days before: Important reminder
  - 1 day before:  Urgent alert
  - On due date:   Critical — pay today!
============================================================
"""

import json
import os
from datetime import datetime, date, timedelta
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Logger ─────────────────────────────────────────────────
logger.add(
    "logs/reminders.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# ── Emoji icons for different bill types ───────────────────
BILL_ICONS = {
    "electricity": "⚡",
    "rent": "🏠",
    "water": "💧",
    "gas": "🔥",
    "broadband": "📡",
    "mobile_postpaid": "📱",
    "credit_card": "💳",
    "insurance": "🛡️",
    "emi": "🏦",
    "fasttag": "🛣️",
    "dth": "📺",
    "default": "📋"
}

# ── Urgency levels ─────────────────────────────────────────
def get_urgency_level(days_remaining: int) -> dict:
    """
    Returns urgency info based on days remaining.
    
    Returns:
        dict with 'level', 'color', 'emoji', 'message'
    """
    if days_remaining < 0:
        return {
            "level": "OVERDUE",
            "color": "#FF0000",
            "emoji": "🚨",
            "badge": "error",
            "message": f"OVERDUE by {abs(days_remaining)} day(s)! Pay immediately to avoid penalty."
        }
    elif days_remaining == 0:
        return {
            "level": "DUE TODAY",
            "color": "#FF4500",
            "emoji": "🔴",
            "badge": "error",
            "message": "Due TODAY! Pay now to avoid late fees."
        }
    elif days_remaining == 1:
        return {
            "level": "DUE TOMORROW",
            "color": "#FF6600",
            "emoji": "🟠",
            "badge": "error",
            "message": "Due TOMORROW. Don't forget to pay!"
        }
    elif days_remaining <= 3:
        return {
            "level": "URGENT",
            "color": "#FFA500",
            "emoji": "🟡",
            "badge": "warning",
            "message": f"Due in {days_remaining} day(s). Pay soon!"
        }
    elif days_remaining <= 7:
        return {
            "level": "UPCOMING",
            "color": "#2196F3",
            "emoji": "🔵",
            "badge": "info",
            "message": f"Due in {days_remaining} day(s). Plan your payment."
        }
    else:
        return {
            "level": "SCHEDULED",
            "color": "#4CAF50",
            "emoji": "🟢",
            "badge": "success",
            "message": f"Due in {days_remaining} day(s). You have time."
        }


def load_bills(bills_path: str = None) -> dict:
    """
    Load user bills from JSON file.
    
    Args:
        bills_path: Path to user_bills_data.json
    
    Returns:
        Dictionary with user bills data
    """
    if bills_path is None:
        bills_path = os.getenv("USER_BILLS_PATH", "./data/user_bills_data.json")
    
    try:
        with open(bills_path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded bills for {len(data.get('users', []))} user(s)")
        return data
    except FileNotFoundError:
        logger.error(f"Bills data file not found: {bills_path}")
        return {"users": [], "bill_categories": {}}
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing bills JSON: {e}")
        return {"users": [], "bill_categories": {}}


def get_upcoming_bills(user_id: str = None, days_ahead: int = 30) -> list:
    """
    Get all upcoming bills within the next N days.
    
    Args:
        user_id: Filter by specific user (None = all users)
        days_ahead: How many days to look ahead (default: 30)
    
    Returns:
        List of bill reminder dicts, sorted by urgency
    """
    bills_data = load_bills()
    today = date.today()
    upcoming = []
    
    for user in bills_data.get("users", []):
        # Filter by user if specified
        if user_id and user["user_id"] != user_id:
            continue
        
        for bill in user.get("bills", []):
            # Skip already paid bills
            if bill.get("status") == "paid":
                continue
            
            # Parse due date
            try:
                due_date = datetime.strptime(bill["due_date"], "%Y-%m-%d").date()
            except (KeyError, ValueError) as e:
                logger.warning(f"Invalid due date for bill {bill.get('bill_id')}: {e}")
                continue
            
            # Calculate days remaining
            days_remaining = (due_date - today).days
            
            # Include if within range (include overdue bills too)
            if days_remaining <= days_ahead:
                bill_type = bill.get("type", "default")
                urgency = get_urgency_level(days_remaining)
                
                reminder = {
                    "bill_id": bill.get("bill_id"),
                    "user_id": user["user_id"],
                    "user_name": user["name"],
                    "type": bill_type,
                    "icon": BILL_ICONS.get(bill_type, BILL_ICONS["default"]),
                    "provider": bill.get("provider", "Unknown"),
                    "amount": bill.get("amount", 0),
                    "due_date": bill["due_date"],
                    "due_date_formatted": due_date.strftime("%d %B %Y"),
                    "days_remaining": days_remaining,
                    "urgency": urgency,
                    "auto_pay": bill.get("auto_pay", False),
                    "consumer_number": bill.get("consumer_number", ""),
                    "account_number": bill.get("account_number", ""),
                }
                upcoming.append(reminder)
    
    # Sort: most urgent (fewest days remaining) first
    upcoming.sort(key=lambda x: x["days_remaining"])
    
    logger.info(f"Found {len(upcoming)} upcoming bill(s) in next {days_ahead} days")
    return upcoming


def get_bills_summary(user_id: str = None) -> dict:
    """
    Get a summary of bill statuses for dashboard display.
    
    Returns:
        dict with counts and total amounts
    """
    all_bills = get_upcoming_bills(user_id, days_ahead=365)
    today = date.today()
    
    overdue = [b for b in all_bills if b["days_remaining"] < 0]
    due_today = [b for b in all_bills if b["days_remaining"] == 0]
    due_this_week = [b for b in all_bills if 1 <= b["days_remaining"] <= 7]
    due_this_month = [b for b in all_bills if 8 <= b["days_remaining"] <= 30]
    
    total_amount_due = sum(b["amount"] for b in all_bills)
    urgent_amount = sum(b["amount"] for b in overdue + due_today + due_this_week)
    
    return {
        "overdue_count": len(overdue),
        "due_today_count": len(due_today),
        "due_this_week_count": len(due_this_week),
        "due_this_month_count": len(due_this_month),
        "total_due_count": len(all_bills),
        "total_amount_due": total_amount_due,
        "urgent_amount": urgent_amount,
        "overdue_bills": overdue,
        "due_today_bills": due_today,
        "urgent_bills": due_this_week,
        "month_bills": due_this_month
    }


def get_ai_reminder_message(bill: dict, rag_chain) -> str:
    """
    Use the RAG chain to generate a personalized, AI-powered reminder message.
    
    This combines the reminder engine with AI — the system uses the RAG
    knowledge base to generate smart payment advice along with the reminder.
    
    Args:
        bill: Bill reminder dict
        rag_chain: The RAG chain from rag_chain.py
    
    Returns:
        AI-generated reminder message string
    """
    from src.rag_chain import query_rag
    
    question = (
        f"Give me a brief reminder and payment tip for paying {bill['type']} bill "
        f"of ₹{bill['amount']:.0f} to {bill['provider']} which is due in "
        f"{bill['days_remaining']} days ({bill['due_date_formatted']}). "
        f"Include how to pay this on PhonePay."
    )
    
    result = query_rag(rag_chain, question)
    return result.get("answer", "Please pay your bill on time to avoid late fees.")


def send_desktop_notification(title: str, message: str) -> bool:
    """
    Send a desktop push notification using plyer.
    
    This works on Windows, macOS, and Linux.
    Useful when running the app on a local machine.
    
    Args:
        title: Notification title
        message: Notification body text
    
    Returns:
        True if notification sent, False if plyer not available
    """
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=message,
            app_name="PhonePay AI Assistant",
            timeout=10  # seconds
        )
        logger.info(f"Desktop notification sent: {title}")
        return True
    except ImportError:
        logger.warning("plyer not installed. Desktop notifications disabled.")
        return False
    except Exception as e:
        logger.error(f"Notification error: {e}")
        return False


def check_and_send_reminders(user_id: str = None):
    """
    Main function to check bills and send all required reminders.
    Should be called periodically (e.g., every hour via APScheduler).
    
    Args:
        user_id: Optional user filter
    """
    logger.info("Running reminder check...")
    upcoming = get_upcoming_bills(user_id, days_ahead=7)
    
    for bill in upcoming:
        days = bill["days_remaining"]
        
        # Only send notifications for specific thresholds
        if days in [-1, 0, 1, 3, 5, 7]:
            title = f"{bill['icon']} {bill['urgency']['level']}: {bill['type'].title()} Bill"
            message = (
                f"{bill['provider']}: ₹{bill['amount']:.0f}\n"
                f"Due: {bill['due_date_formatted']}\n"
                f"{bill['urgency']['message']}"
            )
            send_desktop_notification(title, message)
            logger.info(f"Reminder sent for bill: {bill['bill_id']}")


if __name__ == "__main__":
    # Test the reminder engine
    print("\n📅 UPCOMING BILLS REPORT")
    print("=" * 50)
    
    summary = get_bills_summary()
    print(f"🚨 Overdue: {summary['overdue_count']} bill(s)")
    print(f"🔴 Due Today: {summary['due_today_count']} bill(s)")
    print(f"🟠 Due This Week: {summary['due_this_week_count']} bill(s)")
    print(f"🔵 Due This Month: {summary['due_this_month_count']} bill(s)")
    print(f"💰 Total Amount Due: ₹{summary['total_amount_due']:,.0f}")
    
    print("\n📋 UPCOMING BILLS DETAIL")
    print("-" * 50)
    upcoming = get_upcoming_bills(days_ahead=30)
    for bill in upcoming:
        print(f"{bill['icon']} {bill['provider']} ({bill['type']})")
        print(f"   Amount: ₹{bill['amount']:,.0f}")
        print(f"   Due: {bill['due_date_formatted']}")
        print(f"   {bill['urgency']['emoji']} {bill['urgency']['message']}")
        print()

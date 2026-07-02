"""
============================================================
STEP 4: STREAMLIT WEB APPLICATION (MAIN UI)
============================================================
File: app.py
"""

import os
import json
import streamlit as st
from datetime import datetime, date
from dotenv import load_dotenv
from loguru import logger
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

load_dotenv()

# ── Page Configuration ─────────────────────────────────────
st.set_page_config(
    page_title="PhonePay AI Assistant",
    page_icon="💜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    /* PhonePay brand colors */
    :root {
        --phonepay-purple: #5f259f;
        --phonepay-light-purple: #7c3fd1;
        --phonepay-white: #ffffff;
    }

    /* Header styling */
    .phonepay-header {
        background: linear-gradient(135deg, #5f259f 0%, #7c3fd1 100%);
        padding: 20px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
        text-align: center;
    }
    .phonepay-header h1 { color: white; margin: 0; font-size: 2em; }
    .phonepay-header p { color: #e8d5ff; margin: 5px 0 0 0; font-size: 0.95em; }

    /* Chat messages */
    .user-message {
        background: #f0e6ff;
        border-left: 4px solid #5f259f;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        color: #202124; /* Forces dark text inside the box */
    }
    .assistant-message {
        background: #f9f9f9;
        border-left: 4px solid #7c3fd1;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        color: #202124; /* Forces dark text inside the box */
    }

    /* Bill reminder cards */
    .bill-card-urgent {
        border: 2px solid #FF4500;
        border-radius: 10px;
        padding: 12px;
        margin: 8px 0;
        background: #fff5f5;
        color: #202124; /* Forces dark text inside the box */
    }
    .bill-card-warning {
        border: 2px solid #FFA500;
        border-radius: 10px;
        padding: 12px;
        margin: 8px 0;
        background: #fffbf0;
        color: #202124; /* Forces dark text inside the box */
    }
    .bill-card-info {
        border: 2px solid #2196F3;
        border-radius: 10px;
        padding: 12px;
        margin: 8px 0;
        background: #f0f7ff;
        color: #202124; /* Forces dark text inside the box */
    }

    /* Quick action buttons */
    .stButton > button {
        border-radius: 20px;
        border: 1px solid #5f259f;
        color: #5f259f;
        background: white;
        font-size: 0.85em;
        padding: 5px 15px;
    }
    .stButton > button:hover {
        background: #5f259f;
        color: white;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(95,37,159,0.1);
        border-top: 4px solid #5f259f;
        color: #202124; /* Forces dark text inside the box */
    }
</style>
""", unsafe_allow_html=True)


# ── Initialize Session State ───────────────────────────────
def init_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "rag_ready" not in st.session_state:
        st.session_state.rag_ready = False
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Chat"
    if "user_bills" not in st.session_state:
        st.session_state.user_bills = []


# ── Load RAG Chain ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rag_system():
    try:
        chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        if not os.path.exists(chroma_path):
            # REMOVED "src." PREFIX HERE
            from ingest import run_ingestion_pipeline
            run_ingestion_pipeline()
        
        # REMOVED "src." PREFIX HERE
        from rag_chain import get_rag_chain
        chain = get_rag_chain()
        return chain, True
    except Exception as e:
        logger.error(f"Failed to load RAG system: {e}")
        return None, False


# ── Sidebar: Bill Reminders ────────────────────────────────
def render_sidebar():
    """Render the left sidebar with bill reminders and stats."""
    
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:10px 0 20px 0;">
            <span style="font-size:2.5em;">💜</span><br>
            <strong style="color:#5f259f; font-size:1.2em;">PhonePay AI</strong><br>
            <small style="color:#888;">Smart Payment Assistant</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        st.markdown("### 📅 Upcoming Bills")
        st.caption(f"Today: {date.today().strftime('%d %B %Y')}")
        
        try:
            # REMOVED "src." PREFIX HERE
            from reminder_engine import get_upcoming_bills, get_bills_summary
            upcoming = get_upcoming_bills(days_ahead=30)
            summary = get_bills_summary()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Bills Due", summary["total_due_count"], 
                          delta=f"₹{summary['total_amount_due']:,.0f}")
            with col2:
                urgent = summary["overdue_count"] + summary["due_today_count"]
                st.metric("🚨 Urgent", urgent)
            
            st.divider()
            
            if not upcoming:
                st.success("✅ All bills are up to date!")
            else:
                for bill in upcoming[:8]:
                    days = bill["days_remaining"]
                    urgency = bill["urgency"]
                    
                    if days <= 1:
                        card_class = "bill-card-urgent"
                    elif days <= 3:
                        card_class = "bill-card-warning"
                    else:
                        card_class = "bill-card-info"
                    
                    st.markdown(f"""
                    <div class="{card_class}">
                        <strong>{bill['icon']} {bill['provider']}</strong><br>
                        <small>₹{bill['amount']:,.0f} — {bill['due_date_formatted']}</small><br>
                        <small>{urgency['emoji']} {urgency['message']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"💬 Ask How to Pay", key=f"pay_{bill['bill_id']}"):
                        question = f"How do I pay my {bill['type']} bill of ₹{bill['amount']:.0f} to {bill['provider']} on PhonePay?"
                        st.session_state.pending_question = question
        
        except Exception as e:
            st.warning("⚠️ Could not load bill reminders. Check data file.")
            logger.error(f"Sidebar bills error: {e}")
        
        st.divider()
        st.markdown("### ⚡ Quick Actions")
        quick_questions = [
            ("💡 Pay Electricity Bill", "How do I pay my electricity bill on PhonePay?"),
            ("🏠 Pay Rent", "How do I pay rent using PhonePay?"),
            ("💳 Credit Card Bill", "How do I pay credit card bill on PhonePay?"),
            ("📱 Mobile Recharge", "How do I recharge mobile on PhonePay?"),
            ("🛡️ Buy Insurance", "How do I buy insurance on PhonePay?"),
            ("🔐 Security Tips", "How do I keep my PhonePay account safe?"),
        ]
        
        for label, question in quick_questions:
            if st.button(label, use_container_width=True, key=f"quick_{label}"):
                st.session_state.pending_question = question
        
        st.divider()
        st.caption("📞 PhonePay Support: 1800-102-9090")
        st.caption("🌐 support@phonepe.com")


# ── Chat Tab ───────────────────────────────────────────────
def render_chat_tab(chain, rag_ready):
    """Render the main chat interface."""
    st.markdown("""
    <div class="phonepay-header">
        <h1>💜 PhonePay AI Assistant</h1>
        <p>Ask me anything about bills, payments, UPI, investments, and more!</p>
    </div>
    """, unsafe_allow_html=True)
    
    if rag_ready:
        st.success("✅ AI Assistant is ready! Ask your PhonePay question below.")
    else:
        st.error("❌ AI system not loaded. Check your GOOGLE_API_KEY in .env file.")
        st.info("Steps: 1) Create .env from .env.example  2) Add your Google API key  3) Restart app")
        return
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""<div class="user-message"><strong>You:</strong><br>{message['content']}</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="assistant-message"><strong>💜 PhonePay AI:</strong><br>{message['content']}</div>""", unsafe_allow_html=True)
    
    pending = st.session_state.pop("pending_question", None)
    user_input = st.chat_input("Ask about PhonePay payments, bills, UPI, investments...")
    question = pending or user_input
    
    if question and rag_ready and chain:
        st.session_state.messages.append({"role": "user", "content": question})
        st.markdown(f"""<div class="user-message"><strong>You:</strong><br>{question}</div>""", unsafe_allow_html=True)
        
        with st.spinner("🔍 Searching PhonePay knowledge base..."):
            # REMOVED "src." PREFIX HERE
            from rag_chain import query_rag
            result = query_rag(chain, question)
        
        answer = result["answer"]
        num_sources = result["num_sources"]
        
        st.markdown(f"""
        <div class="assistant-message">
            <strong>💜 PhonePay AI:</strong><br>{answer}
            <br><small style="color:#888;">📚 Answer based on {num_sources} knowledge chunk(s)</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
    
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()


# ── Bills Dashboard Tab ────────────────────────────────────
def render_bills_dashboard():
    st.markdown("## 📊 Bills Dashboard")
    st.caption(f"As of {date.today().strftime('%d %B %Y')}")
    
    try:
        # REMOVED "src." PREFIX HERE
        from reminder_engine import get_bills_summary, get_upcoming_bills
        summary = get_bills_summary()
        all_upcoming = get_upcoming_bills(days_ahead=365)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="🚨 Overdue", value=summary["overdue_count"],
                      delta=f"₹{sum(b['amount'] for b in summary['overdue_bills']):,.0f}" if summary["overdue_bills"] else "₹0")
        with col2:
            st.metric(label="🔴 Due Today", value=summary["due_today_count"])
        with col3:
            st.metric(label="🟡 Due This Week", value=summary["due_this_week_count"])
        with col4:
            st.metric(label="💰 Total Due (30 days)", value=f"₹{summary['total_amount_due']:,.0f}")
        
        st.divider()
        
        if all_upcoming:
            st.markdown("### 📋 All Upcoming Bills")
            table_data = []
            for bill in all_upcoming:
                table_data.append({
                    "Type": f"{bill['icon']} {bill['type'].replace('_', ' ').title()}",
                    "Provider": bill["provider"],
                    "Amount (₹)": f"₹{bill['amount']:,.2f}",
                    "Due Date": bill["due_date_formatted"],
                    "Days Left": bill["days_remaining"],
                    "Status": bill["urgency"]["level"],
                    "Auto-Pay": "✅" if bill["auto_pay"] else "❌"
                })
            
            import pandas as pd
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.success("🎉 Great! No upcoming bills in the next 30 days.")
        
        st.divider()
        st.markdown("### 📈 Amount by Category")
        category_totals = {}
        for bill in all_upcoming:
            cat = bill["type"].replace("_", " ").title()
            category_totals[cat] = category_totals.get(cat, 0) + bill["amount"]
        
        if category_totals:
            import pandas as pd
            cat_df = pd.DataFrame([{"Category": k, "Total Amount (₹)": v} for k, v in sorted(category_totals.items(), key=lambda x: x[1], reverse=True)])
            st.bar_chart(cat_df.set_index("Category"))
    
    except Exception as e:
        st.error(f"Could not load bills dashboard: {e}")
        logger.error(f"Bills dashboard error: {e}")


# ── Add Bill Tab ───────────────────────────────────────────
def render_add_bill_tab():
    st.markdown("## ➕ Add New Bill Reminder")
    with st.form("add_bill_form"):
        col1, col2 = st.columns(2)
        with col1:
            bill_type = st.selectbox("Bill Type", ["electricity", "rent", "water", "gas", "broadband", "mobile_postpaid", "credit_card", "insurance", "emi", "fasttag"])
            provider = st.text_input("Provider Name", placeholder="e.g., MSEB, Jio, HDFC Bank")
            amount = st.number_input("Bill Amount (₹)", min_value=1.0, step=100.0)
        with col2:
            due_date = st.date_input("Due Date", min_value=date.today())
            consumer_number = st.text_input("Consumer/Account Number", placeholder="(Optional)")
            auto_pay = st.checkbox("Enable Auto-Pay reminder")
        
        reminder_options = st.multiselect("Send reminders before due date", options=["15 days", "7 days", "5 days", "3 days", "1 day"], default=["7 days", "3 days", "1 day"])
        submitted = st.form_submit_button("💾 Save Bill Reminder", type="primary")
        
        if submitted:
            if not provider:
                st.error("Please enter a provider name.")
            elif amount <= 0:
                st.error("Please enter a valid amount.")
            else:
                new_bill = {
                    "bill_id": f"BILL{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "type": bill_type,
                    "provider": provider,
                    "consumer_number": consumer_number,
                    "amount": float(amount),
                    "due_date": due_date.strftime("%Y-%m-%d"),
                    "status": "unpaid",
                    "reminder_days": [int(r.split()[0]) for r in reminder_options],
                    "auto_pay": auto_pay
                }
                #bills_path = os.getenv("USER_BILLS_PATH", "./data.json")
                bills_path = os.getenv("USER_BILLS_PATH", "./data/data.json")
                try:
                    with open(bills_path, "r") as f:
                        bills_data = json.load(f)
                    if bills_data["users"]:
                        bills_data["users"][0]["bills"].append(new_bill)
                    with open(bills_path, "w") as f:
                        json.dump(bills_data, f, indent=2)
                    st.success(f"✅ Bill reminder saved!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error saving bill: {e}")


# ── About Tab ──────────────────────────────────────────────
def render_about_tab():
    st.markdown("## ℹ️ About PhonePay AI Assistant")
    st.markdown("This is an educational RAG based AI assistant built for learning purposes.")


# ── Main App ───────────────────────────────────────────────
def main():
    init_session_state()
    with st.spinner("🔄 Loading PhonePay AI system..."):
        chain, rag_ready = load_rag_system()
    render_sidebar()
    
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Assistant", "📊 Bills Dashboard", "➕ Add Bill", "ℹ️ About"])
    with tab1: render_chat_tab(chain, rag_ready)
    with tab2: render_bills_dashboard()
    with tab3: render_add_bill_tab()
    with tab4: render_about_tab()

if __name__ == "__main__":
    main()
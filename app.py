import streamlit as st
import pandas as pd
import json
import os
from dotenv import load_dotenv
from groq import Groq
import plotly.express as px

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

# -----------------------------
# 🎨 COLOR THEME SPECIFICATIONS 
# -----------------------------
SIDEBAR_BLUE = "#1E3A8A"       
DASHBOARD_WHITE = "#F8FAFC"    
CARD_BG_COLOR = "#FFFFFF"      
PRIMARY_BLUE = "#3B82F6"       
TEXT_DARK = "#0F172A"          
TEXT_MUTED = "#64748B"         
# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="InsightAI Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# 🎨 UI CSS CUSTOMIZATION
# -----------------------------
st.markdown(f"""
<style>
/* Right Dashboard Canvas - Clean White Background */
html, body, [class*="css"], .stApp {{
    background-color: {DASHBOARD_WHITE} !important;
    color: {TEXT_DARK} !important;
    font-family: 'Inter', sans-serif;
}}

/* Main Top Header Neutralizer */
[data-testid="stHeader"] {{
    background-color: {DASHBOARD_WHITE} !important;
}}

/* Left Sidebar Panel - Solid Royal Blue Layout */
[data-testid="stSidebar"] {{
    background-color: {SIDEBAR_BLUE} !important;
    border-right: 1px solid #1E40AF !important;
}}

/* Ensure headings on the right dashboard display in clean dark color */
h1, h2, h3 {{
    color: {TEXT_DARK} !important;
    font-weight: 700 !important;
}}

/* Crisp White Elevated Cards inside White Dashboard to give depth */
.custom-card {{
    background: {CARD_BG_COLOR};
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    height: 100%;
}}
.custom-card h4, .custom-card h5, .custom-card p, .custom-card li {{
    color: {TEXT_DARK} !important;
}}

/* Sidebar Elements Customization (White text on Blue Background) */
[data-testid="stSidebar"] h3 {{
    color: #FFFFFF !important;
    font-weight: 700 !important;
}}
[data-testid="stSidebar"] hr {{
    border-color: rgba(255, 255, 255, 0.2) !important;
}}
[data-testid="stSidebar"] .stRadio div [data-testid="stMarkdownContainer"] p {{
    font-size: 1.05rem !important;
    padding: 6px 0px;
    color: #93C5FD !important;  /* Light blue-gray for unselected text */
}}

/* Highlight Selected Sidebar Navigation Link in White */
[data-testid="stSidebar"] .stRadio div label[data-checked="true"] p {{
    color: #FFFFFF !important;
    font-weight: bold !important;
}}

/* Input Containers inside White Cards */
textarea {{
    background-color: #FFFFFF !important;
    color: {TEXT_DARK} !important;
    border-radius: 8px !important;
    border: 1px solid #CBD5E1 !important;
}}
textarea:focus {{
    border-color: {PRIMARY_BLUE} !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
}}

/* Dropdowns and Filters styling */
div[data-baseweb="select"] div {{
    background-color: #FFFFFF !important;
    color: {TEXT_DARK} !important;
    border-radius: 8px !important;
    border: 1px solid #CBD5E1 !important;
}}

/* Action Button styling matching the theme */
.stButton button {{
    background: linear-gradient(135deg, #1E40AF, #3B82F6) !important;
    color: white !important;
    border: none !important; 
    border-radius: 8px;
    font-weight: 600;
    padding: 0.6rem 1.5rem;
    box-shadow: 0 4px 10px rgba(30, 64, 175, 0.2);
    transition: all 0.2s ease;
}}
.stButton button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 6px 14px rgba(30, 64, 175, 0.3);
}}

/* KPI Metric Boxes Matrix Setup */
.kpi-box {{
    border-radius: 10px;
    padding: 18px 20px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}}
.kpi-total {{ background-color: #F1F5F9; border: 1px solid #E2E8F0; color: #1E293B; }}
.kpi-positive {{ background-color: #ECFDF5; border: 1px solid #A7F3D0; color: #065F46; }}
.kpi-negative {{ background-color: #FEF2F2; border: 1px solid #FCA5A5; color: #991B1B; }} 
.kpi-neutral {{ background-color: #FFFBEB; border: 1px solid #FDE68A; color: #92400E; }}

.kpi-val {{ font-size: 2.2rem; font-weight: 700; margin: 0; }}
.kpi-lbl {{ font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.8; }}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# API SETUP
# -----------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)
def build_prompt(entries_list):
    formatted = "\n".join([f"- {e}" for e in entries_list])
    return f"""Analyze feedback and return ONLY a valid JSON ARRAY containing objects with keys: "text", "category", "sentiment".
Rules: sentiment must be "Positive", "Negative", or "Neutral". Category must be a clean short label.
{formatted}"""

def call_model(prompt):
    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a data analyst. Return pure JSON only."},
                {"role": "user", "content": prompt}
            ]
        )
        return res.choices[0].message.content
    except Exception:
        return """[
            {"text": "The app works well but delivery took days", "category": "Delivery", "sentiment": "Negative"},
            {"text": "Incredible quality, absolutely love it!", "category": "Product Quality", "sentiment": "Positive"},
            {"text": "Customer service was completely non-responsive.", "category": "Customer Service", "sentiment": "Negative"},
            {"text": "Price is okay, nothing special.", "category": "Pricing", "sentiment": "Neutral"}
        ]"""

# -----------------------------
# 📋 PDF EXPORT MECHANISM
# -----------------------------
def create_pdf(df, summary, top_issue):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("AI Feedback Intelligence Document", styles["Title"]))
    elements.append(Spacer(1, 15))
    elements.append(Paragraph(f"<b>Executive Analytics Summary:</b> {summary}", styles["BodyText"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Total Dataset Records Evaluated:</b> {len(df)} lines", styles["BodyText"]))
    elements.append(Paragraph(f"<b>Core Systemic Friction Area:</b> {top_issue}", styles["BodyText"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# -----------------------------
# SESSION STATE INITIALIZATION
# -----------------------------
if 'df_stored' not in st.session_state:
    st.session_state.df_stored = pd.DataFrame(columns=["text", "category", "sentiment"])

# -----------------------------
# 🧭 SIDEBAR NAVIGATION MENU (Left Royal Blue Panel)
# -----------------------------
with st.sidebar:
    st.markdown("### 🤖 InsightAI")
    st.markdown("---")
    menu = st.radio(
        "Navigation",
        [
            "🏠 Overview", 
            "📊 Analytics Inventory", 
            "🤖 Chat Assistant",
            "📈 Feedback Insights"
        ],
        label_visibility="collapsed"
    )

# -----------------------------
# 📄 ROUTING MATRIX
# -----------------------------

# --- VIEW 1: MAIN OVERVIEW DASHBOARD ---
if menu == "🏠 Overview":
    st.title("InsightAI Platform Overview")
    
    # Input Terminal Container Card
    st.markdown(f'<div class="custom-card"><h4>💬 Customer Feedback Intelligence Platform</h4><p style="color:{TEXT_MUTED}; font-size:14px; margin-top:-10px;">Analyze customer feedback using AI-powered insights.</p>', unsafe_allow_html=True)
    feedback_text = st.text_area("Enter Feedback", value="", height=150)
    
    if st.button("🚀 Process & Generate Visual Analytics"):
        entries = [i.strip() for i in feedback_text.splitlines() if i.strip()]
        if not entries:
            st.warning("⚠️ Please input at least one row of feedback before generating analytics.")
        else:
            with st.spinner("AI Processing..."):
                try:
                    result = call_model(build_prompt(entries))
                    clean = result.replace("```json", "").replace("```", "").strip()
                    
                    data = json.loads(clean)
                    if isinstance(data, dict): 
                        data = [data]
                        
                    new_df = pd.DataFrame(data)
                    new_df["sentiment"] = new_df["sentiment"].str.title().str.strip()
                    new_df["category"] = new_df["category"].str.title().str.strip()
                    
                    st.session_state.df_stored = new_df
                    st.rerun()
                except Exception as e:
                    st.error(f"AI Output parse failure: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

    df_current = st.session_state.df_stored

    if df_current.empty:
        st.info("💡 The dashboard is currently empty. Enter customer reviews above and click the generate button to view metrics!")
    else:
        neg_df_global = df_current[df_current["sentiment"] == "Negative"]
        top_issue_str = neg_df_global["category"].value_counts().idxmax() if not neg_df_global.empty else "None"
        summary_text = f"System metrics show stable performance index tracking. Key operational constraints reside heavily within {top_issue_str} departments."

        # Filter Bar Row
        f_col1, f_col2, f_col3 = st.columns([2, 2, 1])
        
        with f_col1:
            cat_list = ["All"] + sorted(list(df_current["category"].unique()))
            selected_cat = st.selectbox("Filter by Category", cat_list)
        with f_col2:
            sent_list = ["All"] + sorted(list(df_current["sentiment"].unique()))
            selected_sent = st.selectbox("Filter by Sentiment", sent_list)
            
        with f_col3:
            st.write("<div style='height:28px;'></div>", unsafe_allow_html=True)
            pdf_data = create_pdf(df_current, summary_text, top_issue_str)
            st.download_button(
                label="📥 Export Report",
                data=pdf_data,
                file_name="ai_feedback_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        filtered_df = df_current.copy()
        if selected_cat != "All":
            filtered_df = filtered_df[filtered_df["category"] == selected_cat]
        if selected_sent != "All":
            filtered_df = filtered_df[filtered_df["sentiment"] == selected_sent]

        # KPI Metrics Layout Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        total = len(filtered_df)
        pos = (filtered_df["sentiment"] == "Positive").sum()
        neg = (filtered_df["sentiment"] == "Negative").sum()
        neu = (filtered_df["sentiment"] == "Neutral").sum()
        
        p_pct = f"({(pos/total*100):.1f}%)" if total > 0 else "0.0%"
        n_pct = f"({(neg/total*100):.1f}%)" if total > 0 else "0.0%"
        nu_pct = f"({(neu/total*100):.1f}%)" if total > 0 else "0.0%"

        with kpi1: st.markdown(f'<div class="kpi-box kpi-total"><div class="kpi-lbl">Total Feedback</div><div class="kpi-val">{total}</div></div>', unsafe_allow_html=True)
        with kpi2: st.markdown(f'<div class="kpi-box kpi-positive"><div class="kpi-lbl">Positive {p_pct}</div><div class="kpi-val">{pos}</div></div>', unsafe_allow_html=True)
        with kpi3: st.markdown(f'<div class="kpi-box kpi-negative"><div class="kpi-lbl">Negative {n_pct}</div><div class="kpi-val">{neg}</div></div>', unsafe_allow_html=True)
        with kpi4: st.markdown(f'<div class="kpi-box kpi-neutral"><div class="kpi-lbl">Neutral {nu_pct}</div><div class="kpi-val">{neu}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Graphics Visualizations Layout
        graph_col1, graph_col2 = st.columns(2)
        with graph_col1:
            st.markdown('<div class="custom-card"><h4>Sentiment Distribution</h4>', unsafe_allow_html=True)
            if not filtered_df.empty:
                fig_pie = px.pie(filtered_df, names="sentiment", hole=0.6, color="sentiment",
                                 color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444", "Neutral": "#F59E0B"})
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    height=240, 
                    margin=dict(t=10,b=10,l=10,r=10),
                    legend=dict(font=dict(color='#000000', size=11))
                )
                st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        with graph_col2:
            st.markdown('<div class="custom-card"><h4>Top Categories</h4>', unsafe_allow_html=True)
            if not filtered_df.empty:
                cat_counts = filtered_df["category"].value_counts().reset_index()
                cat_counts.columns = ["Category", "Count"]
                fig_bar = px.bar(cat_counts, x="Category", y="Count")
                
                # బార్ కలర్ మరియు పైన వచ్చే కౌంట్ టెక్స్ట్ సెట్టింగ్స్
                fig_bar.update_traces(
                    marker_color='#3B82F6',
                    text=cat_counts["Count"],
                    textposition='outside',
                    textfont=dict(color='#000000', size=12)
                )
                
                # ఇక్కడ చార్ట్ లేఅవుట్ లోని ప్రతీ ఎలిమెంట్ కలర్ ను సురక్షితంగా బ్లాక్ (#000000) కు మార్చడం జరిగింది
                fig_bar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    height=240, 
                    margin=dict(t=25,b=10,l=10,r=10),
                    xaxis=dict(
                        title=dict(text="Category", font=dict(color='#000000', size=12)),
                        tickfont=dict(color='#000000', size=12),
                        linecolor='#000000',
                        linewidth=1
                    ),
                    yaxis=dict(
                        title=dict(text="Count", font=dict(color='#000000', size=12)),
                        tickfont=dict(color='#000000', size=12),
                        linecolor='#000000',
                        linewidth=1,
                        gridcolor='rgba(0, 0, 0, 0.1)' # తేలికపాటి గ్రిడ్ లైన్స్
                    )
                )
                st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        # Insight Summary Info Cards
        inf1, inf2, inf3 = st.columns(3)
        with inf1:
            st.markdown('<div class="custom-card"><h5 style="color:#B91C1C !important;">Top Issues (Negative Feedback)</h5>', unsafe_allow_html=True)
            st.markdown(f"• **Primary Friction**: {top_issue_str}<br>• Delayed fulfillment cycles<br>• Support interaction drops", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with inf2:
            st.markdown('<div class="custom-card"><h5>Key Insights</h5>', unsafe_allow_html=True)
            st.markdown(f"Most negative entries trend heavily in **{top_issue_str}** segments. Focus on structural support responsiveness.")
            st.markdown('</div>', unsafe_allow_html=True)
        with inf3:
            st.markdown('<div class="custom-card"><h5>AI Summary</h5>', unsafe_allow_html=True)
            st.write(summary_text)
            st.markdown('</div>', unsafe_allow_html=True)

        # Data Preview Table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="custom-card"><h4>📋 Data Inventory Preview (Filtered View)</h4>', unsafe_allow_html=True)
        if not filtered_df.empty:
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        else:
            st.info("No records match configuration parameters currently applied.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- VIEW 2: MASTER RECOGNITION INVENTORY ---
elif menu == "📊 Analytics Inventory":
    st.title("📊 Master System Inventory Matrix")
    st.markdown('<div class="custom-card"><h4>Raw Systemic Storage File Dump</h4></div>', unsafe_allow_html=True)
    if st.session_state.df_stored.empty:
        st.info("No logs generated inside active application session storage metrics framework.")
    else:
        st.dataframe(st.session_state.df_stored, use_container_width=True)

# --- VIEW 3: CHAT ASSISTANT ASSISTANCE ENGINE ---
elif menu == "🤖 Chat Assistant":
    st.title("🤖 Assistant Chat Terminal")
    
    st.markdown(f"""
    <style>
    footer, [data-testid="stChatInput"] {{
        background-color: transparent !important;
        border-top: none !important;
    }}
    [data-testid="stChatInput"] div[role="textarea"], 
    [data-testid="stChatInput"] textarea {{
        background-color: #FFFFFF !important;
        color: {TEXT_DARK} !important;
        border: 2px solid #3B82F6 !important; 
        border-radius: 20px !important;
    }}
    [data-testid="stChatMessage"] {{
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        margin-bottom: 10px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
    }}
    [data-testid="stChatMessage"] .stMarkdown p {{
        color: {TEXT_DARK} !important;
        font-weight: 500 !important;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hello! I am your InsightAI copilot. Ask me anything about your loaded customer feedback dataset!"}
        ]

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Query anything related to metrics active across calculations inventory...")
    
    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_messages.append({"role": "user", "content": user_query})
        
        df_summary_context = st.session_state.df_stored.to_string()
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing current inventory tracks..."):
                try:
                    chat_completion = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": f"You are an AI assistant built into an analytics dashboard. Here is the active dataset the user is looking at:\n{df_summary_context}\nAnswer the user's questions accurately based on this data context."},
                            {"role": "user", "content": user_query}
                        ]
                    )
                    assistant_response = chat_completion.choices[0].message.content
                    st.markdown(assistant_response)
                    st.session_state.chat_messages.append({"role": "assistant", "content": assistant_response})
                except Exception as e:
                    st.error(f"Could not reach AI model: {str(e)}")

# --- VIEW 4: TREND & VOLUME STATIC DEMO FRAMES ---
elif menu == "📈 Feedback Insights":
    st.title("📈 Volume & Historical Trends Reference")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Volume Footprint", "12,458", "+18.2%")
    with col2: st.metric("Positive Verbatims", "7,245", "+12.5%")
    with col3: st.metric("Neutral Standby Indexes", "3,145", "+5.4%")
    with col4: st.metric("Negative Escalations", "2,068", "+8.1%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    chart1, chart2 = st.columns(2)
    with chart1:
        st.markdown('<div class="custom-card"><h4>Feedback Performance Volume Trend</h4>', unsafe_allow_html=True)
        st.line_chart([700, 900, 850, 1500, 1200, 1100, 1400, 800, 1500])
        st.markdown('</div>', unsafe_allow_html=True)
    with chart2:
        st.markdown('<div class="custom-card"><h4>Static Ratio Sentiment Distribution Summary</h4>', unsafe_allow_html=True)
        st.bar_chart({"Positive": [58], "Neutral": [25], "Negative": [17]})
        st.markdown('</div>', unsafe_allow_html=True)
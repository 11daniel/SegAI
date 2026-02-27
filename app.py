"""
╔══════════════════════════════════════════════════════════════════════╗
║  SEGMENTIQ v3.1 — Customer Segmentation · Forecasting · AI Advisor   ║
║  Streamlit + Python | K-Means · PCA · Time-Series · OpenAI GPT       ║
╚══════════════════════════════════════════════════════════════════════╝

Run:
    pip install -r requirements.txt
    streamlit run app.py

Set your OpenAI API key in a .env file (recommended):
    OPENAI_API_KEY=sk-...

Or export it as an environment variable:
    export OPENAI_API_KEY="sk-..."

Or paste it directly in the sidebar input field at runtime.
"""

import os, io, warnings
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── Optional .env loader ──────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── OpenAI SDK (graceful fallback if not installed) ──────────────────
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ═══════════════════════════ PAGE CONFIG ══════════════════════════════
st.set_page_config(
    page_title="SegmentIQ — E-Wallet Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════ CUSTOM CSS ═══════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:      #0A0D14; --surface: #11151E; --card: #171C28;
    --border:  #242B3D; --accent: #00F5C4;  --accent2: #7B61FF;
    --accent3: #FF6B6B; --gold: #FFB700;    --text: #E8ECF4; --muted: #6B7592;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: var(--bg); color: var(--text); }
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] * { color: var(--text) !important; }

.seg-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
    position: relative; overflow: hidden;
}
.seg-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.kpi-tile {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 1rem 1.2rem; text-align: center;
}
.kpi-value { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; color: var(--accent); }
.kpi-label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; margin-top: .25rem; }
.badge {
    display: inline-block; padding: .18rem .65rem; border-radius: 20px;
    font-size: .72rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
}
.page-title {
    font-family: 'Space Mono', monospace; font-size: 1.8rem; font-weight: 700; letter-spacing: -.02em;
    background: linear-gradient(120deg, var(--accent) 0%, var(--accent2) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.page-sub { color: var(--muted); font-size: .9rem; margin-top: -.3rem; margin-bottom: 1.5rem; }

/* ── Chat ── */
.chat-wrap {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.2rem 1.4rem;
    max-height: 500px; overflow-y: auto; margin-bottom: 1rem;
}
.chat-msg-user {
    background: var(--accent2); color: #fff;
    border-radius: 14px 14px 2px 14px; padding: .65rem 1rem;
    margin: .4rem 0 .4rem auto; max-width: 78%; font-size: .88rem; width: fit-content; margin-left: auto;
}
.chat-msg-bot {
    background: var(--card); border: 1px solid var(--border); color: var(--text);
    border-radius: 14px 14px 14px 2px; padding: .65rem 1rem;
    margin: .4rem auto .4rem 0; max-width: 88%; font-size: .88rem;
}

/* ── Forecast insight tiles ── */
.fi-box {
    background: linear-gradient(135deg, #171C28 0%, #1a1030 100%);
    border: 1px solid var(--accent2); border-radius: 10px;
    padding: .8rem 1rem; margin: .3rem 0;
}
.fi-label { color: var(--muted); font-size: .7rem; text-transform: uppercase; letter-spacing: .06em; }
.fi-value { font-family: 'Space Mono', monospace; font-size: 1.1rem; font-weight: 700; color: var(--accent); }

.divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

/* ── Help System ── */
.help-banner {
    background: linear-gradient(135deg, #1a1030 0%, #0f1a2a 100%);
    border: 1px solid #7B61FF;
    border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1.2rem;
}
.help-banner h4 { color: #7B61FF; font-family: 'Space Mono', monospace; margin: 0 0 .5rem; font-size: .95rem; }
.help-banner p, .help-banner li { color: #C4CAD8; font-size: .85rem; line-height: 1.6; }
.help-banner ul { padding-left: 1.2rem; margin: .4rem 0; }
.help-step {
    display: flex; gap: .8rem; align-items: flex-start;
    background: var(--card); border-radius: 8px; padding: .7rem 1rem;
    margin-bottom: .5rem; border: 1px solid var(--border);
}
.help-step-num {
    background: var(--accent2); color: #fff;
    border-radius: 50%; width: 24px; height: 24px; min-width: 24px;
    display: flex; align-items: center; justify-content: center;
    font-size: .75rem; font-weight: 700;
}
.help-step-text { font-size: .83rem; color: var(--text); line-height: 1.5; }
.help-step-text strong { color: var(--accent); }
.welcome-box {
    background: linear-gradient(135deg, #0f1a2a 0%, #1a1030 100%);
    border: 1px solid #00F5C4; border-radius: 14px;
    padding: 1.6rem 2rem; margin-bottom: 1.5rem; text-align: center;
}
.welcome-box h2 { color: #00F5C4; font-family: 'Space Mono', monospace; margin: 0 0 .5rem; }
.welcome-box p { color: #C4CAD8; font-size: .9rem; line-height: 1.6; margin: 0; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════ CONSTANTS ════════════════════════════════

SEGMENT_COLORS = ["#00F5C4","#7B61FF","#FF6B6B","#FFB700","#3DD6F5","#FF8C42","#B4E33D","#E040FB"]

REGIONS = {
    "Metro Manila": ["Makati","BGC","Quezon City","Pasig","Mandaluyong"],
    "Visayas":      ["Cebu City","Iloilo","Bacolod","Dumaguete"],
    "Mindanao":     ["Davao","Cagayan de Oro","Zamboanga","General Santos"],
    "Luzon":        ["Baguio","Angeles","Cabanatuan","Lipa"],
}

LIFESTYLE_TAGS = [
    "Tech-Savvy","Frugal Shopper","Brand Loyal","Deal Hunter","Impulse Buyer",
    "Social Influencer","Budget-Conscious","Premium Seeker","Casual Spender","Frequent Traveler"
]

PURCHASE_CATEGORIES = [
    "Food & Dining","Bills & Utilities","E-Commerce","Transport",
    "Health & Wellness","Entertainment","Travel","Retail Fashion","Electronics","Remittance"
]

AD_CAMPAIGNS = {
    "High-Value Digital Natives": {"channel":"Instagram / TikTok Ads","message":"Exclusive cashback rewards on top brands. Pay smarter.","cta":"Claim 10% Cashback","budget_pct":30},
    "Young Professionals":        {"channel":"Facebook / LinkedIn Ads","message":"Manage your finances on the go. Zero fees, real savings.","cta":"Open Free Account","budget_pct":25},
    "Budget-Conscious Families":  {"channel":"Facebook Feed + SMS","message":"Send money to family, pay bills — for free!","cta":"Send ₱50 Free","budget_pct":20},
    "Senior Savers":              {"channel":"Facebook + Email","message":"Simple, safe digital payments. We'll guide you every step.","cta":"Watch How It Works","budget_pct":10},
    "Rural Remittance Users":     {"channel":"SMS / Viber","message":"Receive money from Manila instantly, anywhere.","cta":"Get ₱20 on First Top-Up","budget_pct":15},
}

QUICK_PROMPTS = [
    "What are the top purchase trends this quarter?",
    "Which segment is most likely to churn?",
    "What new product should we launch next?",
    "Recommend a campaign for Budget-Conscious Families",
    "Which region has the highest growth potential?",
    "What do high-income users want most?",
    "Forecast next month's transaction volume",
    "Which purchase category is rising the fastest?",
]

SYSTEM_PROMPT = """You are SegmentIQ's AI Market Analyst — an expert in Philippine e-wallet consumer behavior, mobile payments, digital marketing, and purchase trend forecasting.

You have access to real-time segmentation and time-series data from the platform. Your job:
1. Identify rising/falling purchase trends and explain what they mean for the business.
2. Recommend products, features, or services consumers want based on behavioral patterns.
3. Give segment-specific marketing recommendations (channels, messaging, CTAs, timing).
4. Interpret forecasts in plain business language — no jargon.
5. Flag churn risks and suggest concrete retention tactics.

Style: Be specific and data-driven. Reference actual numbers. Use bullet points for lists.
Bold key insights with **double asterisks**. Keep responses under 350 words unless asked for a report.
Use Philippine Peso (₱). Be aware of Philippine market nuances (OFW remittances, payday cycles, GCash/Maya ecosystem).
"""


# ═══════════════════════════ DATA ENGINE ══════════════════════════════

@st.cache_data
def generate_synthetic_data(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ages = rng.integers(18, 72, n)
    genders = rng.choice(["Male","Female","Non-binary"], n, p=[0.47,0.50,0.03])
    regions = rng.choice(list(REGIONS.keys()), n, p=[0.40,0.20,0.25,0.15])
    cities = [rng.choice(REGIONS[r]) for r in regions]
    income_classes = rng.choice(["A","B","C","D","E"], n, p=[0.05,0.15,0.35,0.35,0.10])
    income_map = {"A":(150000,500000),"B":(60000,150000),"C":(20000,60000),"D":(8000,20000),"E":(2000,8000)}
    monthly_income = np.array([rng.integers(*income_map[c]) for c in income_classes])
    monthly_txn = np.clip(rng.integers(0,80,n) + (monthly_income/10000).astype(int), 0, 120)
    avg_txn_amount = np.clip(rng.normal(monthly_income*0.12/monthly_txn.clip(1), 200), 50, 5000)
    wallet_balance = np.clip(rng.normal(monthly_income*0.08, monthly_income*0.05), 0, 50000)
    lifestyles = rng.choice(LIFESTYLE_TAGS, n)
    app_usage_days = rng.integers(1, 30, n)
    devices = rng.choice(["Android","iOS","KaiOS"], n, p=[0.68,0.27,0.05])
    churn_score = np.clip(1.0-(monthly_txn/120)*0.5-(app_usage_days/30)*0.3+rng.normal(0,0.1,n), 0, 1)
    top_category = rng.choice(PURCHASE_CATEGORIES, n)
    return pd.DataFrame({
        "age":ages,"gender":genders,"region":regions,"city":cities,
        "income_class":income_classes,"monthly_income":monthly_income,
        "monthly_transactions":monthly_txn,"avg_transaction_amount":avg_txn_amount.round(2),
        "wallet_balance":wallet_balance.round(2),"lifestyle":lifestyles,
        "app_usage_days":app_usage_days,"device":devices,
        "churn_risk":churn_score.round(3),"top_category":top_category,
    })


@st.cache_data
def generate_timeseries(seed: int = 42, months: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end_date = datetime(2025, 12, 1)
    dates = [end_date - timedelta(days=30*i) for i in range(months-1, -1, -1)]
    base_volumes = {
        "Food & Dining":8500,"Bills & Utilities":12000,"E-Commerce":9500,
        "Transport":7000,"Health & Wellness":3200,"Entertainment":4100,
        "Travel":2800,"Retail Fashion":5500,"Electronics":3900,"Remittance":6500,
    }
    trends = {
        "Food & Dining":1.008,"Bills & Utilities":1.003,"E-Commerce":1.015,
        "Transport":1.005,"Health & Wellness":1.018,"Entertainment":1.012,
        "Travel":1.020,"Retail Fashion":1.006,"Electronics":1.009,"Remittance":1.004,
    }
    seasonality = {
        "Food & Dining":   [1.0,0.98,1.0,1.02,1.0,0.95,0.98,1.0,1.02,1.05,1.10,1.20],
        "E-Commerce":      [1.0,0.90,0.95,1.0,1.0,1.0,1.0,1.0,1.05,1.10,1.25,1.40],
        "Travel":          [0.8,0.8,1.0,1.1,1.2,1.3,1.3,1.2,1.0,0.9,0.9,1.1],
        "Entertainment":   [1.0,0.95,1.0,1.0,1.1,1.15,1.15,1.1,1.0,1.0,1.05,1.15],
        "Retail Fashion":  [0.9,0.9,1.0,1.0,1.0,0.95,0.95,1.0,1.05,1.0,1.1,1.3],
    }
    records = []
    for cat, base in base_volumes.items():
        seas = seasonality.get(cat, [1.0]*12)
        trend = trends[cat]
        for i, d in enumerate(dates):
            volume = int(base * (trend**i) * seas[d.month-1] * rng.normal(1.0, 0.04))
            amount = volume * rng.normal(350, 40)
            records.append({"date":d,"category":cat,"txn_count":volume,"txn_amount":round(amount,2)})
    return pd.DataFrame(records)


def run_kmeans(df, n_clusters, features):
    X = df[features].copy()
    for col in X.select_dtypes("object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    X_scaled = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels) if n_clusters > 1 else 0.0
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df = df.copy()
    df["cluster"] = labels
    df["pca_x"] = coords[:, 0]
    df["pca_y"] = coords[:, 1]
    return df, sil, pca.explained_variance_ratio_


def auto_label_segments(df, n_clusters):
    labels = {}
    for i in range(n_clusters):
        sub = df[df["cluster"]==i]
        age_med = sub["age"].median()
        income_med = sub["monthly_income"].median()
        txn_med = sub["monthly_transactions"].median()
        if income_med > 80000 and txn_med > 40:            name = "High-Value Digital Natives"
        elif age_med < 32 and income_med > 30000:           name = "Young Professionals"
        elif sub["region"].mode()[0] in ["Mindanao","Luzon"] and income_med < 25000:
                                                            name = "Rural Remittance Users"
        elif age_med > 50:                                  name = "Senior Savers"
        else:                                               name = "Budget-Conscious Families"
        labels[i] = name if name not in labels.values() else f"{name} #{i}"
    return labels


def linear_forecast(series: pd.Series, horizon: int = 6):
    y = series.values.astype(float)
    X = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    future_X = np.arange(len(y), len(y)+horizon).reshape(-1, 1)
    preds = model.predict(future_X)
    residual_std = np.std(y - model.predict(X)) * 0.4
    rng = np.random.default_rng(99)
    preds_noisy = preds + rng.normal(0, residual_std, horizon)
    ci = residual_std * 1.96
    return preds_noisy, preds_noisy - ci, preds_noisy + ci


def compute_category_growth(ts_df):
    growth = {}
    for cat in PURCHASE_CATEGORIES:
        s = ts_df[ts_df["category"]==cat].sort_values("date")["txn_count"]
        if len(s) >= 6:
            old = s.iloc[-6:-3].mean(); new = s.iloc[-3:].mean()
            growth[cat] = round((new-old)/old*100, 1) if old > 0 else 0.0
        else:
            growth[cat] = 0.0
    return growth


def build_ai_context(df_seg, ts_df) -> str:
    seg_sum = df_seg.groupby("segment_name").agg(
        count=("age","count"), avg_age=("age","mean"),
        avg_income=("monthly_income","mean"), avg_txn=("monthly_transactions","mean"),
        avg_churn=("churn_risk","mean"),
        top_lifestyle=("lifestyle", lambda x: x.mode()[0]),
        top_category=("top_category", lambda x: x.mode()[0]),
        top_region=("region", lambda x: x.mode()[0]),
    ).round(1).reset_index()

    seg_lines = [
        f"  • {r['segment_name']}: {r['count']} users, avg age {r['avg_age']:.0f}, "
        f"avg income ₱{r['avg_income']:,.0f}/mo, avg {r['avg_txn']:.0f} txns/mo, "
        f"churn {r['avg_churn']:.2f}, lifestyle: {r['top_lifestyle']}, "
        f"top category: {r['top_category']}, region: {r['top_region']}"
        for _, r in seg_sum.iterrows()
    ]
    growth = compute_category_growth(ts_df)
    cat_lines = [f"  • {k}: {v:+.1f}%" for k,v in sorted(growth.items(), key=lambda x: -x[1])]
    top3 = sorted(growth, key=growth.get, reverse=True)[:3]
    bot2 = sorted(growth, key=growth.get)[:2]

    return (
        f"=== SEGMENTIQ DATA CONTEXT ===\n"
        f"Date: {datetime.now().strftime('%B %Y')} | Market: Philippines E-Wallet\n\n"
        f"OVERALL: {len(df_seg):,} customers | "
        f"Avg income ₱{df_seg['monthly_income'].mean():,.0f}/mo | "
        f"Avg txns {df_seg['monthly_transactions'].mean():.1f}/mo\n\n"
        f"SEGMENTS:\n" + "\n".join(seg_lines) + "\n\n"
        f"PURCHASE CATEGORY 3-MONTH GROWTH:\n" + "\n".join(cat_lines) + "\n\n"
        f"TOP RISING: {', '.join(top3)}\n"
        f"DECLINING: {', '.join(bot2)}"
    )


def rule_based_response(question: str, df_seg: pd.DataFrame, ts_df: pd.DataFrame) -> str:
    q = question.lower()
    growth = compute_category_growth(ts_df)
    top_cats = sorted(growth, key=growth.get, reverse=True)
    churn_seg = df_seg.groupby("segment_name")["churn_risk"].mean().idxmax()
    churn_val  = df_seg.groupby("segment_name")["churn_risk"].mean().max()
    top_inc_seg = df_seg.groupby("segment_name")["monthly_income"].mean().idxmax()
    top_inc_val = df_seg.groupby("segment_name")["monthly_income"].mean().max()
    top_cat_overall = df_seg["top_category"].mode()[0]

    if any(w in q for w in ["trend","rising","growing","hot","popular","next trend","upcoming"]):
        return (
            f"📈 **Top Rising Purchase Trends (Last 3 Months)**\n\n"
            f"• **{top_cats[0]}** — {growth[top_cats[0]]:+.1f}% growth (fastest)\n"
            f"• **{top_cats[1]}** — {growth[top_cats[1]]:+.1f}% growth\n"
            f"• **{top_cats[2]}** — {growth[top_cats[2]]:+.1f}% growth\n\n"
            f"**What this means:** {top_cats[0]} is your next big opportunity. "
            f"Launch targeted cashback promos and merchant partnership deals in this category immediately. "
            f"Consider a dedicated wallet feature (e.g., spend tracker or loyalty rewards) for {top_cats[1]}.\n\n"
            f"**Action:** Run A/B test on {top_cats[0]} push notification campaign targeting users "
            f"with 10+ monthly transactions."
        )
    elif any(w in q for w in ["churn","leave","retain","at risk","losing","drop"]):
        return (
            f"⚠️ **Churn Risk Alert**\n\n"
            f"Highest-risk segment: **{churn_seg}** (score: {churn_val:.2f}/1.0)\n\n"
            f"**Retention Playbook:**\n"
            f"• Send personalized re-engagement push with ₱20 cashback — within 72 hrs of last login\n"
            f"• Offer free bill payment voucher for users dormant 14+ days\n"
            f"• Run a 7-day streak bonus challenge to rebuild app habit\n"
            f"• For high-value users: personal outreach via in-app chat + loyalty tier upgrade\n\n"
            f"**Timing:** Trigger on day 7 of inactivity, escalate on day 14."
        )
    elif any(w in q for w in ["want","need","product","feature","launch","recommend","suggest","new"]):
        return (
            f"💡 **What Consumers Want Right Now**\n\n"
            f"Based on current behavioral patterns:\n"
            f"• **Top purchase category:** {top_cat_overall}\n"
            f"• **Fastest growing:** {top_cats[0]} (+{growth[top_cats[0]]:.1f}%)\n\n"
            f"**Recommended Product/Feature Launches:**\n"
            f"• 🛒 **{top_cats[0]} Cashback Wallet** — real-time spend tracking + 5% cashback cap\n"
            f"• 💳 **BNPL (Buy Now, Pay Later)** for Electronics & Retail Fashion segments\n"
            f"• 🏥 **Micro Health Insurance** — tied to Health & Wellness spending habits\n"
            f"• ✈️ **Travel Savings Jar** — automated round-up feature for Frequent Traveler users\n"
            f"• 📦 **Subscription Management Hub** — for Entertainment & Bills power users\n\n"
            f"**Priority:** Launch {top_cats[0]} feature first — highest ROI based on growth rate."
        )
    elif any(w in q for w in ["region","area","geography","city","location","where"]):
        reg_txn = df_seg.groupby("region")["monthly_transactions"].mean().idxmax()
        return (
            f"🗺️ **Regional Opportunity Analysis**\n\n"
            f"• **{reg_txn}** leads in avg monthly transaction volume\n"
            f"• **Mindanao** — highest untapped growth potential (low penetration + rising income)\n"
            f"• **Visayas** — strong Remittance usage, ideal for OFW product cross-sell\n"
            f"• **Metro Manila** — highest ARPU, prioritize premium features here\n\n"
            f"**Next Quarter Focus:** Davao & Cebu City expansion campaign with localized messaging."
        )
    elif any(w in q for w in ["forecast","next month","predict","future","projection","estimate"]):
        rising = top_cats[0]
        last_vol = int(ts_df[ts_df["category"]==rising].sort_values("date")["txn_count"].iloc[-1])
        proj = int(last_vol * (1 + growth[rising]/100/3))
        return (
            f"🔮 **Next Month Forecast**\n\n"
            f"• **{rising}:** projected {proj:,} transactions (+{growth[rising]/3:.1f}% MoM)\n"
            f"• **Overall platform:** ~+2.8% transaction volume growth expected\n"
            f"• **Revenue peak window:** 15th–16th (payday cycle) and 30th (end-of-month bills)\n\n"
            f"**Prepare for:**\n"
            f"• +5–8% server load spike during payday cycles\n"
            f"• Increased {top_cats[1]} transactions driven by seasonal demand\n"
            f"• Potential dip in Travel category post-holiday period"
        )
    elif any(w in q for w in ["high income","wealthy","premium","rich","affluent","high value"]):
        return (
            f"💎 **High-Income Segment: {top_inc_seg}**\n\n"
            f"Average income: ₱{top_inc_val:,.0f}/month\n\n"
            f"**What they want:**\n"
            f"• Premium cashback tiers (3–5%) on travel, dining, and international brands\n"
            f"• Investment micro-products: money market access, UITF, stock market integration\n"
            f"• Concierge-level support: dedicated chat agent, priority resolution\n"
            f"• Exclusive merchant deals: Shangri-La, airlines, luxury retail, fine dining\n"
            f"• Seamless cross-border payments (for OFW families & frequent travelers)\n\n"
            f"**Best channel:** Instagram retargeting + in-app banner to top 10% power users"
        )
    elif any(w in q for w in ["campaign","marketing","advertis","promote","ads"]):
        return (
            f"📣 **Campaign Strategy Snapshot**\n\n"
            f"| Segment | Channel | Key Message | Budget |\n"
            f"|---------|---------|-------------|--------|\n"
            + "\n".join([
                f"| {k} | {v['channel']} | {v['message'][:40]}... | {v['budget_pct']}% |"
                for k,v in AD_CAMPAIGNS.items()
            ]) +
            f"\n\n**Top Priority:** Allocate 30% budget to High-Value Digital Natives — highest LTV segment."
        )
    else:
        return (
            f"🤖 **SegmentIQ AI Analyst — Ready to Help**\n\n"
            f"I can answer questions about:\n"
            f"• 📈 **Purchase trends** — what's rising and falling\n"
            f"• 💡 **Product recommendations** — what consumers want next\n"
            f"• ⚠️ **Churn risk** — who's leaving and how to retain them\n"
            f"• 🗺️ **Regional growth** — where the best opportunities are\n"
            f"• 🔮 **Forecasts** — next month projections\n"
            f"• 📣 **Campaign strategy** — channels, messages, budget\n\n"
            f"Try: *'What are the top purchase trends this quarter?'*\n\n"
            f"💡 Add your **OpenAI API key** in the sidebar for full GPT-4o powered analysis."
        )


def chat_with_gpt(messages: list, context: str, api_key: str) -> str:
    """Call OpenAI ChatGPT API and return the assistant reply."""
    if not OPENAI_AVAILABLE:
        return "⚠️ `openai` package not installed. Run: `pip install openai`"
    if not api_key:
        return "⚠️ No API key found. Add your OpenAI API key in the sidebar or `.env` file."
    try:
        client = OpenAI(api_key=api_key)
        # Build the message list with system prompt + data context prepended
        full_messages = [
            {"role": "system", "content": SYSTEM_PROMPT + f"\n\n{context}"}
        ] + messages
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=full_messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ OpenAI API error: {str(e)}"


# ═══════════════════════════ SIDEBAR ══════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:.8rem 0 .2rem;font-family:Space Mono,monospace;
         font-size:1.1rem;font-weight:700;color:#00F5C4;'>💳 SegmentIQ v3.0</div>
    <div style='font-size:.75rem;color:#6B7592;margin-bottom:1.2rem;'>
    E-Wallet Intelligence + AI Advisor</div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📁 Data Source")
    data_mode = st.radio("", ["Synthetic Dataset","Upload CSV"], label_visibility="collapsed")
    uploaded_file = None
    if data_mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    n_samples = st.slider("Synthetic records", 500, 5000, 1500, 250) if data_mode == "Synthetic Dataset" else None

    st.markdown("---")
    st.markdown("#### 🎯 Segmentation")
    n_clusters = st.slider("Segments (K)", 2, 8, 5)
    use_demographic   = st.checkbox("Demographic",   True)
    use_geographic    = st.checkbox("Geographic",    True)
    use_behavioral    = st.checkbox("Behavioral",    True)
    use_psychographic = st.checkbox("Psychographic", True)

    st.markdown("---")
    st.markdown("#### 🔍 Filters")
    # Filters rendered dynamically after data load — placeholders set here
    _filter_placeholder = st.empty()

    st.markdown("---")
    st.markdown("#### 🤖 AI Advisor Key")
    api_key_input = st.text_input(
        "OpenAI API Key", type="password",
        value=os.environ.get("OPENAI_API_KEY", ""),
        placeholder="sk-...",
        help="Get yours at platform.openai.com — enables GPT-4o powered analysis.\nOr set OPENAI_API_KEY in your .env file."
    )
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
    if api_key_input and OPENAI_AVAILABLE:
        st.success("✅ GPT-4o connected")
    elif not OPENAI_AVAILABLE:
        st.warning("Run `pip install openai` to enable GPT")
    else:
        st.info("Rule-based mode active")

    st.markdown("---")

    # ── Help Toggle ──────────────────────────────────────────────────
    st.markdown("#### ❓ Help & Guide")
    if "show_help" not in st.session_state:
        st.session_state.show_help = False
    if st.button("📖 Open User Guide", use_container_width=True):
        st.session_state.show_help = not st.session_state.show_help

    if st.session_state.show_help:
        with st.expander("🗺️ Navigation Guide", expanded=True):
            st.markdown("""
**How to use SegmentIQ:**

1. **Set Data Source** — Use synthetic data or upload your own CSV
2. **Choose Segments (K)** — How many customer groups to find (2–8)
3. **Select Features** — Which traits to use for grouping
4. **Apply Filters** — Region and age range to focus the analysis
5. **Explore the tabs** — Each tab is a different view of your data
6. **Ask the AI** — Go to the AI Advisor tab for insights
            """)
        with st.expander("📊 Tab Descriptions"):
            st.markdown("""
- **Segments** — See all customer groups in a scatter plot and pie chart
- **Geography** — Where your customers are located (region, city, device)
- **Psychographics** — Lifestyle, age, and gender breakdowns
- **Campaigns** — Recommended marketing strategy per segment
- **Forecasting** — 6-month transaction volume forecasts
- **AI Advisor** — Chat with an AI for business recommendations
- **Data Explorer** — Browse, filter, and export raw customer records
            """)
        with st.expander("🔧 Settings Explained"):
            st.markdown("""
**Segments (K):** The number of clusters. Start with 5. Higher = more granular groups. Watch the Silhouette Score — closer to 1.0 is better.

**Feature Groups:**
- *Demographic* — Age, gender, income class
- *Geographic* — Region
- *Behavioral* — Transactions, wallet balance, app usage
- *Psychographic* — Lifestyle, device type

**Silhouette Score:** A quality measure for the clustering. Values above 0.3 are good; above 0.5 is excellent.
            """)
        with st.expander("🤖 AI Advisor Help"):
            st.markdown("""
The AI Advisor answers questions about your customer data.

**Without API key** — Smart rule-based responses (always works!)
**With OpenAI key** — Full GPT-4o powered analysis

**Try asking:**
- "Which segment is most likely to churn?"
- "What product should we launch next?"
- "What are the top purchase trends?"
- "Which region has the highest growth?"

Paste your OpenAI key above to unlock GPT-4o mode.
            """)

    st.markdown("---")
    st.caption("© 2025 SegmentIQ | v3.0")


# ═══════════════════════════ LOAD DATA ════════════════════════════════

# ── Column alias map: maps common external column names → internal names ──
COLUMN_ALIASES = {
    # age / birth year
    "age": "age", "year_birth": "age", "birth_year": "age", "Age": "age",
    # gender
    "gender": "gender", "Gender": "gender", "sex": "gender",
    # income
    "income": "monthly_income", "monthly_income": "monthly_income",
    "Income": "monthly_income", "annual_income": "monthly_income",
    # region / location
    "region": "region", "Region": "region", "location": "region", "area": "region", "state": "region",
    # city
    "city": "city", "City": "city",
    # income class
    "income_class": "income_class",
    # transactions
    "monthly_transactions": "monthly_transactions", "numwebpurchases": "monthly_transactions",
    "NumWebPurchases": "monthly_transactions", "num_purchases": "monthly_transactions",
    "numcatalogpurchases": "monthly_transactions", "NumCatalogPurchases": "monthly_transactions",
    "numstorepurchases": "monthly_transactions", "NumStorePurchases": "monthly_transactions",
    # avg transaction
    "avg_transaction_amount": "avg_transaction_amount",
    # wallet / balance
    "wallet_balance": "wallet_balance", "balance": "wallet_balance",
    # lifestyle
    "lifestyle": "lifestyle", "marital_status": "lifestyle", "Marital_Status": "lifestyle",
    "education": "lifestyle", "Education": "lifestyle",
    # device
    "device": "device",
    # app usage
    "app_usage_days": "app_usage_days", "recency": "app_usage_days", "Recency": "app_usage_days",
    # churn risk
    "churn_risk": "churn_risk", "complain": "churn_risk", "Complain": "churn_risk",
    # top category
    "top_category": "top_category",
}

def detect_and_normalize_csv(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Normalize an arbitrary CSV into the app's internal schema as best as possible.
    Returns (normalized_df, schema_info_dict).
    """
    df = df.copy()

    # ── Step 1: Build rename map, but only one source col per target ──
    # Priority order: exact internal name > birth-year derivable > other aliases
    # We use a seen_targets set so the first (best) match wins for each target.
    lower_map = {c.lower().replace(" ", "_"): c for c in df.columns}
    rename = {}
    seen_targets = set()

    # Pass 1: prefer columns whose name IS already the internal target name
    for orig_col in df.columns:
        internal = COLUMN_ALIASES.get(orig_col.lower().replace(" ", "_"))
        if internal and internal == orig_col.lower().replace(" ", "_"):
            # exact match — this column is already correctly named; keep it, skip renaming
            seen_targets.add(internal)

    # Pass 2: map remaining columns, one source per target
    for ext_col, orig_col in lower_map.items():
        internal = COLUMN_ALIASES.get(ext_col)
        if internal is None:
            continue
        if internal in seen_targets:
            continue  # target already filled — skip this source column
        # Don't rename a column to its own name (it's already correct)
        if orig_col == internal:
            seen_targets.add(internal)
            continue
        rename[orig_col] = internal
        seen_targets.add(internal)

    df.rename(columns=rename, inplace=True)

    # ── Step 2: Drop any leftover duplicate columns (keep first occurrence) ──
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # ── Step 3: Derive age from Year_Birth if still missing ──
    if "age" not in df.columns:
        yb_col = [c for c in df.columns if c.lower() == "year_birth"]
        if yb_col:
            df["age"] = datetime.now().year - pd.to_numeric(df[yb_col[0]], errors="coerce")

    # ── Step 4: Derive monthly_transactions by summing purchase cols if missing ──
    if "monthly_transactions" not in df.columns:
        purchase_cols = [c for c in df.columns if "purchase" in c.lower() or "num" in c.lower()]
        numeric_purchase = [c for c in purchase_cols if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_purchase:
            df["monthly_transactions"] = df[numeric_purchase].sum(axis=1)

    # ── Step 5: Scale annual income to monthly if values look annual ──
    if "monthly_income" in df.columns:
        med = df["monthly_income"].dropna().median()
        if pd.notna(med) and med > 200000:
            df["monthly_income"] = df["monthly_income"] / 12

    # ── Step 6: Fill missing internal columns with sensible defaults ──
    if "region" not in df.columns:
        df["region"] = "Unknown"
    if "city" not in df.columns:
        df["city"] = "Unknown"
    if "gender" not in df.columns:
        df["gender"] = "Unknown"
    if "income_class" not in df.columns:
        if "monthly_income" in df.columns:
            inc = pd.to_numeric(df["monthly_income"], errors="coerce").fillna(0)
            df["income_class"] = pd.cut(inc, bins=[0, 8000, 20000, 60000, 150000, 1e9],
                labels=["E", "D", "C", "B", "A"]).astype(str)
        else:
            df["income_class"] = "C"
    if "lifestyle" not in df.columns:
        df["lifestyle"] = "Unknown"
    if "device" not in df.columns:
        df["device"] = "Android"
    if "app_usage_days" not in df.columns:
        df["app_usage_days"] = 15
    if "wallet_balance" not in df.columns:
        if "monthly_income" in df.columns:
            df["wallet_balance"] = pd.to_numeric(df["monthly_income"], errors="coerce").fillna(0) * 0.08
        else:
            df["wallet_balance"] = 5000.0
    if "monthly_income" not in df.columns:
        df["monthly_income"] = 30000.0
    if "age" not in df.columns:
        df["age"] = 35
    if "churn_risk" not in df.columns:
        if "monthly_transactions" in df.columns:
            txn = pd.to_numeric(df["monthly_transactions"], errors="coerce").fillna(0)
            max_t = txn.max() or 1
            df["churn_risk"] = 1 - (txn / max_t * 0.7)
        else:
            df["churn_risk"] = 0.5
    if "top_category" not in df.columns:
        spend_cols = [c for c in df.columns if c.lower().startswith("mnt") or "spend" in c.lower() or "amount" in c.lower()]
        numeric_spend = [c for c in spend_cols if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_spend:
            clean_names = {c: c.replace("Mnt", "").replace("mnt", "").replace("_", " ").strip() for c in numeric_spend}
            df["top_category"] = df[numeric_spend].idxmax(axis=1).map(clean_names)
        else:
            df["top_category"] = "General"

    # ── Step 7: Coerce key numeric columns safely (Series only, never DataFrame) ──
    for col in ["age", "monthly_income", "monthly_transactions", "wallet_balance", "app_usage_days", "churn_risk"]:
        if col not in df.columns:
            continue
        series = df[col]
        # Guard: if somehow still a DataFrame (shouldn't happen after dedup), take first col
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
            df[col] = series
        fallback = series.median() if pd.api.types.is_numeric_dtype(series) else 0
        df[col] = pd.to_numeric(series, errors="coerce").fillna(fallback)

    # Clip age to plausible range
    if "age" in df.columns:
        df["age"] = df["age"].clip(18, 90)

    # ── Step 8: Build schema info for the UI banner ──
    matched = list(rename.values())
    derived = [c for c in ["age", "monthly_transactions", "churn_risk", "top_category", "region"]
               if c not in matched]
    schema_info = {
        "original_columns": matched,
        "derived_columns": derived,
        "is_ewallet": any(c in df.columns for c in ["region", "monthly_transactions", "wallet_balance", "lifestyle", "device"]),
        "has_region": df["region"].nunique() > 1 if "region" in df.columns else False,
        "has_age": "age" in df.columns,
        "numeric_cols": df.select_dtypes(include="number").columns.tolist(),
        "categorical_cols": df.select_dtypes(include="object").columns.tolist(),
    }
    return df, schema_info

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file, sep=None, engine="python")
    df_raw, schema_info = detect_and_normalize_csv(df_raw)
    is_custom_csv = True
else:
    df_raw = generate_synthetic_data(n_samples)
    schema_info = {"is_ewallet": True, "has_region": True, "has_age": True,
                   "original_columns": list(df_raw.columns), "derived_columns": [],
                   "numeric_cols": df_raw.select_dtypes(include="number").columns.tolist(),
                   "categorical_cols": df_raw.select_dtypes(include="object").columns.tolist()}
    is_custom_csv = False

# ── Dynamic sidebar filters based on actual data ──────────────────────
with _filter_placeholder.container():
    if schema_info["has_region"] and df_raw["region"].nunique() > 1:
        region_options = sorted(df_raw["region"].dropna().unique().tolist())
        selected_region = st.multiselect("Region", region_options, default=region_options)
    else:
        selected_region = None

    if schema_info["has_age"]:
        age_min = int(df_raw["age"].min())
        age_max = int(df_raw["age"].max())
        age_range = st.slider("Age range", age_min, age_max, (age_min, age_max))
    else:
        age_range = None

# ── Apply filters ─────────────────────────────────────────────────────
df_filtered = df_raw.copy()
if selected_region is not None and len(selected_region) > 0:
    df_filtered = df_filtered[df_filtered["region"].isin(selected_region)]
if age_range is not None:
    df_filtered = df_filtered[df_filtered["age"].between(*age_range)]

ts_df = generate_timeseries()

# ── Feature pool: only use columns that actually exist ─────────────────
feature_pool = []
if use_demographic:
    for f in ["age","gender","income_class","monthly_income"]:
        if f in df_filtered.columns: feature_pool.append(f)
if use_geographic:
    if "region" in df_filtered.columns and df_filtered["region"].nunique() > 1:
        feature_pool.append("region")
if use_behavioral:
    for f in ["monthly_transactions","avg_transaction_amount","wallet_balance","app_usage_days"]:
        if f in df_filtered.columns: feature_pool.append(f)
if use_psychographic:
    for f in ["lifestyle","device"]:
        if f in df_filtered.columns and df_filtered[f].nunique() > 1: feature_pool.append(f)

# Fallback: use ALL numeric columns from the CSV if pool is too small
if len(feature_pool) < 2 and is_custom_csv:
    numeric_fallback = [c for c in df_filtered.select_dtypes(include="number").columns
                        if c not in ["cluster","pca_x","pca_y"] and df_filtered[c].nunique() > 1]
    feature_pool = numeric_fallback[:10]  # cap at 10

if len(feature_pool) < 2:
    st.error("⚠️ Select at least 2 feature groups in the sidebar."); st.stop()
if len(df_filtered) < 50:
    st.error("⚠️ Too few records after filtering. Adjust sidebar filters."); st.stop()

df_seg, silhouette, pca_var = run_kmeans(df_filtered, n_clusters, feature_pool)
seg_labels = auto_label_segments(df_seg, n_clusters)
df_seg["segment_name"] = df_seg["cluster"].map(seg_labels)
df_seg["color"] = df_seg["cluster"].map(lambda x: SEGMENT_COLORS[x % len(SEGMENT_COLORS)])
ai_context = build_ai_context(df_seg, ts_df)

# ═══════════════════════════ HEADER ═══════════════════════════════════
dash_subtitle = "Uploaded Dataset" if is_custom_csv else "Mobile & E-Wallet · Segmentation · Time-Series Forecasting · AI Market Advisor"
st.markdown(f"""
<div class='page-title'>SegmentIQ — Customer Intelligence Dashboard</div>
<div class='page-sub'>{dash_subtitle}</div>
""", unsafe_allow_html=True)

# ── Custom CSV detection banner ────────────────────────────────────────
if is_custom_csv:
    orig = schema_info.get("original_columns", [])
    derived = schema_info.get("derived_columns", [])
    st.markdown(f"""
    <div class='help-banner' style='border-color:#FFB700;margin-bottom:1rem;'>
      <h4 style='color:#FFB700;'>📂 Custom Dataset Detected</h4>
      <p>Your CSV has been automatically mapped to SegmentIQ's schema.
      <strong style='color:#00F5C4;'>Matched columns:</strong> {", ".join(orig) if orig else "none directly"} &nbsp;|&nbsp;
      <strong style='color:#7B61FF;'>Auto-derived:</strong> {", ".join(derived) if derived else "none"}<br>
      <span style='color:#6B7592;font-size:.8rem;'>Tabs requiring columns not in your data will use sensible defaults. Geography and Campaigns tabs work best with e-wallet data.</span>
      </p>
    </div>
    """, unsafe_allow_html=True)

# ── First-visit welcome banner ─────────────────────────────────────────
if "welcomed" not in st.session_state:
    st.session_state.welcomed = False

if not st.session_state.welcomed:
    st.markdown("""
    <div class='welcome-box'>
      <h2>👋 Welcome to SegmentIQ!</h2>
      <p>This dashboard helps you understand your e-wallet customers through AI-powered segmentation,
      geographic analysis, purchase trend forecasting, and personalized campaign recommendations.<br><br>
      <strong style='color:#00F5C4;'>New here?</strong> Open the <strong>📖 User Guide</strong> in the left sidebar for a full walkthrough,
      or simply click any tab above to start exploring. Each tab has a built-in help panel explaining what you're looking at.
      </p>
    </div>
    """, unsafe_allow_html=True)
    col_dismiss, _ = st.columns([1, 3])
    with col_dismiss:
        if st.button("✅ Got it, let's go!", type="primary"):
            st.session_state.welcomed = True
            st.rerun()

c1,c2,c3,c4,c5 = st.columns(5)
kpi3_val = f"₱{df_filtered['monthly_income'].mean():,.0f}" if "monthly_income" in df_filtered.columns else "N/A"
kpi3_tip = "Average monthly income of all filtered customers." if "monthly_income" in df_filtered.columns else "Income data not found in dataset."
kpi4_val = f"{df_filtered['monthly_transactions'].mean():.1f}" if "monthly_transactions" in df_filtered.columns else "N/A"
kpi4_tip = "Average transactions per customer per month." if "monthly_transactions" in df_filtered.columns else "Transaction count derived or not available."
kpis = [
    (f"{len(df_filtered):,}","Total Customers","Number of customer records after applying your sidebar filters."),
    (str(n_clusters),"Segments Found","The number of distinct customer groups identified by K-Means clustering."),
    (f"{silhouette:.3f}","Silhouette Score","Clustering quality score (0–1). Above 0.3 is good, above 0.5 is excellent."),
    (kpi3_val,"Avg Monthly Income",kpi3_tip),
    (kpi4_val,"Avg Monthly Txns",kpi4_tip),
]
for col,(val,label,tip) in zip([c1,c2,c3,c4,c5],kpis):
    with col:
        st.markdown(f"<div class='kpi-tile'><div class='kpi-value'>{val}</div><div class='kpi-label'>{label}</div></div>",
                    unsafe_allow_html=True)
        st.caption(tip)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ═══════════════════════════ TABS ═════════════════════════════════════
tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
    "📊 Segments","🗺️ Geography","🧠 Psychographics",
    "📣 Campaigns","📈 Forecasting","🤖 AI Advisor","🔬 Data Explorer",
])

# ────────────────── TAB 1: SEGMENTS ──────────────────────────────────
with tab1:
    with st.expander("💡 What is this tab? (click to learn)"):
        st.markdown("""
        <div class='help-banner'>
        <h4>📊 Segments — Customer Grouping Overview</h4>
        <p>This tab uses <strong>K-Means clustering</strong> to automatically group your customers into distinct segments based on their demographic, behavioral, and lifestyle traits.</p>
        <ul>
          <li><strong>Cluster Scatter (PCA 2D):</strong> Each dot is a customer. Dots of the same color belong to the same segment. Clusters that are tightly packed are more distinct.</li>
          <li><strong>Segment Distribution (Pie):</strong> Shows what percentage of your customers fall into each group.</li>
          <li><strong>Segment Profile Summary:</strong> A table comparing average age, income, transactions, wallet balance, and churn risk across segments.</li>
          <li><strong>Income & Transaction Distributions:</strong> Box plots showing the range and spread of income/transactions per segment.</li>
        </ul>
        <p><strong>Tip:</strong> Adjust the number of segments (K) in the sidebar and watch how the clusters change. A higher Silhouette Score means more distinct groups.</p>
        </div>
        """, unsafe_allow_html=True)

    l,r = st.columns([1.1,1], gap="large")
    with l:
        st.markdown("### Cluster Scatter (PCA 2D)")
        _hover_cols = [c for c in ["age","region","monthly_income","lifestyle"] if c in df_seg.columns]
        fig = px.scatter(df_seg,x="pca_x",y="pca_y",color="segment_name",
            color_discrete_sequence=SEGMENT_COLORS,opacity=0.75,height=420,
            hover_data=_hover_cols if _hover_cols else None,
            labels={"pca_x":f"PC1 ({pca_var[0]*100:.1f}% var)","pca_y":f"PC2 ({pca_var[1]*100:.1f}% var)"},
            template="plotly_dark")
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(17,21,30,0.8)",
            font=dict(family="DM Sans"),margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)
    with r:
        st.markdown("### Segment Distribution")
        counts=df_seg["segment_name"].value_counts().reset_index()
        counts.columns=["Segment","Count"]
        fig2=px.pie(counts,values="Count",names="Segment",
            color_discrete_sequence=SEGMENT_COLORS,hole=0.45,height=420,template="plotly_dark")
        fig2.update_traces(textinfo="percent+label",textfont_size=11)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",showlegend=False,
            font=dict(family="DM Sans"),margin=dict(l=0,r=0,t=10,b=0),
            annotations=[dict(text=f"<b>{n_clusters}</b><br>Segs",font_size=16,showarrow=False,font_color="#00F5C4")])
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Segment Profile Summary")
    _agg = {"Count": ("age","count") if "age" in df_seg.columns else (df_seg.columns[0],"count")}
    if "age" in df_seg.columns:            _agg["Avg Age"]        = ("age","mean")
    if "monthly_income" in df_seg.columns: _agg["Avg Income"]     = ("monthly_income","mean")
    if "monthly_transactions" in df_seg.columns: _agg["Avg Txns"] = ("monthly_transactions","mean")
    if "wallet_balance" in df_seg.columns: _agg["Avg Balance"]    = ("wallet_balance","mean")
    if "churn_risk" in df_seg.columns:     _agg["Churn Risk"]     = ("churn_risk","mean")
    if "top_category" in df_seg.columns:   _agg["Top Category"]   = ("top_category", lambda x: x.mode()[0])
    summary = df_seg.groupby("segment_name").agg(**_agg).round(1).reset_index()
    summary.rename(columns={"segment_name":"Segment"}, inplace=True)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("### Key Numeric Distributions by Segment")
    _box_cols = [c for c in ["monthly_income","monthly_transactions","churn_risk","age"] if c in df_seg.columns]
    if len(_box_cols) >= 1:
        ncols_box = min(len(_box_cols), 2)
        _box_titles = tuple(_box_cols[:ncols_box])
        fig3 = make_subplots(rows=1, cols=ncols_box, subplot_titles=_box_titles)
        for i,seg in enumerate(df_seg["segment_name"].unique()):
            sub=df_seg[df_seg["segment_name"]==seg]; c=SEGMENT_COLORS[i%len(SEGMENT_COLORS)]
            for ci, bcol in enumerate(_box_cols[:ncols_box]):
                fig3.add_trace(go.Box(y=sub[bcol],name=seg,marker_color=c,showlegend=(ci==0)),row=1,col=ci+1)
        fig3.update_layout(template="plotly_dark",height=360,paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,21,30,0.8)",font=dict(family="DM Sans"),margin=dict(l=0,r=0,t=30,b=0),showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)


# ────────────────── TAB 2: GEOGRAPHY ─────────────────────────────────
with tab2:
    with st.expander("💡 What is this tab? (click to learn)"):
        st.markdown("""
        <div class='help-banner'>
        <h4>🗺️ Geography — Where Your Customers Are</h4>
        <p>This tab breaks down your customer segments by physical location — useful for targeting regional marketing campaigns or prioritizing where to expand.</p>
        <ul>
          <li><strong>Regional Segment Distribution:</strong> Stacked bar chart showing how each customer segment is distributed across Metro Manila, Visayas, Mindanao, and Luzon.</li>
          <li><strong>Top Cities:</strong> The 10 cities with the most customers, colored by segment.</li>
          <li><strong>Device OS by Region:</strong> A sunburst chart showing which devices (Android, iOS, KaiOS) are most used in each region — critical for mobile campaign planning.</li>
          <li><strong>Income Class × Region Heatmap:</strong> Shows concentration of different income classes per region. Darker = more customers.</li>
        </ul>
        <p><strong>Tip:</strong> Use the Region filter in the sidebar to focus on a specific area.</p>
        </div>
        """, unsafe_allow_html=True)

    _has_real_region = df_seg["region"].nunique() > 1

    if _has_real_region:
        st.markdown("### Regional Segment Distribution")
        geo=df_seg.groupby(["region","segment_name"]).size().reset_index(name="count")
        fig=px.bar(geo,x="region",y="count",color="segment_name",
            color_discrete_sequence=SEGMENT_COLORS,barmode="stack",template="plotly_dark",height=380)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(17,21,30,0.8)",
            font=dict(family="DM Sans"),margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📍 No region data found in your dataset. Showing segment distribution by other dimensions instead.")
        geo_alt = df_seg.groupby(["segment_name"]).size().reset_index(name="count")
        fig_alt = px.bar(geo_alt, x="segment_name", y="count", color="segment_name",
            color_discrete_sequence=SEGMENT_COLORS, template="plotly_dark", height=280,
            title="Segment Size Distribution")
        fig_alt.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,21,30,0.8)",
            font=dict(family="DM Sans"), margin=dict(l=0,r=0,t=40,b=0), showlegend=False)
        st.plotly_chart(fig_alt, use_container_width=True)

    ca,cb=st.columns(2)
    with ca:
        _city_col = "city" if "city" in df_seg.columns and df_seg["city"].nunique() > 1 else None
        if _city_col:
            st.markdown("### Top Cities")
            cs=df_seg.groupby([_city_col,"segment_name"]).size().reset_index(name="count")
            top10=cs.groupby(_city_col)["count"].sum().nlargest(10).index
            fig2=px.bar(cs[cs[_city_col].isin(top10)].sort_values("count"),
                x="count",y=_city_col,color="segment_name",color_discrete_sequence=SEGMENT_COLORS,
                orientation="h",template="plotly_dark",height=380)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(17,21,30,0.8)",
                font=dict(family="DM Sans"),margin=dict(l=0,r=0,t=10,b=0),showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown("### Income Class Distribution")
            ic = df_seg.groupby(["income_class","segment_name"]).size().reset_index(name="count")
            fig2 = px.bar(ic, x="income_class", y="count", color="segment_name",
                color_discrete_sequence=SEGMENT_COLORS, template="plotly_dark", height=380)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,21,30,0.8)",
                font=dict(family="DM Sans"), margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    with cb:
        _dev_col = "device" if "device" in df_seg.columns and df_seg["device"].nunique() > 1 else None
        if _dev_col and _has_real_region:
            st.markdown("### Device OS by Region")
            dr=df_seg.groupby(["region","device"]).size().reset_index(name="count")
            fig3=px.sunburst(dr,path=["region","device"],values="count",
                color_discrete_sequence=SEGMENT_COLORS,template="plotly_dark",height=380)
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",font=dict(family="DM Sans"),margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig3, use_container_width=True)
        elif "marital_status" in df_seg.columns or "lifestyle" in df_seg.columns:
            _grp_col = "lifestyle" if "lifestyle" in df_seg.columns and df_seg["lifestyle"].nunique() > 1 else None
            if _grp_col:
                st.markdown("### Lifestyle by Segment")
                ls = df_seg.groupby([_grp_col,"segment_name"]).size().reset_index(name="count")
                fig3 = px.sunburst(ls, path=[_grp_col,"segment_name"], values="count",
                    color_discrete_sequence=SEGMENT_COLORS, template="plotly_dark", height=380)
                fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans"), margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.markdown("### Segment Size")
            seg_sizes = df_seg.groupby("segment_name").size().reset_index(name="count")
            fig3 = px.pie(seg_sizes, values="count", names="segment_name",
                color_discrete_sequence=SEGMENT_COLORS, template="plotly_dark", height=380, hole=0.4)
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans"), margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig3, use_container_width=True)

    if _has_real_region and "income_class" in df_seg.columns:
        st.markdown("### Income Class × Region Heatmap")
        heat=df_seg.groupby(["region","income_class"]).size().unstack(fill_value=0)
        for inc_class in ["A","B","C","D","E"]:
            if inc_class not in heat.columns: heat[inc_class] = 0
        heat = heat.reindex(columns=["A","B","C","D","E"])
        fig4=go.Figure(go.Heatmap(z=heat.values,x=heat.columns.tolist(),y=heat.index.tolist(),
            colorscale=[[0,"#11151E"],[0.5,"#7B61FF"],[1,"#00F5C4"]],
            text=heat.values,texttemplate="%{text}"))
        fig4.update_layout(template="plotly_dark",height=260,paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,21,30,0.8)",font=dict(family="DM Sans"),
            margin=dict(l=0,r=0,t=10,b=0),xaxis_title="Income Class",yaxis_title="Region")
        st.plotly_chart(fig4, use_container_width=True)


# ────────────────── TAB 3: PSYCHOGRAPHICS ────────────────────────────
with tab3:
    with st.expander("💡 What is this tab? (click to learn)"):
        st.markdown("""
        <div class='help-banner'>
        <h4>🧠 Psychographics — Who Your Customers Are</h4>
        <p>Beyond demographics, psychographics reveal the attitudes, lifestyles, and interests of your customer segments — the "why" behind their spending behavior.</p>
        <ul>
          <li><strong>Lifestyle Tags by Segment:</strong> A heatmap showing which lifestyle labels (e.g. "Deal Hunter", "Tech-Savvy", "Brand Loyal") are most common in each segment.</li>
          <li><strong>Age Distribution:</strong> Violin plots showing the age spread per segment. The wider the shape, the more customers of that age.</li>
          <li><strong>Gender Mix:</strong> Stacked bar chart showing the male/female/non-binary breakdown per segment — useful for ad creative decisions.</li>
          <li><strong>Purchase Category Preference:</strong> Which spending categories (Food, E-Commerce, Travel, etc.) each segment uses most.</li>
        </ul>
        <p><strong>Tip:</strong> Combine lifestyle and purchase category data to craft highly personalized campaign messages.</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Lifestyle / categorical heatmap ───────────────────────────────
    _lifestyle_col = None
    for _lc in ["lifestyle","Education","Marital_Status","education","marital_status"]:
        if _lc in df_seg.columns and df_seg[_lc].nunique() > 1:
            _lifestyle_col = _lc; break

    if _lifestyle_col:
        st.markdown(f"### {_lifestyle_col.replace('_',' ').title()} by Segment")
        fig=px.density_heatmap(df_seg,x="segment_name",y=_lifestyle_col,
            color_continuous_scale=["#11151E","#7B61FF","#00F5C4"],template="plotly_dark",height=400)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(17,21,30,0.8)",
            font=dict(family="DM Sans"),margin=dict(l=0,r=0,t=10,b=0),coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    p1,p2=st.columns(2)
    with p1:
        if "age" in df_seg.columns:
            st.markdown("### Age Distribution")
            fig2=px.violin(df_seg,x="segment_name",y="age",color="segment_name",
                color_discrete_sequence=SEGMENT_COLORS,box=True,points=False,template="plotly_dark",height=350)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(17,21,30,0.8)",
                font=dict(family="DM Sans"),showlegend=False,margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Age data not available in this dataset.")

    with p2:
        _gender_col = "gender" if "gender" in df_seg.columns and df_seg["gender"].nunique() > 1 else None
        if _gender_col:
            st.markdown("### Gender Mix")
            gd=df_seg.groupby(["segment_name",_gender_col]).size().reset_index(name="count")
            gd["pct"]=(gd["count"]/gd.groupby("segment_name")["count"].transform("sum")*100).round(1)
            fig3=px.bar(gd,x="segment_name",y="pct",color=_gender_col,
                color_discrete_sequence=["#7B61FF","#00F5C4","#FF6B6B"],
                barmode="stack",template="plotly_dark",height=350)
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(17,21,30,0.8)",
                font=dict(family="DM Sans"),margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig3, use_container_width=True)
        elif "income_class" in df_seg.columns:
            st.markdown("### Income Class Mix")
            ic=df_seg.groupby(["segment_name","income_class"]).size().reset_index(name="count")
            ic["pct"]=(ic["count"]/ic.groupby("segment_name")["count"].transform("sum")*100).round(1)
            fig3=px.bar(ic,x="segment_name",y="pct",color="income_class",
                color_discrete_sequence=SEGMENT_COLORS,barmode="stack",template="plotly_dark",height=350)
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(17,21,30,0.8)",
                font=dict(family="DM Sans"),margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig3, use_container_width=True)

    if "top_category" in df_seg.columns and df_seg["top_category"].nunique() > 1:
        st.markdown("### Purchase Category Preference by Segment")
        cs=df_seg.groupby(["segment_name","top_category"]).size().reset_index(name="count")
        fig4=px.bar(cs,x="segment_name",y="count",color="top_category",
            barmode="stack",template="plotly_dark",height=360,
            color_discrete_sequence=px.colors.qualitative.Set3)
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(17,21,30,0.8)",
            font=dict(family="DM Sans"),margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig4, use_container_width=True)


# ────────────────── TAB 4: CAMPAIGNS ─────────────────────────────────
with tab4:
    with st.expander("💡 What is this tab? (click to learn)"):
        st.markdown("""
        <div class='help-banner'>
        <h4>📣 Campaigns — Ready-to-Use Marketing Playbooks</h4>
        <p>This tab generates a tailored advertising strategy for each customer segment — complete with recommended channels, messaging, call-to-action (CTA), and budget allocation.</p>
        <ul>
          <li><strong>Campaign Cards:</strong> Each card represents one customer segment. It shows the segment's profile (age, income, region, lifestyle, device) and the recommended ad campaign for them.</li>
          <li><strong>Churn Score:</strong> Shown in red if high (above 0.5). Prioritize these segments for retention campaigns.</li>
          <li><strong>Budget %:</strong> The recommended share of your total ad budget to allocate to each segment.</li>
          <li><strong>Budget Allocation Funnel:</strong> A visual breakdown of suggested budget split across all segments.</li>
        </ul>
        <p><strong>Tip:</strong> Segments with high churn risk AND high income are the most valuable to retain — consider a dedicated win-back campaign for them.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📣 Ad Campaign Playbook")
    for i,seg_name in enumerate(df_seg["segment_name"].unique()):
        sub=df_seg[df_seg["segment_name"]==seg_name]
        color=SEGMENT_COLORS[i%len(SEGMENT_COLORS)]
        camp_key=seg_name.split(" #")[0]
        campaign=AD_CAMPAIGNS.get(camp_key,list(AD_CAMPAIGNS.values())[i%len(AD_CAMPAIGNS)])
        avg_inc = sub["monthly_income"].mean() if "monthly_income" in sub.columns else None
        avg_age = sub["age"].mean() if "age" in sub.columns else None
        top_reg = sub["region"].mode()[0] if "region" in sub.columns and sub["region"].nunique() > 0 else "N/A"
        top_life = sub["lifestyle"].mode()[0] if "lifestyle" in sub.columns else (
                   sub["Education"].mode()[0] if "Education" in sub.columns else "N/A")
        top_dev = sub["device"].mode()[0] if "device" in sub.columns else "N/A"
        churn = sub["churn_risk"].mean() if "churn_risk" in sub.columns else 0.5
        count = len(sub)
        # Build dynamic stat grid
        stat_items = []
        if avg_age is not None: stat_items.append(("AVG AGE", f"{avg_age:.0f} yrs"))
        if avg_inc is not None: stat_items.append(("AVG INCOME", f"₱{avg_inc:,.0f}/mo"))
        if top_reg != "N/A": stat_items.append(("TOP REGION", top_reg))
        if top_life != "N/A": stat_items.append(("LIFESTYLE / EDU", top_life))
        if top_dev != "N/A": stat_items.append(("DEVICE", top_dev))
        stat_html = "".join([
            f"<div><div style='font-size:.7rem;color:#6B7592;'>{k}</div><div style='font-weight:600;'>{v}</div></div>"
            for k,v in stat_items
        ])
        st.markdown(f"""
        <div class='seg-card' style='border-left:3px solid {color};'>
          <div style='display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:.5rem;'>
            <div>
              <span style='font-family:Space Mono,monospace;font-size:1rem;font-weight:700;color:{color};'>{seg_name}</span>
              <span class='badge' style='background:{color}22;color:{color};margin-left:.6rem;'>{count:,} users · {count/len(df_seg)*100:.1f}%</span>
            </div>
            <div style='text-align:right;font-size:.8rem;color:#6B7592;'>
              Churn: <span style='color:{"#FF6B6B" if churn>0.5 else "#00F5C4"};font-weight:600;'>{churn:.2f}</span>
              &nbsp;|&nbsp; Budget: <span style='color:#FFB700;font-weight:600;'>{campaign["budget_pct"]}%</span>
            </div>
          </div>
          <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.7rem;margin:1rem 0;'>
            {stat_html}
          </div>
          <div style='background:#0A0D14;border-radius:8px;padding:.9rem 1rem;margin-top:.5rem;'>
            <div style='font-size:.72rem;color:#6B7592;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem;'>📡 Recommended Campaign</div>
            <div style='display:grid;grid-template-columns:auto 1fr;gap:.4rem .8rem;font-size:.88rem;'>
              <span style='color:#6B7592;'>Channel</span><span style='color:{color};font-weight:600;'>{campaign["channel"]}</span>
              <span style='color:#6B7592;'>Message</span><span>"{campaign["message"]}"</span>
              <span style='color:#6B7592;'>CTA</span>
              <span style='background:{color};color:#0A0D14;padding:.1rem .6rem;border-radius:4px;font-weight:700;display:inline-block;'>{campaign["cta"]}</span>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 💰 Budget Allocation")
    bd=pd.DataFrame({"Segment":list(AD_CAMPAIGNS.keys()),"Budget %":[v["budget_pct"] for v in AD_CAMPAIGNS.values()]})
    fig=px.funnel(bd.sort_values("Budget %",ascending=False),x="Budget %",y="Segment",
        color_discrete_sequence=SEGMENT_COLORS,template="plotly_dark",height=320)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font=dict(family="DM Sans"),margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig, use_container_width=True)


# ────────────────── TAB 5: FORECASTING ───────────────────────────────
with tab5:
    with st.expander("💡 What is this tab? (click to learn)"):
        st.markdown("""
        <div class='help-banner'>
        <h4>📈 Forecasting — Predict Future Purchase Trends</h4>
        <p>This tab shows 24 months of historical transaction data across 10 purchase categories and forecasts the next 1–12 months using linear trend models.</p>
        <ul>
          <li><strong>All Categories Overview:</strong> A line chart showing monthly transaction volume for every category over the last 2 years. Spot trends and seasonal patterns here.</li>
          <li><strong>3-Month Growth Ranking:</strong> A bar chart ranking categories by recent growth. Green = rising, red = falling. Use this to prioritize your next product or campaign.</li>
          <li><strong>Category Drill-Down:</strong> Pick specific categories to see individual forecasts. The dotted line is the predicted future, and the shaded area is the 95% confidence interval (the range it's likely to fall within).</li>
          <li><strong>6-Month Revenue Forecast Table:</strong> A summary table of projected transactions and revenue per category over the next 6 months.</li>
        </ul>
        <p><strong>Tip:</strong> Use the "Forecast Horizon" slider to adjust how many months ahead you want to see. Enable "Show 95% CI" to see the uncertainty range.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📈 Purchase Trend Forecasting — Time-Series Analysis")
    st.caption("24-month historical data + 6-month linear trend forecast with 95% confidence intervals.")

    # All-category overview
    cat_pivot = ts_df.pivot_table(index="date",columns="category",values="txn_count",aggfunc="sum")
    fig_all = go.Figure()
    for i,cat in enumerate(PURCHASE_CATEGORIES):
        if cat in cat_pivot.columns:
            fig_all.add_trace(go.Scatter(x=cat_pivot.index,y=cat_pivot[cat],name=cat,
                line=dict(color=SEGMENT_COLORS[i%len(SEGMENT_COLORS)],width=2),opacity=0.85))
    fig_all.update_layout(template="plotly_dark",height=320,
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(17,21,30,0.8)",
        font=dict(family="DM Sans"),margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(orientation="h",y=-0.3,font_size=10),
        title=dict(text="All Categories — Monthly Transaction Volume",font_size=13,x=0),
        xaxis_title="",yaxis_title="Transaction Count")
    st.plotly_chart(fig_all, use_container_width=True)

    # Category growth ranking
    growth = compute_category_growth(ts_df)
    gdf = pd.DataFrame([{"Category":k,"Growth (%)":v} for k,v in growth.items()]).sort_values("Growth (%)",ascending=True)
    gdf["col"] = gdf["Growth (%)"].apply(lambda x: "#00F5C4" if x>=0 else "#FF6B6B")
    fig_rank = go.Figure(go.Bar(
        x=gdf["Growth (%)"],y=gdf["Category"],orientation="h",
        marker_color=gdf["col"],text=gdf["Growth (%)"].apply(lambda x:f"{x:+.1f}%"),textposition="outside"))
    fig_rank.update_layout(template="plotly_dark",height=360,
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(17,21,30,0.8)",
        font=dict(family="DM Sans"),margin=dict(l=0,r=60,t=30,b=0),
        title=dict(text="3-Month Category Growth Ranking",font_size=13,x=0),
        xaxis_title="3-Month Growth (%)",xaxis=dict(zeroline=True,zerolinecolor="#6B7592"))
    st.plotly_chart(fig_rank, use_container_width=True)

    # Drill-down forecasts
    st.markdown("### Category Drill-Down + Forecast")
    fc1,fc2=st.columns([1,3])
    with fc1:
        selected_cats=st.multiselect("Select Categories",PURCHASE_CATEGORIES,
            default=["E-Commerce","Health & Wellness","Travel","Food & Dining"])
        forecast_horizon=st.slider("Forecast Horizon (months)",1,12,6)
        show_ci=st.checkbox("Show 95% CI",True)

    if selected_cats:
        ncols=2
        rows=[selected_cats[i:i+ncols] for i in range(0,len(selected_cats),ncols)]
        for row_cats in rows:
            cols=st.columns(len(row_cats))
            for col,cat in zip(cols,row_cats):
                with col:
                    cat_ts=ts_df[ts_df["category"]==cat].sort_values("date")
                    hist_dates=cat_ts["date"].tolist()
                    hist_vals=cat_ts["txn_count"].tolist()
                    preds,lo,hi=linear_forecast(cat_ts["txn_count"],forecast_horizon)
                    last_date=hist_dates[-1]
                    fut_dates=[last_date+timedelta(days=30*(i+1)) for i in range(forecast_horizon)]
                    g3m=(cat_ts["txn_count"].iloc[-1]/cat_ts["txn_count"].iloc[-3]-1)*100 if len(cat_ts)>=3 else 0
                    fc_g=(preds[-1]/hist_vals[-1]-1)*100
                    color=SEGMENT_COLORS[PURCHASE_CATEGORIES.index(cat)%len(SEGMENT_COLORS)]
                    ci_rgba=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)"

                    st.markdown(f"""
                    <div style='font-family:Space Mono,monospace;font-size:.85rem;
                         font-weight:700;color:{color};margin-bottom:.3rem;'>{cat}</div>
                    <div style='display:flex;gap:.6rem;margin-bottom:.4rem;'>
                      <div class='fi-box' style='flex:1;border-color:{color}44;'>
                        <div class='fi-label'>3M Growth</div>
                        <div class='fi-value' style='color:{"#00F5C4" if g3m>=0 else "#FF6B6B"};font-size:.9rem;'>{g3m:+.1f}%</div>
                      </div>
                      <div class='fi-box' style='flex:1;border-color:{color}44;'>
                        <div class='fi-label'>Forecast +{forecast_horizon}M</div>
                        <div class='fi-value' style='color:{"#00F5C4" if fc_g>=0 else "#FF6B6B"};font-size:.9rem;'>{fc_g:+.1f}%</div>
                      </div>
                    </div>""", unsafe_allow_html=True)

                    fig_f=go.Figure()
                    fig_f.add_trace(go.Scatter(x=hist_dates,y=hist_vals,name="Historical",
                        line=dict(color=color,width=2.5),mode="lines"))
                    fig_f.add_trace(go.Scatter(x=fut_dates,y=preds,name="Forecast",
                        line=dict(color=color,width=2,dash="dot"),mode="lines"))
                    if show_ci:
                        fig_f.add_trace(go.Scatter(
                            x=fut_dates+fut_dates[::-1],y=list(hi)+list(lo[::-1]),
                            fill="toself",fillcolor=ci_rgba,
                            line=dict(color="rgba(0,0,0,0)"),showlegend=False))
                    fig_f.add_vline(x=last_date,line_dash="dash",line_color="#6B7592",line_width=1)
                    fig_f.update_layout(template="plotly_dark",height=230,
                        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(17,21,30,0.8)",
                        font=dict(family="DM Sans",size=10),margin=dict(l=0,r=0,t=5,b=0),showlegend=False)
                    st.plotly_chart(fig_f, use_container_width=True)

    # Forecast summary table
    st.markdown("### 🔮 6-Month Revenue Forecast Summary")
    rows_f=[]
    for cat in PURCHASE_CATEGORIES:
        ct=ts_df[ts_df["category"]==cat].sort_values("date")
        pc,_,_=linear_forecast(ct["txn_count"],6)
        pa,_,_=linear_forecast(ct["txn_amount"],6)
        rows_f.append({
            "Category":cat,
            "Current Txns/Mo":f"{int(ct['txn_count'].iloc[-1]):,}",
            "Forecast Txns (M+6)":f"{int(pc[-1]):,}",
            "Forecast Revenue (M+6)":f"₱{pa[-1]:,.0f}",
            "6M Revenue Total":f"₱{sum(pa):,.0f}",
            "Trend":"▲ Rising" if pc[-1]>ct["txn_count"].iloc[-1] else "▼ Falling",
        })
    st.dataframe(pd.DataFrame(rows_f),use_container_width=True,hide_index=True)


# ────────────────── TAB 6: AI ADVISOR ────────────────────────────────
with tab6:
    st.markdown("### 🤖 AI Market Advisor")
    st.markdown("""
    <div class='seg-card'>
      <div style='font-size:.88rem;color:#E8ECF4;line-height:1.6;'>
        Ask about <strong style='color:#00F5C4;'>purchase trends</strong>,
        <strong style='color:#7B61FF;'>what consumers want next</strong>,
        <strong style='color:#FF6B6B;'>churn risks</strong>, or
        <strong style='color:#FFB700;'>product & campaign recommendations</strong>
        — all grounded in your live segmentation and time-series data.
        <br><span style='color:#6B7592;font-size:.8rem;'>
        With API key: powered by GPT-4o · Without key: smart rule-based responses
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Session state init
    if "chat_history"   not in st.session_state: st.session_state.chat_history  = []
    if "api_messages"   not in st.session_state: st.session_state.api_messages  = []

    # ── Quick prompt chips ──────────────────────────────────────────
    st.markdown("**Quick Questions:**")
    chip_cols = st.columns(4)
    for idx, prompt in enumerate(QUICK_PROMPTS):
        if chip_cols[idx % 4].button(prompt, key=f"chip_{idx}", use_container_width=True):
            st.session_state["pending_prompt"] = prompt

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Chat display ────────────────────────────────────────────────
    chat_html = "<div class='chat-wrap'>"
    if not st.session_state.chat_history:
        chat_html += """<div style='text-align:center;padding:2.5rem 1rem;color:#6B7592;font-size:.9rem;'>
          👋 Hi! I'm your SegmentIQ AI Analyst.<br>
          Ask me what your customers want, which trends are rising,<br>
          or who's most likely to churn — I'll give you actionable intel.
        </div>"""
    for msg in st.session_state.chat_history:
        # Simple markdown → HTML conversion for bold
        content = msg["content"]
        import re
        content = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#00F5C4;">\1</strong>', content)
        content = content.replace("\n","<br>")
        if msg["role"] == "user":
            chat_html += f"<div class='chat-msg-user'>{content}</div>"
        else:
            chat_html += f"<div class='chat-msg-bot'>{content}</div>"
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # ── Input row ────────────────────────────────────────────────────
    pending = st.session_state.pop("pending_prompt", "")
    inp_c, btn_c, clr_c = st.columns([5,1,1])
    with inp_c:
        user_input = st.text_input("",value=pending,
            placeholder="Ask: What are the top trends? What should we launch next? Who might churn?",
            label_visibility="collapsed", key="chat_input")
    with btn_c:
        send = st.button("Send 🚀", use_container_width=True, type="primary")
    with clr_c:
        if st.button("Clear 🗑️", use_container_width=True):
            st.session_state.chat_history=[]; st.session_state.api_messages=[]; st.rerun()

    if send and user_input.strip():
        user_msg = user_input.strip()
        st.session_state.chat_history.append({"role":"user","content":user_msg})
        st.session_state.api_messages.append({"role":"user","content":user_msg})

        with st.spinner("Analyzing your data..."):
            active_key = os.environ.get("OPENAI_API_KEY","")
            if active_key and OPENAI_AVAILABLE:
                reply = chat_with_gpt(st.session_state.api_messages, ai_context, active_key)
            else:
                reply = rule_based_response(user_msg, df_seg, ts_df)

        st.session_state.chat_history.append({"role":"assistant","content":reply})
        st.session_state.api_messages.append({"role":"assistant","content":reply})
        st.rerun()

    # ── Data context preview ─────────────────────────────────────────
    with st.expander("🔎 View Data Context Sent to AI"):
        st.code(ai_context, language="text")

    # ── Sample insights auto-generated ──────────────────────────────
    st.markdown("---")
    st.markdown("### 💡 Auto-Generated Insights")
    growth = compute_category_growth(ts_df)
    top3 = sorted(growth, key=growth.get, reverse=True)[:3]
    bot2 = sorted(growth, key=growth.get)[:2]
    churn_seg = df_seg.groupby("segment_name")["churn_risk"].mean().idxmax()
    churn_val  = df_seg.groupby("segment_name")["churn_risk"].mean().max()

    ins1,ins2,ins3 = st.columns(3)
    with ins1:
        st.markdown(f"""
        <div class='fi-box' style='border-color:#00F5C4;height:100%;'>
          <div class='fi-label'>🔥 Hottest Category</div>
          <div class='fi-value'>{top3[0]}</div>
          <div style='color:#6B7592;font-size:.78rem;margin-top:.3rem;'>
            {growth[top3[0]]:+.1f}% in 3 months<br>
            Consider a dedicated cashback campaign
          </div>
        </div>""", unsafe_allow_html=True)
    with ins2:
        st.markdown(f"""
        <div class='fi-box' style='border-color:#FF6B6B;height:100%;'>
          <div class='fi-label'>⚠️ Churn Alert</div>
          <div class='fi-value' style='color:#FF6B6B;'>{churn_seg}</div>
          <div style='color:#6B7592;font-size:.78rem;margin-top:.3rem;'>
            Score: {churn_val:.2f}/1.0<br>
            Launch retention campaign now
          </div>
        </div>""", unsafe_allow_html=True)
    with ins3:
        top_cat_all = df_seg["top_category"].mode()[0]
        st.markdown(f"""
        <div class='fi-box' style='border-color:#7B61FF;height:100%;'>
          <div class='fi-label'>🎯 Top Consumer Need</div>
          <div class='fi-value' style='color:#7B61FF;'>{top_cat_all}</div>
          <div style='color:#6B7592;font-size:.78rem;margin-top:.3rem;'>
            Most popular purchase category<br>
            Build a dedicated wallet feature
          </div>
        </div>""", unsafe_allow_html=True)


# ────────────────── TAB 7: DATA EXPLORER ─────────────────────────────
with tab7:
    with st.expander("💡 What is this tab? (click to learn)"):
        st.markdown("""
        <div class='help-banner'>
        <h4>🔬 Data Explorer — Browse & Export Customer Records</h4>
        <p>This tab gives you direct access to the underlying customer-level data used to generate all the charts and insights in this dashboard.</p>
        <ul>
          <li><strong>Filters:</strong> Filter customers by segment, income class (A=highest, E=lowest), and sort by any numeric column.</li>
          <li><strong>Customer Table:</strong> Each row is one customer record showing age, gender, region, income, transactions, wallet balance, lifestyle, device, churn risk, and their segment assignment.</li>
          <li><strong>Download CSV:</strong> Export the filtered data to use in Excel, Google Sheets, or your own analysis tools.</li>
          <li><strong>Correlation Matrix:</strong> A heatmap showing how numeric variables relate to each other. Cyan = strong positive correlation, red = strong negative. For example, you can see if high income correlates with lower churn risk.</li>
        </ul>
        <p><strong>Tip:</strong> Sort by "churn_risk" descending to immediately see your most at-risk customers.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🔬 Segmented Customer Records")
    cf1,cf2,cf3=st.columns(3)
    with cf1:
        filter_seg=st.multiselect("Segment",df_seg["segment_name"].unique().tolist(),
            default=df_seg["segment_name"].unique().tolist())
    with cf2:
        if "income_class" in df_seg.columns and df_seg["income_class"].nunique() > 1:
            filter_inc=st.select_slider("Income Class",options=["All","A","B","C","D","E"],value="All")
        else:
            filter_inc = "All"
    with cf3:
        _sortable = [c for c in ["monthly_income","monthly_transactions","churn_risk","wallet_balance","age"]
                     if c in df_seg.columns]
        if not _sortable:
            _sortable = df_seg.select_dtypes(include="number").columns.tolist()[:5]
        sort_col = st.selectbox("Sort by", _sortable)

    view_df=df_seg[df_seg["segment_name"].isin(filter_seg)].copy()
    if filter_inc != "All" and "income_class" in view_df.columns:
        view_df=view_df[view_df["income_class"]==filter_inc]
    view_df=view_df.sort_values(sort_col,ascending=False)

    # Show all available display columns + segment_name
    _preferred = ["age","gender","region","city","income_class","monthly_income","monthly_transactions",
                  "avg_transaction_amount","wallet_balance","lifestyle","device","app_usage_days",
                  "churn_risk","top_category","segment_name"]
    dcols = [c for c in _preferred if c in view_df.columns]
    # Also include any original CSV columns not in preferred list (up to 10 extra)
    _extra = [c for c in view_df.columns if c not in dcols and c not in ["cluster","pca_x","pca_y","color"]]
    dcols = dcols + _extra[:10]

    st.dataframe(view_df[dcols].head(500),use_container_width=True,hide_index=True)
    st.caption(f"Showing top 500 of {len(view_df):,} filtered records")

    buf=io.StringIO(); view_df[dcols].to_csv(buf,index=False)
    st.download_button("⬇️ Download CSV",data=buf.getvalue(),file_name="segmentiq_export.csv",mime="text/csv")

    st.markdown("### Feature Correlation Matrix")
    num_cols = view_df.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c not in ["cluster","pca_x","pca_y"] and view_df[c].nunique() > 1][:12]
    if len(num_cols) >= 2:
        corr=view_df[num_cols].corr()
        fig=go.Figure(go.Heatmap(z=corr.values,x=corr.columns.tolist(),y=corr.index.tolist(),
            colorscale=[[0,"#FF6B6B"],[0.5,"#11151E"],[1,"#00F5C4"]],
            zmid=0,text=corr.round(2).values,texttemplate="%{text}"))
        fig.update_layout(template="plotly_dark",height=380,paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,21,30,0.8)",font=dict(family="DM Sans"),margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns for a correlation matrix.")

"""
╔══════════════════════════════════════════════════════════════════════╗
║  SEGMENTIQ v3.1 — Customer Segmentation · Forecasting · AI Advisor   ║
║  Gradio + Python | K-Means · PCA · Time-Series · OpenAI GPT          ║
╚══════════════════════════════════════════════════════════════════════╝

Run:
    pip install -r requirements.txt
    python app_gradio.py

Set your OpenAI API key in a .env file (recommended):
    OPENAI_API_KEY=sk-...
"""

import os, io, warnings, re
import gradio as gr
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
            f"Launch targeted cashback promos and merchant partnership deals in this category immediately.\n\n"
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
            f"• ✈️ **Travel Savings Jar** — automated round-up feature for Frequent Traveler users\n\n"
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
            f"• Increased {top_cats[1]} transactions driven by seasonal demand"
        )
    elif any(w in q for w in ["high income","wealthy","premium","rich","affluent","high value"]):
        return (
            f"💎 **High-Income Segment: {top_inc_seg}**\n\n"
            f"Average income: ₱{top_inc_val:,.0f}/month\n\n"
            f"**What they want:**\n"
            f"• Premium cashback tiers (3–5%) on travel, dining, and international brands\n"
            f"• Investment micro-products: money market access, UITF, stock market integration\n"
            f"• Concierge-level support: dedicated chat agent, priority resolution\n"
            f"• Exclusive merchant deals: Shangri-La, airlines, luxury retail, fine dining\n\n"
            f"**Best channel:** Instagram retargeting + in-app banner to top 10% power users"
        )
    elif any(w in q for w in ["campaign","marketing","advertis","promote","ads"]):
        rows = "\n".join([
            f"| {k} | {v['channel']} | {v['message'][:40]}... | {v['budget_pct']}% |"
            for k,v in AD_CAMPAIGNS.items()
        ])
        return (
            f"📣 **Campaign Strategy Snapshot**\n\n"
            f"| Segment | Channel | Key Message | Budget |\n"
            f"|---------|---------|-------------|--------|\n"
            f"{rows}\n\n"
            f"**Top Priority:** Allocate 30% budget to High-Value Digital Natives — highest LTV segment."
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
            f"💡 Add your **OpenAI API key** in the Settings tab for full GPT-4o powered analysis."
        )


def chat_with_gpt(messages: list, context: str, api_key: str) -> str:
    if not OPENAI_AVAILABLE:
        return "⚠️ `openai` package not installed. Run: `pip install openai`"
    if not api_key:
        return "⚠️ No API key found. Add your OpenAI API key in the Settings tab."
    try:
        client = OpenAI(api_key=api_key)
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


# ═══════════════════════════ CSV NORMALIZATION ═════════════════════════

COLUMN_ALIASES = {
    "age": "age", "year_birth": "age", "birth_year": "age", "Age": "age",
    "gender": "gender", "Gender": "gender", "sex": "gender",
    "income": "monthly_income", "monthly_income": "monthly_income",
    "Income": "monthly_income", "annual_income": "monthly_income",
    "region": "region", "Region": "region", "location": "region", "area": "region", "state": "region",
    "city": "city", "City": "city",
    "income_class": "income_class",
    "monthly_transactions": "monthly_transactions", "numwebpurchases": "monthly_transactions",
    "NumWebPurchases": "monthly_transactions", "num_purchases": "monthly_transactions",
    "avg_transaction_amount": "avg_transaction_amount",
    "wallet_balance": "wallet_balance", "balance": "wallet_balance",
    "lifestyle": "lifestyle", "marital_status": "lifestyle", "Marital_Status": "lifestyle",
    "education": "lifestyle", "Education": "lifestyle",
    "device": "device",
    "app_usage_days": "app_usage_days", "recency": "app_usage_days", "Recency": "app_usage_days",
    "churn_risk": "churn_risk", "complain": "churn_risk", "Complain": "churn_risk",
    "top_category": "top_category",
}


def detect_and_normalize_csv(df: pd.DataFrame):
    df = df.copy()
    lower_map = {c.lower().replace(" ", "_"): c for c in df.columns}
    rename = {}
    seen_targets = set()

    for orig_col in df.columns:
        internal = COLUMN_ALIASES.get(orig_col.lower().replace(" ", "_"))
        if internal and internal == orig_col.lower().replace(" ", "_"):
            seen_targets.add(internal)

    for ext_col, orig_col in lower_map.items():
        internal = COLUMN_ALIASES.get(ext_col)
        if internal is None:
            continue
        if internal in seen_targets:
            continue
        if orig_col == internal:
            seen_targets.add(internal)
            continue
        rename[orig_col] = internal
        seen_targets.add(internal)

    df.rename(columns=rename, inplace=True)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    if "age" not in df.columns:
        yb_col = [c for c in df.columns if c.lower() == "year_birth"]
        if yb_col:
            df["age"] = datetime.now().year - pd.to_numeric(df[yb_col[0]], errors="coerce")

    if "monthly_transactions" not in df.columns:
        purchase_cols = [c for c in df.columns if "purchase" in c.lower() or "num" in c.lower()]
        numeric_purchase = [c for c in purchase_cols if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_purchase:
            df["monthly_transactions"] = df[numeric_purchase].sum(axis=1)

    if "monthly_income" in df.columns:
        med = df["monthly_income"].dropna().median()
        if pd.notna(med) and med > 200000:
            df["monthly_income"] = df["monthly_income"] / 12

    defaults = {
        "region": "Unknown", "city": "Unknown", "gender": "Unknown",
        "lifestyle": "Unknown", "device": "Android", "app_usage_days": 15,
        "monthly_income": 30000.0, "age": 35,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    if "income_class" not in df.columns:
        if "monthly_income" in df.columns:
            inc = pd.to_numeric(df["monthly_income"], errors="coerce").fillna(0)
            df["income_class"] = pd.cut(inc, bins=[0, 8000, 20000, 60000, 150000, 1e9],
                labels=["E", "D", "C", "B", "A"]).astype(str)
        else:
            df["income_class"] = "C"

    if "wallet_balance" not in df.columns:
        if "monthly_income" in df.columns:
            df["wallet_balance"] = pd.to_numeric(df["monthly_income"], errors="coerce").fillna(0) * 0.08
        else:
            df["wallet_balance"] = 5000.0

    if "churn_risk" not in df.columns:
        if "monthly_transactions" in df.columns:
            txn = pd.to_numeric(df["monthly_transactions"], errors="coerce").fillna(0)
            max_t = txn.max() or 1
            df["churn_risk"] = 1 - (txn / max_t * 0.7)
        else:
            df["churn_risk"] = 0.5

    if "top_category" not in df.columns:
        spend_cols = [c for c in df.columns if c.lower().startswith("mnt") or "spend" in c.lower()]
        numeric_spend = [c for c in spend_cols if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_spend:
            clean_names = {c: c.replace("Mnt", "").replace("mnt", "").replace("_", " ").strip() for c in numeric_spend}
            df["top_category"] = df[numeric_spend].idxmax(axis=1).map(clean_names)
        else:
            df["top_category"] = "General"

    for col in ["age", "monthly_income", "monthly_transactions", "wallet_balance", "app_usage_days", "churn_risk"]:
        if col not in df.columns:
            continue
        series = df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
            df[col] = series
        fallback = series.median() if pd.api.types.is_numeric_dtype(series) else 0
        df[col] = pd.to_numeric(series, errors="coerce").fillna(fallback)

    if "age" in df.columns:
        df["age"] = df["age"].clip(18, 90)

    schema_info = {
        "original_columns": list(rename.values()),
        "is_ewallet": any(c in df.columns for c in ["region", "monthly_transactions", "wallet_balance"]),
        "has_region": df["region"].nunique() > 1 if "region" in df.columns else False,
        "has_age": "age" in df.columns,
    }
    return df, schema_info


# ═══════════════════════════ CORE PROCESSING ══════════════════════════

def process_data(uploaded_file, n_samples, n_clusters,
                 use_demographic, use_geographic, use_behavioral, use_psychographic,
                 selected_regions, age_min_val, age_max_val):
    """Main data processing pipeline. Returns (df_seg, ts_df, silhouette, pca_var, is_custom)."""
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file, sep=None, engine="python")
        df_raw, schema_info = detect_and_normalize_csv(df_raw)
        is_custom = True
    else:
        df_raw = generate_synthetic_data(int(n_samples))
        is_custom = False

    # Apply filters
    df_filtered = df_raw.copy()
    if selected_regions and len(selected_regions) > 0 and "region" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["region"].isin(selected_regions)]
    if "age" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["age"].between(age_min_val, age_max_val)]

    if len(df_filtered) < 50:
        return None, None, None, None, is_custom, "⚠️ Too few records after filtering. Adjust filters."

    # Build feature pool
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

    if len(feature_pool) < 2 and is_custom:
        numeric_fallback = [c for c in df_filtered.select_dtypes(include="number").columns
                            if c not in ["cluster","pca_x","pca_y"] and df_filtered[c].nunique() > 1]
        feature_pool = numeric_fallback[:10]

    if len(feature_pool) < 2:
        return None, None, None, None, is_custom, "⚠️ Select at least 2 feature groups."

    df_seg, silhouette, pca_var = run_kmeans(df_filtered, int(n_clusters), feature_pool)
    seg_labels = auto_label_segments(df_seg, int(n_clusters))
    df_seg["segment_name"] = df_seg["cluster"].map(seg_labels)
    df_seg["color"] = df_seg["cluster"].map(lambda x: SEGMENT_COLORS[x % len(SEGMENT_COLORS)])
    ts_df = generate_timeseries()

    return df_seg, ts_df, silhouette, pca_var, is_custom, None


# ═══════════════════════════ TAB RENDER FUNCTIONS ═════════════════════

def render_segments(df_seg, silhouette, pca_var, n_clusters):
    if df_seg is None:
        return None, None, None, None, "No data available."

    # Scatter
    hover_cols = [c for c in ["age","region","monthly_income","lifestyle"] if c in df_seg.columns]
    fig_scatter = px.scatter(df_seg, x="pca_x", y="pca_y", color="segment_name",
        color_discrete_sequence=SEGMENT_COLORS, opacity=0.75, height=420,
        hover_data=hover_cols if hover_cols else None,
        labels={"pca_x":f"PC1 ({pca_var[0]*100:.1f}% var)","pca_y":f"PC2 ({pca_var[1]*100:.1f}% var)"},
        template="plotly_dark")
    fig_scatter.update_traces(marker=dict(size=5))
    fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,21,30,0.8)",
        font=dict(family="DM Sans"), margin=dict(l=0,r=0,t=10,b=0))

    # Pie
    counts = df_seg["segment_name"].value_counts().reset_index()
    counts.columns = ["Segment","Count"]
    fig_pie = px.pie(counts, values="Count", names="Segment",
        color_discrete_sequence=SEGMENT_COLORS, hole=0.45, height=420, template="plotly_dark")
    fig_pie.update_traces(textinfo="percent+label", textfont_size=11)
    fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
        font=dict(family="DM Sans"), margin=dict(l=0,r=0,t=10,b=0))

    # Summary table
    agg = {"Count": ("age","count")}
    if "age" in df_seg.columns:            agg["Avg Age"]       = ("age","mean")
    if "monthly_income" in df_seg.columns: agg["Avg Income"]    = ("monthly_income","mean")
    if "monthly_transactions" in df_seg.columns: agg["Avg Txns"] = ("monthly_transactions","mean")
    if "wallet_balance" in df_seg.columns: agg["Avg Balance"]   = ("wallet_balance","mean")
    if "churn_risk" in df_seg.columns:     agg["Churn Risk"]    = ("churn_risk","mean")
    if "top_category" in df_seg.columns:   agg["Top Category"]  = ("top_category", lambda x: x.mode()[0])
    summary = df_seg.groupby("segment_name").agg(**agg).round(1).reset_index()
    summary.rename(columns={"segment_name":"Segment"}, inplace=True)

    # Box plots
    box_cols = [c for c in ["monthly_income","monthly_transactions","churn_risk","age"] if c in df_seg.columns]
    ncols_box = min(len(box_cols), 2)
    fig_box = make_subplots(rows=1, cols=ncols_box, subplot_titles=tuple(box_cols[:ncols_box]))
    for i, seg in enumerate(df_seg["segment_name"].unique()):
        sub = df_seg[df_seg["segment_name"]==seg]
        c = SEGMENT_COLORS[i%len(SEGMENT_COLORS)]
        for ci, bcol in enumerate(box_cols[:ncols_box]):
            fig_box.add_trace(go.Box(y=sub[bcol], name=seg, marker_color=c, showlegend=(ci==0)), row=1, col=ci+1)
    fig_box.update_layout(template="plotly_dark", height=360, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,21,30,0.8)", font=dict(family="DM Sans"),
        margin=dict(l=0,r=0,t=30,b=0), showlegend=False)

    return fig_scatter, fig_pie, summary, fig_box


def render_geography(df_seg):
    if df_seg is None:
        return None, None, None, None

    has_real_region = df_seg["region"].nunique() > 1

    # Regional bar
    if has_real_region:
        geo = df_seg.groupby(["region","segment_name"]).size().reset_index(name="count")
        fig_region = px.bar(geo, x="region", y="count", color="segment_name",
            color_discrete_sequence=SEGMENT_COLORS, barmode="stack", template="plotly_dark", height=380)
        fig_region.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,21,30,0.8)",
            font=dict(family="DM Sans"), margin=dict(l=0,r=0,t=10,b=0))
    else:
        geo_alt = df_seg.groupby("segment_name").size().reset_index(name="count")
        fig_region = px.bar(geo_alt, x="segment_name", y="count", color="segment_name",
            color_discrete_sequence=SEGMENT_COLORS, template="plotly_dark", height=380)
        fig_region.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,21,30,0.8)",
            font=dict(family="DM Sans"), margin=dict(l=0,r=0,t=10,b=0), showlegend=False)

    # Cities
    city_col = "city" if "city" in df_seg.columns and df_seg["city"].nunique() > 1 else None
    if city_col:
        cs = df_seg.groupby([city_col,"segment_name"]).size().reset_index(name="count")
        top10 = cs.groupby(city_col)["count"].sum().nlargest(10).index
        fig_city = px.bar(cs[cs[city_col].isin(top10)].sort_values("count"),
            x="count", y=city_col, color="segment_name",
            color_discrete_sequence=SEGMENT_COLORS, orientation="h",
            template="plotly_dark", height=380)
        fig_city.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,21,30,0.8)",
            font=dict(family="DM Sans"), margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
    else:
        ic = df_seg.groupby(["income_class","segment_name"]).size().reset_index(name="count")
        fig_city = px.bar(ic, x="income_class", y="count", color="segment_name",
            color_discrete_sequence=SEGMENT_COLORS, template="plotly_dark", height=380)
        fig_city.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,21,30,0.8)",
            font=dict(family="DM Sans"), margin=dict(l=0,r=0,t=10,b=0), showlegend=False)

    # Device sunburst
    if "device" in df_seg.columns and df_seg["device"].nunique() > 1 and has_real_region:
        dr = df_seg.groupby(["region","device"]).size().reset_index(name="count")
        fig_device = px.sunburst(dr, path=["region","device"], values="count",
            color_discrete_sequence=SEGMENT_COLORS, template="plotly_dark", height=380)
        fig_device.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans"),
            margin=dict(l=0,r=0,t=10,b=0))
    else:
        seg_sizes = df_seg.groupby("segment_name").size().reset_index(name="count")
        fig_device = px.pie(seg_sizes, values="count", names="segment_name",
            color_discrete_sequence=SEGMENT_COLORS, template="plotly_dark", height=380, hole=0.4)
        fig_device.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans"),
            margin=dict(l=0,r=0,t=10,b=0))

    # Heatmap
    if has_real_region and "income_class" in df_seg.columns:
        heat = df_seg.groupby(["region","income_class"]).size().unstack(fill_value=0)
        for ic in ["A","B","C","D","E"]:
            if ic not in heat.columns: heat[ic] = 0
        heat = heat.reindex(columns=["A","B","C","D","E"])
        fig_heat = go.Figure(go.Heatmap(z=heat.values, x=heat.columns.tolist(), y=heat.index.tolist(),
            colorscale=[[0,"#11151E"],[0.5,"#7B61FF"],[1,"#00F5C4"]],
            text=heat.values, texttemplate="%{text}"))
        fig_heat.update_layout(template="plotly_dark", height=260, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,21,30,0.8)", font=dict(family="DM Sans"),
            margin=dict(l=0,r=0,t=10,b=0))
    else:
        fig_heat = None

    return fig_region, fig_city, fig_device, fig_heat


def render_psychographics(df_seg):
    if df_seg is None:
        return None, None, None, None

    # Lifestyle heatmap
    lifestyle_col = None
    for lc in ["lifestyle","Education","Marital_Status","education","marital_status"]:
        if lc in df_seg.columns and df_seg[lc].nunique() > 1:
            lifestyle_col = lc; break

    if lifestyle_col:
        fig_lifestyle = px.density_heatmap(df_seg, x="segment_name", y=lifestyle_col,
            color_continuous_scale=["#11151E","#7B61FF","#00F5C4"], template="plotly_dark", height=400)
        fig_lifestyle.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,21,30,0.8)",
            font=dict(family="DM Sans"), margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False)
    else:
        fig_lifestyle = None

    # Age violin
    if "age" in df_seg.columns:
        fig_age = px.violin(df_seg, x="segment_name", y="age", color="segment_name",
            color_discrete_sequence=SEGMENT_COLORS, box=True, points=False,
            template="plotly_dark", height=350)
        fig_age.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,21,30,0.8)",
            font=dict(family="DM Sans"), showlegend=False, margin=dict(l=0,r=0,t=10,b=0))
    else:
        fig_age = None

    # Gender mix
    gender_col = "gender" if "gender" in df_seg.columns and df_seg["gender"].nunique() > 1 else None
    if gender_col:
        gd = df_seg.groupby(["segment_name",gender_col]).size().reset_index(name="count")
        gd["pct"] = (gd["count"]/gd.groupby("segment_name")["count"].transform("sum")*100).round(1)
        fig_gender = px.bar(gd, x="segment_name", y="pct", color=gender_col,
            color_discrete_sequence=["#7B61FF","#00F5C4","#FF6B6B"],
            barmode="stack", template="plotly_dark", height=350)
        fig_gender.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,21,30,0.8)",
            font=dict(family="DM Sans"), margin=dict(l=0,r=0,t=10,b=0))
    else:
        fig_gender = None

    # Purchase categories
    if "top_category" in df_seg.columns and df_seg["top_category"].nunique() > 1:
        cs = df_seg.groupby(["segment_name","top_category"]).size().reset_index(name="count")
        fig_cat = px.bar(cs, x="segment_name", y="count", color="top_category",
            barmode="stack", template="plotly_dark", height=360,
            color_discrete_sequence=px.colors.qualitative.Set3)
        fig_cat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,21,30,0.8)",
            font=dict(family="DM Sans"), margin=dict(l=0,r=0,t=10,b=0))
    else:
        fig_cat = None

    return fig_lifestyle, fig_age, fig_gender, fig_cat


def render_campaigns(df_seg):
    if df_seg is None:
        return "No data available.", None

    html_parts = ["<div style='font-family:DM Sans,sans-serif;'>"]
    for i, seg_name in enumerate(df_seg["segment_name"].unique()):
        sub = df_seg[df_seg["segment_name"]==seg_name]
        color = SEGMENT_COLORS[i%len(SEGMENT_COLORS)]
        camp_key = seg_name.split(" #")[0]
        campaign = AD_CAMPAIGNS.get(camp_key, list(AD_CAMPAIGNS.values())[i%len(AD_CAMPAIGNS)])
        avg_inc = sub["monthly_income"].mean() if "monthly_income" in sub.columns else None
        avg_age = sub["age"].mean() if "age" in sub.columns else None
        top_reg = sub["region"].mode()[0] if "region" in sub.columns else "N/A"
        churn = sub["churn_risk"].mean() if "churn_risk" in sub.columns else 0.5
        count = len(sub)

        stats = []
        if avg_age is not None: stats.append(f"<b>Age:</b> {avg_age:.0f} yrs")
        if avg_inc is not None: stats.append(f"<b>Income:</b> ₱{avg_inc:,.0f}/mo")
        if top_reg != "N/A": stats.append(f"<b>Region:</b> {top_reg}")
        churn_color = "#FF6B6B" if churn > 0.5 else "#00F5C4"
        stats.append(f"<b>Churn:</b> <span style='color:{churn_color}'>{churn:.2f}</span>")
        stats.append(f"<b>Budget:</b> <span style='color:#FFB700'>{campaign['budget_pct']}%</span>")

        html_parts.append(f"""
        <div style='background:#171C28;border:1px solid #242B3D;border-left:4px solid {color};
                    border-radius:10px;padding:1rem 1.2rem;margin-bottom:1rem;'>
            <div style='font-size:1rem;font-weight:700;color:{color};margin-bottom:.5rem;'>
                {seg_name}
                <span style='background:{color}22;color:{color};border-radius:12px;
                             padding:.1rem .5rem;font-size:.75rem;margin-left:.5rem;'>
                    {count:,} users · {count/len(df_seg)*100:.1f}%
                </span>
            </div>
            <div style='display:flex;gap:1.5rem;flex-wrap:wrap;font-size:.85rem;
                        color:#E8ECF4;margin-bottom:.8rem;'>
                {'&nbsp;|&nbsp;'.join(stats)}
            </div>
            <div style='background:#0A0D14;border-radius:8px;padding:.8rem 1rem;font-size:.88rem;'>
                <div style='color:#6B7592;font-size:.72rem;text-transform:uppercase;
                            letter-spacing:.08em;margin-bottom:.4rem;'>📡 Recommended Campaign</div>
                <div><span style='color:#6B7592;'>Channel:</span>
                     <span style='color:{color};font-weight:600;'> {campaign['channel']}</span></div>
                <div style='margin:.3rem 0;'><span style='color:#6B7592;'>Message:</span>
                     " {campaign['message']}"</div>
                <div><span style='color:#6B7592;'>CTA:</span>
                     <span style='background:{color};color:#0A0D14;padding:.1rem .6rem;
                                  border-radius:4px;font-weight:700;'> {campaign['cta']}</span></div>
            </div>
        </div>
        """)
    html_parts.append("</div>")

    # Budget funnel
    bd = pd.DataFrame({"Segment":list(AD_CAMPAIGNS.keys()), "Budget %":[v["budget_pct"] for v in AD_CAMPAIGNS.values()]})
    fig_budget = px.funnel(bd.sort_values("Budget %", ascending=False), x="Budget %", y="Segment",
        color_discrete_sequence=SEGMENT_COLORS, template="plotly_dark", height=320)
    fig_budget.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans"),
        margin=dict(l=0,r=0,t=10,b=0))

    return "".join(html_parts), fig_budget


def render_forecasting(ts_df, selected_cats, forecast_horizon, show_ci):
    if ts_df is None:
        return None, None, None

    # All categories overview
    cat_pivot = ts_df.pivot_table(index="date", columns="category", values="txn_count", aggfunc="sum")
    fig_all = go.Figure()
    for i, cat in enumerate(PURCHASE_CATEGORIES):
        if cat in cat_pivot.columns:
            fig_all.add_trace(go.Scatter(x=cat_pivot.index, y=cat_pivot[cat], name=cat,
                line=dict(color=SEGMENT_COLORS[i%len(SEGMENT_COLORS)], width=2), opacity=0.85))
    fig_all.update_layout(template="plotly_dark", height=320,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,21,30,0.8)",
        font=dict(family="DM Sans"), margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(orientation="h", y=-0.3, font_size=10),
        title=dict(text="All Categories — Monthly Transaction Volume", font_size=13, x=0),
        xaxis_title="", yaxis_title="Transaction Count")

    # Growth ranking
    growth = compute_category_growth(ts_df)
    gdf = pd.DataFrame([{"Category":k,"Growth (%)":v} for k,v in growth.items()]).sort_values("Growth (%)")
    gdf["col"] = gdf["Growth (%)"].apply(lambda x: "#00F5C4" if x>=0 else "#FF6B6B")
    fig_rank = go.Figure(go.Bar(
        x=gdf["Growth (%)"], y=gdf["Category"], orientation="h",
        marker_color=gdf["col"], text=gdf["Growth (%)"].apply(lambda x:f"{x:+.1f}%"), textposition="outside"))
    fig_rank.update_layout(template="plotly_dark", height=360,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,21,30,0.8)",
        font=dict(family="DM Sans"), margin=dict(l=0,r=60,t=30,b=0),
        title=dict(text="3-Month Category Growth Ranking", font_size=13, x=0),
        xaxis_title="3-Month Growth (%)")

    # Forecast summary table
    rows_f = []
    for cat in PURCHASE_CATEGORIES:
        ct = ts_df[ts_df["category"]==cat].sort_values("date")
        pc, _, _ = linear_forecast(ct["txn_count"], 6)
        pa, _, _ = linear_forecast(ct["txn_amount"], 6)
        rows_f.append({
            "Category": cat,
            "Current Txns/Mo": f"{int(ct['txn_count'].iloc[-1]):,}",
            "Forecast Txns (M+6)": f"{int(pc[-1]):,}",
            "Forecast Revenue (M+6)": f"₱{pa[-1]:,.0f}",
            "6M Revenue Total": f"₱{sum(pa):,.0f}",
            "Trend": "▲ Rising" if pc[-1] > ct["txn_count"].iloc[-1] else "▼ Falling",
        })

    return fig_all, fig_rank, pd.DataFrame(rows_f)


def render_drill_down(ts_df, selected_cats, forecast_horizon, show_ci):
    """Render individual category forecast charts."""
    if ts_df is None or not selected_cats:
        return None

    fig = make_subplots(rows=max(1, (len(selected_cats)+1)//2), cols=2,
                        subplot_titles=selected_cats)

    for idx, cat in enumerate(selected_cats):
        row = idx // 2 + 1
        col = idx % 2 + 1
        cat_ts = ts_df[ts_df["category"]==cat].sort_values("date")
        hist_dates = cat_ts["date"].tolist()
        hist_vals = cat_ts["txn_count"].tolist()
        preds, lo, hi = linear_forecast(cat_ts["txn_count"], int(forecast_horizon))
        last_date = hist_dates[-1]
        fut_dates = [last_date + timedelta(days=30*(i+1)) for i in range(int(forecast_horizon))]
        color = SEGMENT_COLORS[PURCHASE_CATEGORIES.index(cat) % len(SEGMENT_COLORS)]
        ci_rgba = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)"

        fig.add_trace(go.Scatter(x=hist_dates, y=hist_vals, name=f"{cat} (hist)",
            line=dict(color=color, width=2), mode="lines", showlegend=False), row=row, col=col)
        fig.add_trace(go.Scatter(x=fut_dates, y=preds, name=f"{cat} (forecast)",
            line=dict(color=color, width=2, dash="dot"), mode="lines", showlegend=False), row=row, col=col)
        if show_ci:
            fig.add_trace(go.Scatter(
                x=fut_dates + fut_dates[::-1], y=list(hi)+list(lo[::-1]),
                fill="toself", fillcolor=ci_rgba,
                line=dict(color="rgba(0,0,0,0)"), showlegend=False), row=row, col=col)

    num_rows = max(1, (len(selected_cats)+1)//2)
    fig.update_layout(template="plotly_dark", height=280*num_rows,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,21,30,0.8)",
        font=dict(family="DM Sans"), margin=dict(l=0,r=0,t=40,b=0), showlegend=False)
    return fig


# ═══════════════════════════ GRADIO APP ═══════════════════════════════

def build_app():
    # Pre-generate data for defaults
    _ts_df = generate_timeseries()

    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
    :root {
        --bg: #0A0D14; --surface: #11151E; --card: #171C28;
        --border: #242B3D; --accent: #00F5C4; --accent2: #7B61FF;
        --accent3: #FF6B6B; --gold: #FFB700; --text: #E8ECF4; --muted: #6B7592;
    }
    body, .gradio-container { background: var(--bg) !important; font-family: 'DM Sans', sans-serif !important; }
    .gradio-container { max-width: 100% !important; }
    h1, h2, h3 { font-family: 'Space Mono', monospace !important; color: var(--accent) !important; }
    .tab-nav button { background: var(--surface) !important; color: var(--muted) !important;
                      border: 1px solid var(--border) !important; }
    .tab-nav button.selected { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }
    label { color: var(--text) !important; }
    .block { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
    button.primary { background: var(--accent2) !important; color: white !important; }
    """

    with gr.Blocks(title="SegmentIQ — E-Wallet Intelligence", theme=gr.themes.Base(), css=custom_css) as app:

        # State
        df_seg_state = gr.State(None)
        ts_df_state = gr.State(_ts_df)
        chat_history_state = gr.State([])
        api_messages_state = gr.State([])

        gr.HTML("""
        <div style='padding:1.5rem 0 1rem;'>
          <div style='font-family:Space Mono,monospace;font-size:2rem;font-weight:700;
               background:linear-gradient(120deg,#00F5C4,#7B61FF);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            💳 SegmentIQ — Customer Intelligence Dashboard
          </div>
          <div style='color:#6B7592;font-size:.9rem;margin-top:.3rem;'>
            Mobile & E-Wallet · Segmentation · Time-Series Forecasting · AI Market Advisor
          </div>
        </div>
        """)

        # ── KPI Row (dynamic) ──────────────────────────────────────────
        kpi_html = gr.HTML()

        with gr.Row():
            # ── LEFT SIDEBAR ───────────────────────────────────────────
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### ⚙️ Settings")

                with gr.Group():
                    gr.Markdown("**📁 Data Source**")
                    data_mode = gr.Radio(["Synthetic Dataset","Upload CSV"],
                                         value="Synthetic Dataset", label="", show_label=False)
                    csv_upload = gr.File(label="Upload CSV", file_types=[".csv"], visible=False)
                    n_samples = gr.Slider(500, 5000, value=1500, step=250, label="Synthetic records")

                with gr.Group():
                    gr.Markdown("**🎯 Segmentation**")
                    n_clusters = gr.Slider(2, 8, value=5, step=1, label="Segments (K)")
                    use_demographic   = gr.Checkbox(True,  label="Demographic")
                    use_geographic    = gr.Checkbox(True,  label="Geographic")
                    use_behavioral    = gr.Checkbox(True,  label="Behavioral")
                    use_psychographic = gr.Checkbox(True,  label="Psychographic")

                with gr.Group():
                    gr.Markdown("**🔍 Filters**")
                    region_filter = gr.CheckboxGroup(
                        choices=list(REGIONS.keys()),
                        value=list(REGIONS.keys()),
                        label="Regions")
                    age_min = gr.Slider(18, 72, value=18, step=1, label="Age Min")
                    age_max = gr.Slider(18, 72, value=72, step=1, label="Age Max")

                run_btn = gr.Button("🚀 Run Analysis", variant="primary")

                with gr.Group():
                    gr.Markdown("**🔑 OpenAI API Key**")
                    api_key_input = gr.Textbox(
                        value=os.environ.get("OPENAI_API_KEY",""),
                        type="password", placeholder="sk-...", label="", show_label=False)

            # ── MAIN CONTENT ──────────────────────────────────────────
            with gr.Column(scale=4):
                with gr.Tabs():

                    # ── TAB 1: SEGMENTS ──────────────────────────────
                    with gr.Tab("📊 Segments"):
                        with gr.Row():
                            seg_scatter = gr.Plot(label="Cluster Scatter (PCA 2D)")
                            seg_pie     = gr.Plot(label="Segment Distribution")
                        seg_summary = gr.Dataframe(label="Segment Profile Summary",
                                                    wrap=True, interactive=False)
                        seg_box = gr.Plot(label="Key Numeric Distributions")

                    # ── TAB 2: GEOGRAPHY ─────────────────────────────
                    with gr.Tab("🗺️ Geography"):
                        geo_region = gr.Plot(label="Regional Segment Distribution")
                        with gr.Row():
                            geo_city   = gr.Plot(label="Top Cities")
                            geo_device = gr.Plot(label="Device OS by Region")
                        geo_heat = gr.Plot(label="Income Class × Region Heatmap")

                    # ── TAB 3: PSYCHOGRAPHICS ─────────────────────────
                    with gr.Tab("🧠 Psychographics"):
                        psych_lifestyle = gr.Plot(label="Lifestyle by Segment")
                        with gr.Row():
                            psych_age    = gr.Plot(label="Age Distribution")
                            psych_gender = gr.Plot(label="Gender Mix")
                        psych_cat = gr.Plot(label="Purchase Category Preference")

                    # ── TAB 4: CAMPAIGNS ─────────────────────────────
                    with gr.Tab("📣 Campaigns"):
                        camp_html   = gr.HTML(label="Ad Campaign Playbook")
                        camp_budget = gr.Plot(label="Budget Allocation Funnel")

                    # ── TAB 5: FORECASTING ────────────────────────────
                    with gr.Tab("📈 Forecasting"):
                        fc_all    = gr.Plot(label="All Categories Overview")
                        fc_rank   = gr.Plot(label="3-Month Growth Ranking")
                        with gr.Row():
                            fc_cats    = gr.CheckboxGroup(
                                choices=PURCHASE_CATEGORIES,
                                value=["E-Commerce","Health & Wellness","Travel","Food & Dining"],
                                label="Select Categories for Drill-Down")
                            with gr.Column():
                                fc_horizon = gr.Slider(1, 12, value=6, step=1, label="Forecast Horizon (months)")
                                fc_show_ci = gr.Checkbox(True, label="Show 95% CI")
                        fc_drill  = gr.Plot(label="Category Drill-Down + Forecast")
                        fc_table  = gr.Dataframe(label="6-Month Revenue Forecast Summary",
                                                  wrap=True, interactive=False)
                        fc_update_btn = gr.Button("🔄 Update Drill-Down", variant="secondary")

                    # ── TAB 6: AI ADVISOR ─────────────────────────────
                    with gr.Tab("🤖 AI Advisor"):
                        gr.HTML("""
                        <div style='background:#171C28;border:1px solid #242B3D;border-radius:10px;
                                    padding:1rem 1.2rem;margin-bottom:1rem;font-size:.88rem;color:#E8ECF4;'>
                            Ask about <strong style='color:#00F5C4;'>purchase trends</strong>,
                            <strong style='color:#7B61FF;'>what consumers want next</strong>,
                            <strong style='color:#FF6B6B;'>churn risks</strong>, or
                            <strong style='color:#FFB700;'>product & campaign recommendations</strong>.
                        </div>
                        """)

                        gr.Markdown("**Quick Questions:**")
                        with gr.Row():
                            quick_btns = [gr.Button(p, size="sm") for p in QUICK_PROMPTS[:4]]
                        with gr.Row():
                            quick_btns2 = [gr.Button(p, size="sm") for p in QUICK_PROMPTS[4:]]

                        chatbot = gr.Chatbot(height=420, label="SegmentIQ AI Analyst")
                        with gr.Row():
                            chat_input = gr.Textbox(
                                placeholder="Ask: What are the top trends? Who might churn?",
                                label="", show_label=False, scale=5)
                            send_btn   = gr.Button("Send 🚀", variant="primary", scale=1)
                            clear_btn  = gr.Button("Clear 🗑️", scale=1)

                        with gr.Accordion("🔎 View Data Context Sent to AI", open=False):
                            ai_context_box = gr.Textbox(label="", interactive=False, lines=10)

                        gr.Markdown("### 💡 Auto-Generated Insights")
                        insights_html = gr.HTML()

                    # ── TAB 7: DATA EXPLORER ──────────────────────────
                    with gr.Tab("🔬 Data Explorer"):
                        with gr.Row():
                            exp_seg_filter  = gr.Dropdown(label="Segment", multiselect=True, interactive=True)
                            exp_inc_filter  = gr.Dropdown(
                                choices=["All","A","B","C","D","E"], value="All",
                                label="Income Class")
                            exp_sort = gr.Dropdown(
                                choices=["monthly_income","monthly_transactions","churn_risk","wallet_balance","age"],
                                value="churn_risk", label="Sort by")
                        explorer_table = gr.Dataframe(label="Customer Records", wrap=True,
                                                       interactive=False)
                        exp_download = gr.DownloadButton(label="⬇️ Download CSV", visible=False)
                        corr_plot = gr.Plot(label="Feature Correlation Matrix")
                        exp_update_btn = gr.Button("🔄 Apply Filters", variant="secondary")

        # ═══════════════════════════ EVENT HANDLERS ═══════════════════

        def toggle_csv_upload(mode):
            return gr.update(visible=(mode == "Upload CSV"))

        data_mode.change(toggle_csv_upload, inputs=data_mode, outputs=csv_upload)

        def run_analysis(uploaded_file, n_samples, n_clusters,
                         use_dem, use_geo, use_beh, use_psy,
                         regions, age_min_val, age_max_val):
            df_seg, ts_df, silhouette, pca_var, is_custom, err = process_data(
                uploaded_file, n_samples, n_clusters,
                use_dem, use_geo, use_beh, use_psy,
                regions, age_min_val, age_max_val
            )
            if err:
                return (None,)*20 + [err, gr.update(), gr.update()]

            # KPI HTML
            kpi3_val = f"₱{df_seg['monthly_income'].mean():,.0f}" if "monthly_income" in df_seg.columns else "N/A"
            kpi4_val = f"{df_seg['monthly_transactions'].mean():.1f}" if "monthly_transactions" in df_seg.columns else "N/A"
            kpis = [
                (f"{len(df_seg):,}", "Total Customers"),
                (str(int(n_clusters)), "Segments Found"),
                (f"{silhouette:.3f}", "Silhouette Score"),
                (kpi3_val, "Avg Monthly Income"),
                (kpi4_val, "Avg Monthly Txns"),
            ]
            kpi_html_str = "<div style='display:flex;gap:1rem;margin-bottom:1.5rem;flex-wrap:wrap;'>"
            for val, label in kpis:
                kpi_html_str += f"""
                <div style='background:#171C28;border:1px solid #242B3D;border-radius:10px;
                            padding:.8rem 1.2rem;text-align:center;flex:1;min-width:120px;'>
                    <div style='font-family:Space Mono,monospace;font-size:1.6rem;
                                font-weight:700;color:#00F5C4;'>{val}</div>
                    <div style='font-size:.7rem;color:#6B7592;text-transform:uppercase;
                                letter-spacing:.08em;margin-top:.2rem;'>{label}</div>
                </div>"""
            kpi_html_str += "</div>"

            # Segments
            fig_scatter, fig_pie, summary, fig_box = render_segments(df_seg, silhouette, pca_var, n_clusters)

            # Geography
            fig_region, fig_city, fig_device, fig_heat = render_geography(df_seg)

            # Psychographics
            fig_lifestyle, fig_age, fig_gender, fig_cat = render_psychographics(df_seg)

            # Campaigns
            camp_html_str, fig_budget = render_campaigns(df_seg)

            # Forecasting
            fig_all, fig_rank, fc_tbl = render_forecasting(ts_df, None, 6, True)
            fig_drill = render_drill_down(ts_df, ["E-Commerce","Health & Wellness","Travel","Food & Dining"], 6, True)

            # AI context & insights
            ai_ctx = build_ai_context(df_seg, ts_df)
            growth = compute_category_growth(ts_df)
            top3 = sorted(growth, key=growth.get, reverse=True)[:3]
            churn_seg = df_seg.groupby("segment_name")["churn_risk"].mean().idxmax()
            churn_val = df_seg.groupby("segment_name")["churn_risk"].mean().max()
            top_cat_all = df_seg["top_category"].mode()[0]
            insights_str = f"""
            <div style='display:flex;gap:1rem;flex-wrap:wrap;'>
              <div style='background:linear-gradient(135deg,#171C28,#1a1030);border:1px solid #00F5C4;
                          border-radius:10px;padding:.8rem 1rem;flex:1;min-width:200px;'>
                <div style='color:#6B7592;font-size:.7rem;text-transform:uppercase;'>🔥 Hottest Category</div>
                <div style='font-family:Space Mono,monospace;font-size:1.1rem;color:#00F5C4;font-weight:700;'>{top3[0]}</div>
                <div style='color:#6B7592;font-size:.78rem;'>{growth[top3[0]]:+.1f}% in 3 months</div>
              </div>
              <div style='background:linear-gradient(135deg,#171C28,#1a1030);border:1px solid #FF6B6B;
                          border-radius:10px;padding:.8rem 1rem;flex:1;min-width:200px;'>
                <div style='color:#6B7592;font-size:.7rem;text-transform:uppercase;'>⚠️ Churn Alert</div>
                <div style='font-family:Space Mono,monospace;font-size:1.1rem;color:#FF6B6B;font-weight:700;'>{churn_seg}</div>
                <div style='color:#6B7592;font-size:.78rem;'>Score: {churn_val:.2f}/1.0</div>
              </div>
              <div style='background:linear-gradient(135deg,#171C28,#1a1030);border:1px solid #7B61FF;
                          border-radius:10px;padding:.8rem 1rem;flex:1;min-width:200px;'>
                <div style='color:#6B7592;font-size:.7rem;text-transform:uppercase;'>🎯 Top Consumer Need</div>
                <div style='font-family:Space Mono,monospace;font-size:1.1rem;color:#7B61FF;font-weight:700;'>{top_cat_all}</div>
                <div style='color:#6B7592;font-size:.78rem;'>Most popular purchase category</div>
              </div>
            </div>"""

            # Explorer defaults
            seg_choices = df_seg["segment_name"].unique().tolist()
            exp_seg_default = seg_choices

            return (
                df_seg, ts_df,
                kpi_html_str,
                # Segments
                fig_scatter, fig_pie, summary, fig_box,
                # Geography
                fig_region, fig_city, fig_device, fig_heat,
                # Psychographics
                fig_lifestyle, fig_age, fig_gender, fig_cat,
                # Campaigns
                camp_html_str, fig_budget,
                # Forecasting
                fig_all, fig_rank, fig_drill, fc_tbl,
                # AI
                ai_ctx, insights_str,
                # Explorer
                gr.update(choices=seg_choices, value=exp_seg_default),
            )

        run_btn.click(
            run_analysis,
            inputs=[csv_upload, n_samples, n_clusters,
                    use_demographic, use_geographic, use_behavioral, use_psychographic,
                    region_filter, age_min, age_max],
            outputs=[
                df_seg_state, ts_df_state,
                kpi_html,
                seg_scatter, seg_pie, seg_summary, seg_box,
                geo_region, geo_city, geo_device, geo_heat,
                psych_lifestyle, psych_age, psych_gender, psych_cat,
                camp_html, camp_budget,
                fc_all, fc_rank, fc_drill, fc_table,
                ai_context_box, insights_html,
                exp_seg_filter,
            ]
        )

        # ── Forecast drill-down update ─────────────────────────────────
        def update_drill_down(ts_df, cats, horizon, show_ci):
            if ts_df is None: return None
            return render_drill_down(ts_df, cats, int(horizon), show_ci)

        fc_update_btn.click(
            update_drill_down,
            inputs=[ts_df_state, fc_cats, fc_horizon, fc_show_ci],
            outputs=[fc_drill]
        )

        # ── Chat handler ───────────────────────────────────────────────
        def chat_handler(user_input, history, api_messages, df_seg, ts_df, api_key, ai_context):
            if not user_input.strip():
                return history, api_messages, ""
            if df_seg is None:
                df_seg = generate_synthetic_data(1500)
                seg_labels = auto_label_segments(run_kmeans(df_seg, 5,
                    ["age","monthly_income","monthly_transactions","wallet_balance"])[0], 5)
                df_seg, _, _ = run_kmeans(df_seg, 5, ["age","monthly_income","monthly_transactions","wallet_balance"])
                df_seg["segment_name"] = df_seg["cluster"].map(seg_labels)
                if ts_df is None:
                    ts_df = generate_timeseries()
                ai_context = build_ai_context(df_seg, ts_df)

            api_messages = api_messages or []
            api_messages.append({"role":"user","content":user_input})
            active_key = api_key or os.environ.get("OPENAI_API_KEY","")
            if active_key and OPENAI_AVAILABLE:
                reply = chat_with_gpt(api_messages, ai_context or "", active_key)
            else:
                reply = rule_based_response(user_input, df_seg, ts_df)
            api_messages.append({"role":"assistant","content":reply})
            history = history or []
            history.append((user_input, reply))
            return history, api_messages, ""

        send_btn.click(
            chat_handler,
            inputs=[chat_input, chatbot, api_messages_state,
                    df_seg_state, ts_df_state, api_key_input, ai_context_box],
            outputs=[chatbot, api_messages_state, chat_input]
        )
        chat_input.submit(
            chat_handler,
            inputs=[chat_input, chatbot, api_messages_state,
                    df_seg_state, ts_df_state, api_key_input, ai_context_box],
            outputs=[chatbot, api_messages_state, chat_input]
        )

        # Quick prompt buttons
        def set_prompt(prompt):
            return prompt

        for btn in quick_btns + quick_btns2:
            btn.click(set_prompt, inputs=[btn], outputs=[chat_input])

        clear_btn.click(
            lambda: ([], [], ""),
            outputs=[chatbot, api_messages_state, chat_input]
        )

        # ── Data Explorer ──────────────────────────────────────────────
        def update_explorer(df_seg, seg_filter, inc_filter, sort_col):
            if df_seg is None:
                return None, None
            view_df = df_seg.copy()
            if seg_filter:
                view_df = view_df[view_df["segment_name"].isin(seg_filter)]
            if inc_filter != "All" and "income_class" in view_df.columns:
                view_df = view_df[view_df["income_class"] == inc_filter]
            if sort_col in view_df.columns:
                view_df = view_df.sort_values(sort_col, ascending=False)

            preferred = ["age","gender","region","city","income_class","monthly_income",
                         "monthly_transactions","avg_transaction_amount","wallet_balance",
                         "lifestyle","device","app_usage_days","churn_risk","top_category","segment_name"]
            dcols = [c for c in preferred if c in view_df.columns]

            # Correlation matrix
            num_cols = [c for c in view_df.select_dtypes(include="number").columns
                        if c not in ["cluster","pca_x","pca_y"] and view_df[c].nunique() > 1][:12]
            if len(num_cols) >= 2:
                corr = view_df[num_cols].corr()
                fig_corr = go.Figure(go.Heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                    colorscale=[[0,"#FF6B6B"],[0.5,"#11151E"],[1,"#00F5C4"]],
                    zmid=0, text=corr.round(2).values, texttemplate="%{text}"))
                fig_corr.update_layout(template="plotly_dark", height=380, paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(17,21,30,0.8)", font=dict(family="DM Sans"),
                    margin=dict(l=0,r=0,t=10,b=0))
            else:
                fig_corr = None

            return view_df[dcols].head(500), fig_corr

        exp_update_btn.click(
            update_explorer,
            inputs=[df_seg_state, exp_seg_filter, exp_inc_filter, exp_sort],
            outputs=[explorer_table, corr_plot]
        )

    return app


# ═══════════════════════════ LAUNCH ═══════════════════════════════════

if __name__ == "__main__":
    app = build_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
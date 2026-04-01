import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
import time

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FRAUD·NET // Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600&display=swap');

/* ── Root & body ── */
:root {
    --cyan:   #00f5ff;
    --red:    #ff003c;
    --green:  #00ff88;
    --yellow: #ffe600;
    --bg:     #020812;
    --panel:  #060f1e;
    --border: #0d2a45;
    --text:   #c8ddf0;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Rajdhani', sans-serif;
}

[data-testid="stSidebar"] {
    background: #030d1a !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Headings ── */
h1, h2, h3 { font-family: 'Orbitron', monospace !important; }

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--cyan) !important;
    color: var(--cyan) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.15em !important;
    padding: 0.6rem 1.6rem !important;
    border-radius: 2px !important;
    text-transform: uppercase;
    transition: all 0.25s;
    box-shadow: 0 0 8px rgba(0,245,255,0.15);
}
.stButton > button:hover {
    background: rgba(0,245,255,0.08) !important;
    box-shadow: 0 0 22px rgba(0,245,255,0.4), inset 0 0 12px rgba(0,245,255,0.05) !important;
}

/* ── Selectbox / inputs ── */
.stSelectbox > div > div, .stSlider > div {
    background: var(--panel) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { font-family: 'Share Tech Mono', monospace !important; font-size: 0.72rem !important; color: #4a7fa5 !important; letter-spacing: 0.1em; }
[data-testid="stMetricValue"] { font-family: 'Orbitron', monospace !important; font-size: 1.8rem !important; color: var(--cyan) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid var(--border); gap: 0; }
.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em;
    color: #3a6080 !important;
    background: transparent !important;
    border: none !important;
    padding: 0.7rem 1.4rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--cyan) !important;
    border-bottom: 2px solid var(--cyan) !important;
}

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg, var(--cyan), #0090ff) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Custom cards ── */
.glasscard {
    background: linear-gradient(135deg, rgba(6,15,30,0.95), rgba(3,8,18,0.98));
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 0 30px rgba(0,0,0,0.5), inset 0 0 20px rgba(0,245,255,0.02);
    position: relative;
    overflow: hidden;
}
.glasscard::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    opacity: 0.4;
}

/* ── Scanline overlay ── */
.scanlines {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px);
    pointer-events: none; z-index: 9999;
}

/* ── Status badges ── */
.badge-safe   { color: var(--green) !important; font-family: 'Orbitron',monospace; font-size:0.7rem; letter-spacing:.1em; }
.badge-fraud  { color: var(--red)   !important; font-family: 'Orbitron',monospace; font-size:0.7rem; letter-spacing:.1em; }
.badge-warn   { color: var(--yellow)!important; font-family: 'Orbitron',monospace; font-size:0.7rem; letter-spacing:.1em; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>

<!-- scanline overlay -->
<div class="scanlines"></div>

<!-- Hero banner -->
<div style="text-align:center; padding: 2.5rem 0 1rem 0;">
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem; letter-spacing:0.3em; color:#1a5a80; margin-bottom:0.6rem;">
        ▸ ANTHROPIC · NEURAL SECURITY DIVISION · v2.6.1
    </div>
    <h1 style="font-family:'Orbitron',monospace; font-size:2.8rem; font-weight:900; margin:0;
               background: linear-gradient(135deg, #00f5ff 0%, #0090ff 50%, #00f5ff 100%);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               text-shadow:none; letter-spacing:0.08em;">
        FRAUD·NET
    </h1>
    <div style="font-family:'Rajdhani',sans-serif; font-size:1.05rem; color:#4a7fa5; letter-spacing:0.25em; margin-top:0.3rem;">
        CREDIT CARD ANOMALY DETECTION SYSTEM
    </div>
    <div style="width:220px; height:1px; background:linear-gradient(90deg,transparent,#00f5ff,transparent); margin:1rem auto 0;"></div>
</div>
""", unsafe_allow_html=True)


# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Orbitron',monospace; font-size:0.75rem; color:#00f5ff;
                letter-spacing:0.15em; padding:0.5rem 0 1rem 0; border-bottom:1px solid #0d2a45;">
        ⬡  SYSTEM CONTROL
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Dataset")
    uploaded = st.file_uploader("Upload creditcard.csv", type=["csv"])

    st.markdown("#### Sample Size")
    sample_size = st.slider("Rows to sample", 5000, 50000, 20000, 1000)

    st.markdown("#### Test Split")
    test_size = st.slider("Test fraction", 0.1, 0.4, 0.2, 0.05)

    st.markdown("#### Models")
    use_lr = st.checkbox("Logistic Regression", value=True)
    use_dt = st.checkbox("Decision Tree", value=True)
    use_rf = st.checkbox("Random Forest", value=True)

    n_est = st.slider("RF — n_estimators", 10, 200, 50, 10)

    st.markdown("---")
    run_btn = st.button("⚡  INITIATE TRAINING", use_container_width=True)

    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.62rem; color:#1a4060;
                padding-top:1.5rem; line-height:1.8;">
        SYS // FRAUDNET-2026<br>
        BUILD // 20260401.1<br>
        STATUS // STANDBY<br>
    </div>
    """, unsafe_allow_html=True)


# ─── HELPERS ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file, n):
    df = pd.read_csv(file)
    df = df.fillna(df.mean(numeric_only=True))
    df = df.sample(min(n, len(df)), random_state=42)
    return df

def make_confusion_fig(cm, title, color):
    fig = go.Figure(go.Heatmap(
        z=cm, x=['Legit','Fraud'], y=['Legit','Fraud'],
        colorscale=[[0,'#020812'],[1,color]],
        showscale=False,
        text=cm, texttemplate="%{text}",
        hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(family='Orbitron', size=13, color='#c8ddf0')),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Share Tech Mono', color='#4a7fa5'),
        margin=dict(l=10,r=10,t=40,b=10), height=260
    )
    return fig

def make_radar_fig(models_data):
    cats = ['Accuracy','Precision','Recall','F1-Score']
    fig = go.Figure()
    palette = {'LR':'#00f5ff','DT':'#ffe600','RF':'#00ff88'}
    for name, vals in models_data.items():
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill='toself', name=name,
            line=dict(color=palette.get(name,'#fff'), width=2),
            fillcolor=palette.get(name,'#fff').replace(')',' ,0.07)').replace('rgb','rgba') if 'rgb' in palette.get(name,'') else palette.get(name,'#fff') + '15'
        ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0,1], color='#1a4060',
                            tickfont=dict(size=9, family='Share Tech Mono')),
            angularaxis=dict(color='#1a4060',
                             tickfont=dict(size=10, family='Share Tech Mono'))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Share Tech Mono', color='#4a7fa5'),
        legend=dict(font=dict(family='Orbitron', size=10), bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=30,r=30,t=30,b=30), height=320
    )
    return fig


# ─── MAIN TABS ──────────────────────────────────────────────────────────────────
tabs = st.tabs(["◈ OVERVIEW", "⬡ MODEL ARENA", "◉ CONFUSION MATRIX", "◈ LIVE SCAN", "⬡ DATA INTEL"])

# ─────────────────────────── TAB 0 : OVERVIEW ──────────────────────────────────
with tabs[0]:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="glasscard">
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;color:#1a5a80;letter-spacing:.15em;">MODULE</div>
            <div style="font-family:'Orbitron',monospace;font-size:1.1rem;color:#00f5ff;margin-top:.3rem;">ANOMALY DETECT</div>
            <div style="font-family:'Rajdhani',sans-serif;font-size:0.92rem;color:#4a7fa5;margin-top:.5rem;line-height:1.6;">
            Binary classification on PCA-transformed transaction features. Three ensemble strategies benchmarked in real-time.
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glasscard">
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;color:#1a5a80;letter-spacing:.15em;">DATASET</div>
            <div style="font-family:'Orbitron',monospace;font-size:1.1rem;color:#00ff88;margin-top:.3rem;">CREDITCARD.CSV</div>
            <div style="font-family:'Rajdhani',sans-serif;font-size:0.92rem;color:#4a7fa5;margin-top:.5rem;line-height:1.6;">
            284,807 transactions · 30 PCA features (V1–V28) + Amount + Time. Highly imbalanced — ~0.17% fraud.
            </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="glasscard">
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;color:#1a5a80;letter-spacing:.15em;">MODELS</div>
            <div style="font-family:'Orbitron',monospace;font-size:1.1rem;color:#ffe600;margin-top:.3rem;">LR · DT · RF</div>
            <div style="font-family:'Rajdhani',sans-serif;font-size:0.92rem;color:#4a7fa5;margin-top:.5rem;line-height:1.6;">
            Logistic Regression, Decision Tree, Random Forest — compared on accuracy, precision, recall & F1.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="glasscard" style="margin-top:.5rem;">
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;color:#1a5a80;letter-spacing:.15em;margin-bottom:.8rem;">
            HOW TO USE
        </div>
        <div style="font-family:'Rajdhani',sans-serif;font-size:1rem;color:#7ab4d4;line-height:2;">
            <span style="color:#00f5ff;">①</span>&nbsp; Upload <b>creditcard.csv</b> in the sidebar<br>
            <span style="color:#00f5ff;">②</span>&nbsp; Configure sample size, test split, and models<br>
            <span style="color:#00f5ff;">③</span>&nbsp; Hit <b>INITIATE TRAINING</b> — models train live<br>
            <span style="color:#00f5ff;">④</span>&nbsp; Explore results across tabs — arena, confusion matrix, live scan
        </div>
    </div>""", unsafe_allow_html=True)


# ─── TRAINING LOGIC ─────────────────────────────────────────────────────────────
if run_btn:
    if uploaded is None:
        st.error("⚠  No data file uploaded. Please upload creditcard.csv in the sidebar.")
    else:
        with st.spinner("Loading dataset..."):
            data = load_data(uploaded, sample_size)

        X = data.drop('Class', axis=1)
        y = data['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        results = {}
        cms = {}
        reports = {}
        model_objs = {}

        progress_bar = st.progress(0)
        status_txt   = st.empty()

        step = 0
        total_models = sum([use_lr, use_dt, use_rf])
        inc = 1 / max(total_models, 1)

        if use_lr:
            status_txt.markdown('<span style="font-family:Share Tech Mono;color:#00f5ff;font-size:.8rem;">⚡ Training Logistic Regression...</span>', unsafe_allow_html=True)
            lr = LogisticRegression(max_iter=1000, solver='liblinear')
            lr.fit(X_train, y_train)
            pred = lr.predict(X_test)
            results['LR'] = accuracy_score(y_test, pred)
            cms['LR'] = confusion_matrix(y_test, pred)
            reports['LR'] = classification_report(y_test, pred, output_dict=True)
            model_objs['LR'] = lr
            step += 1; progress_bar.progress(step * inc)

        if use_dt:
            status_txt.markdown('<span style="font-family:Share Tech Mono;color:#ffe600;font-size:.8rem;">⚡ Training Decision Tree...</span>', unsafe_allow_html=True)
            dt = DecisionTreeClassifier(random_state=42)
            dt.fit(X_train, y_train)
            pred = dt.predict(X_test)
            results['DT'] = accuracy_score(y_test, pred)
            cms['DT'] = confusion_matrix(y_test, pred)
            reports['DT'] = classification_report(y_test, pred, output_dict=True)
            model_objs['DT'] = dt
            step += 1; progress_bar.progress(step * inc)

        if use_rf:
            status_txt.markdown('<span style="font-family:Share Tech Mono;color:#00ff88;font-size:.8rem;">⚡ Training Random Forest...</span>', unsafe_allow_html=True)
            rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
            rf.fit(X_train, y_train)
            pred = rf.predict(X_test)
            results['RF'] = accuracy_score(y_test, pred)
            cms['RF'] = confusion_matrix(y_test, pred)
            reports['RF'] = classification_report(y_test, pred, output_dict=True)
            model_objs['RF'] = rf
            step += 1; progress_bar.progress(1.0)

        progress_bar.empty()
        status_txt.markdown('<span style="font-family:Orbitron;color:#00ff88;font-size:.8rem;letter-spacing:.1em;">✓ TRAINING COMPLETE</span>', unsafe_allow_html=True)

        st.session_state['results']     = results
        st.session_state['cms']         = cms
        st.session_state['reports']     = reports
        st.session_state['model_objs']  = model_objs
        st.session_state['X_test']      = X_test
        st.session_state['y_test']      = y_test
        st.session_state['trained']     = True
        st.session_state['data']        = data


# ─────────────────────────── TAB 1 : MODEL ARENA ───────────────────────────────
with tabs[1]:
    if st.session_state.get('trained'):
        results = st.session_state['results']
        reports = st.session_state['reports']

        palette = {'LR':'#00f5ff','DT':'#ffe600','RF':'#00ff88'}

        # ── Accuracy metrics ──
        cols = st.columns(len(results))
        for i, (name, acc) in enumerate(results.items()):
            with cols[i]:
                color = palette[name]
                st.markdown(f"""
                <div class="glasscard" style="text-align:center; border-color:{color}30;">
                    <div style="font-family:'Orbitron',monospace;font-size:0.7rem;color:{color};letter-spacing:.15em;">{name}</div>
                    <div style="font-family:'Orbitron',monospace;font-size:2.6rem;color:{color};margin:.4rem 0;">{acc:.4f}</div>
                    <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#1a5a80;">ACCURACY</div>
                </div>""", unsafe_allow_html=True)

        # ── Bar chart ──
        fig_bar = go.Figure()
        for name, acc in results.items():
            fig_bar.add_trace(go.Bar(
                x=[name], y=[acc],
                marker=dict(color=palette[name], opacity=0.85,
                            line=dict(color=palette[name], width=1)),
                text=[f"{acc:.4f}"], textposition='outside',
                textfont=dict(family='Orbitron', size=11, color=palette[name]),
                name=name
            ))
        fig_bar.update_layout(
            title=dict(text="ACCURACY BENCHMARK", font=dict(family='Orbitron',size=13,color='#c8ddf0')),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0.98,1.002], gridcolor='#0d2a45', color='#4a7fa5',
                       tickfont=dict(family='Share Tech Mono',size=10)),
            xaxis=dict(color='#4a7fa5', tickfont=dict(family='Orbitron',size=11)),
            showlegend=False, margin=dict(l=20,r=20,t=40,b=20), height=320,
            bargap=0.4
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Radar chart ──
        st.markdown("### ◈  Model Capability Radar")
        radar_data = {}
        for name, rep in reports.items():
            fraud = rep.get('1', rep.get(1, {}))
            radar_data[name] = [
                results[name],
                fraud.get('precision',0),
                fraud.get('recall',0),
                fraud.get('f1-score',0)
            ]
        st.plotly_chart(make_radar_fig(radar_data), use_container_width=True)

    else:
        st.markdown("""
        <div class="glasscard" style="text-align:center;padding:3rem;">
            <div style="font-family:'Orbitron',monospace;font-size:1rem;color:#1a4060;letter-spacing:.15em;">
                ◈ NO TRAINING DATA<br><br>
                <span style="font-family:'Rajdhani',sans-serif;font-size:0.95rem;color:#2a5070;">
                Upload creditcard.csv and hit INITIATE TRAINING
                </span>
            </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────── TAB 2 : CONFUSION MATRIX ──────────────────────────────
with tabs[2]:
    if st.session_state.get('trained'):
        cms     = st.session_state['cms']
        palette = {'LR':'#00f5ff','DT':'#ffe600','RF':'#00ff88'}
        cols    = st.columns(len(cms))
        for i, (name, cm) in enumerate(cms.items()):
            with cols[i]:
                st.plotly_chart(
                    make_confusion_fig(cm, f"{name} — CONFUSION MATRIX", palette[name]),
                    use_container_width=True
                )

        # ── Per-class report table ──
        st.markdown("### ◈  Detailed Classification Report")
        for name, rep in st.session_state['reports'].items():
            color = palette[name]
            fraud_rep = rep.get('1', rep.get(1, {}))
            legit_rep = rep.get('0', rep.get(0, {}))
            st.markdown(f"""
            <div class="glasscard" style="border-color:{color}30;">
                <div style="font-family:'Orbitron',monospace;font-size:0.8rem;color:{color};
                            letter-spacing:.15em;margin-bottom:.8rem;">{name} — REPORT</div>
                <table style="width:100%;font-family:'Share Tech Mono',monospace;font-size:0.78rem;
                              color:#4a7fa5;border-collapse:collapse;">
                    <tr style="color:#1a5a80;font-size:0.65rem;letter-spacing:.1em;">
                        <td style="padding:.35rem .6rem;">CLASS</td>
                        <td style="padding:.35rem .6rem;">PRECISION</td>
                        <td style="padding:.35rem .6rem;">RECALL</td>
                        <td style="padding:.35rem .6rem;">F1-SCORE</td>
                        <td style="padding:.35rem .6rem;">SUPPORT</td>
                    </tr>
                    <tr style="border-top:1px solid #0d2a45;">
                        <td style="padding:.35rem .6rem;color:#00ff88;">LEGIT</td>
                        <td style="padding:.35rem .6rem;">{legit_rep.get('precision',0):.4f}</td>
                        <td style="padding:.35rem .6rem;">{legit_rep.get('recall',0):.4f}</td>
                        <td style="padding:.35rem .6rem;">{legit_rep.get('f1-score',0):.4f}</td>
                        <td style="padding:.35rem .6rem;">{int(legit_rep.get('support',0))}</td>
                    </tr>
                    <tr style="border-top:1px solid #0d2a45;">
                        <td style="padding:.35rem .6rem;color:#ff003c;">FRAUD</td>
                        <td style="padding:.35rem .6rem;">{fraud_rep.get('precision',0):.4f}</td>
                        <td style="padding:.35rem .6rem;">{fraud_rep.get('recall',0):.4f}</td>
                        <td style="padding:.35rem .6rem;">{fraud_rep.get('f1-score',0):.4f}</td>
                        <td style="padding:.35rem .6rem;">{int(fraud_rep.get('support',0))}</td>
                    </tr>
                </table>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Train models first to see confusion matrices.")


# ─────────────────────────── TAB 3 : LIVE SCAN ─────────────────────────────────
with tabs[3]:
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#1a5a80;
                letter-spacing:.15em;margin-bottom:1rem;">
        ◉  MANUAL TRANSACTION SCAN — enter feature values to classify
    </div>""", unsafe_allow_html=True)

    if not st.session_state.get('trained'):
        st.warning("Train at least one model first.")
    else:
        model_objs = st.session_state['model_objs']
        X_test     = st.session_state['X_test']

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown("#### Load a test transaction")
            idx = st.number_input("Test set row index", 0, len(X_test)-1, 0, 1)
            sample_row = X_test.iloc[[idx]]
            use_random = st.button("🎲 Random transaction")
            if use_random:
                sample_row = X_test.sample(1)

        with col_right:
            st.markdown("#### Scan result")
            model_choice = st.selectbox("Model", list(model_objs.keys()))
            if st.button("⚡ SCAN TRANSACTION"):
                model = model_objs[model_choice]
                pred  = model.predict(sample_row)[0]
                proba = model.predict_proba(sample_row)[0] if hasattr(model, 'predict_proba') else None
                actual = st.session_state['y_test'].iloc[X_test.index.get_loc(sample_row.index[0])] if sample_row.index[0] in st.session_state['y_test'].index else "—"

                verdict_color = "#ff003c" if pred == 1 else "#00ff88"
                verdict_label = "⚠  FRAUD DETECTED" if pred == 1 else "✓  TRANSACTION CLEAR"

                st.markdown(f"""
                <div class="glasscard" style="border-color:{verdict_color}50;text-align:center;padding:1.5rem;">
                    <div style="font-family:'Orbitron',monospace;font-size:1.3rem;
                                color:{verdict_color};letter-spacing:.1em;">{verdict_label}</div>
                    {"<div style='font-family:Share Tech Mono;font-size:.8rem;color:#4a7fa5;margin-top:.7rem;'>FRAUD PROB: <span style='color:#ff003c'>{:.3f}</span> &nbsp;|&nbsp; LEGIT PROB: <span style='color:#00ff88'>{:.3f}</span></div>".format(proba[1], proba[0]) if proba is not None else ""}
                    <div style="font-family:Share Tech Mono;font-size:.68rem;color:#1a4060;margin-top:.5rem;">
                        MODEL: {model_choice} &nbsp;·&nbsp; ACTUAL: {"FRAUD" if actual == 1 else "LEGIT"}
                    </div>
                </div>""", unsafe_allow_html=True)

        # Feature display
        st.markdown("#### Transaction feature vector")
        st.dataframe(
            sample_row.T.rename(columns={sample_row.index[0]: "value"}).style
            .background_gradient(cmap='Blues', subset=['value'])
            .format("{:.5f}"),
            height=320
        )


# ─────────────────────────── TAB 4 : DATA INTEL ────────────────────────────────
with tabs[4]:
    if st.session_state.get('data') is not None:
        data = st.session_state['data']

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{len(data):,}")
        col2.metric("Features", f"{data.shape[1]-1}")
        fraud_count = int(data['Class'].sum())
        col3.metric("Fraud Transactions", f"{fraud_count}")
        col4.metric("Fraud Rate", f"{fraud_count/len(data)*100:.3f}%")

        # Class distribution
        counts = data['Class'].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=['Legit','Fraud'], values=[counts.get(0,0), counts.get(1,0)],
            marker=dict(colors=['#00ff88','#ff003c'],
                        line=dict(color='#020812', width=3)),
            textfont=dict(family='Orbitron', size=11),
            hole=0.55
        ))
        fig_pie.update_layout(
            title=dict(text="CLASS DISTRIBUTION", font=dict(family='Orbitron',size=13,color='#c8ddf0')),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Share Tech Mono', color='#4a7fa5'),
            legend=dict(font=dict(family='Orbitron',size=10), bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=10,r=10,t=40,b=10), height=300
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Amount distribution
        fig_hist = px.histogram(
            data, x='Amount', color='Class',
            color_discrete_map={0:'#00f5ff', 1:'#ff003c'},
            nbins=60, barmode='overlay', opacity=0.7
        )
        fig_hist.update_layout(
            title=dict(text="TRANSACTION AMOUNT DISTRIBUTION", font=dict(family='Orbitron',size=13,color='#c8ddf0')),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='#0d2a45', color='#4a7fa5', tickfont=dict(family='Share Tech Mono',size=10)),
            yaxis=dict(gridcolor='#0d2a45', color='#4a7fa5', tickfont=dict(family='Share Tech Mono',size=10)),
            font=dict(family='Share Tech Mono',color='#4a7fa5'),
            legend=dict(font=dict(family='Orbitron',size=10), bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=10,r=10,t=40,b=10), height=300
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    else:
        st.markdown("""
        <div class="glasscard" style="text-align:center;padding:3rem;">
            <div style="font-family:'Orbitron',monospace;font-size:0.9rem;color:#1a4060;letter-spacing:.15em;">
                ◈ AWAITING DATA FEED
            </div>
        </div>""", unsafe_allow_html=True)

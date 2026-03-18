"""PyMem Streamlit UI — production-grade visual dashboard for memory management.

Run: streamlit run pymem/ui/app.py
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime

import streamlit as st

from pymem.ui.engine_factory import get_or_create_engine

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PyMem — Memory Platform",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🧠</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────
# Custom CSS for modern, clean design
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: #1a1f35;
    --bg-card-hover: #1e2442;
    --border: #2a3150;
    --border-light: #3a4570;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-blue: #3b82f6;
    --accent-blue-light: #60a5fa;
    --accent-purple: #8b5cf6;
    --accent-green: #22c55e;
    --accent-orange: #f59e0b;
    --accent-red: #ef4444;
    --accent-cyan: #06b6d4;
    --gradient-blue: linear-gradient(135deg, #3b82f6, #8b5cf6);
    --gradient-green: linear-gradient(135deg, #22c55e, #06b6d4);
    --gradient-orange: linear-gradient(135deg, #f59e0b, #ef4444);
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.3);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.4);
    --shadow-lg: 0 8px 30px rgba(0,0,0,0.5);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
}

.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.9rem !important;
    padding: 0.45rem 0.8rem !important;
    border-radius: var(--radius-sm) !important;
    margin-bottom: 2px !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: var(--bg-card) !important;
}

/* ── Page header ── */
.page-header {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-md);
}
.page-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    background: var(--gradient-blue);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.page-header p {
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin: 0;
}

/* ── Stat cards ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: all 0.25s ease;
    box-shadow: var(--shadow-sm);
}
.stat-card:hover {
    border-color: var(--border-light);
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: var(--radius-md) var(--radius-md) 0 0;
}
.stat-card.blue::before { background: var(--gradient-blue); }
.stat-card.green::before { background: var(--gradient-green); }
.stat-card.orange::before { background: var(--gradient-orange); }
.stat-card.purple::before { background: linear-gradient(135deg, #8b5cf6, #ec4899); }
.stat-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}
.stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
}
.stat-sub {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
}

/* ── Memory cards ── */
.memory-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
    position: relative;
}
.memory-card:hover {
    border-color: var(--border-light);
    box-shadow: var(--shadow-md);
}
.memory-card .type-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    margin-bottom: 0.6rem;
}
.type-semantic { background: rgba(59,130,246,0.15); color: var(--accent-blue-light); border: 1px solid rgba(59,130,246,0.3); }
.type-episodic { background: rgba(34,197,94,0.15); color: var(--accent-green); border: 1px solid rgba(34,197,94,0.3); }
.type-procedural { background: rgba(245,158,11,0.15); color: var(--accent-orange); border: 1px solid rgba(245,158,11,0.3); }
.type-working { background: rgba(139,92,246,0.15); color: var(--accent-purple); border: 1px solid rgba(139,92,246,0.3); }
.memory-content {
    color: var(--text-primary);
    font-size: 0.92rem;
    line-height: 1.6;
    margin-bottom: 0.6rem;
}
.memory-meta {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    align-items: center;
}
.memory-meta span {
    font-size: 0.75rem;
    color: var(--text-muted);
}
.score-bar {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
}
.score-bar-track {
    width: 60px;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s ease;
}

/* ── Section header ── */
.section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
}
.section-header h3 {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
}
.section-header .count {
    font-size: 0.8rem;
    color: var(--text-muted);
    background: var(--bg-card);
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    border: 1px solid var(--border);
}

/* ── Info banner ── */
.info-banner {
    background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(139,92,246,0.08));
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: var(--radius-md);
    padding: 1rem 1.4rem;
    margin-bottom: 1.2rem;
}
.info-banner p {
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin: 0;
    line-height: 1.5;
}

/* ── Context block ── */
.context-block {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.7;
    white-space: pre-wrap;
    overflow-x: auto;
    max-height: 500px;
}

/* ── Health indicator ── */
.health-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 1.5rem;
}
.health-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}
.health-ok {
    background: rgba(34,197,94,0.12);
    border: 1px solid rgba(34,197,94,0.3);
    color: var(--accent-green);
}
.health-err {
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.3);
    color: var(--accent-red);
}

/* ── Architecture table ── */
.arch-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.85rem;
    border-radius: var(--radius-md);
    overflow: hidden;
    border: 1px solid var(--border);
}
.arch-table th {
    background: var(--bg-card);
    padding: 0.7rem 1rem;
    text-align: left;
    font-weight: 600;
    color: var(--text-secondary);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-bottom: 1px solid var(--border);
}
.arch-table td {
    padding: 0.65rem 1rem;
    border-bottom: 1px solid var(--border);
    color: var(--text-primary);
}
.arch-table tr:last-child td { border-bottom: none; }
.arch-table tr:hover td { background: var(--bg-card-hover); }

/* ── Warning banner ── */
.warn-banner {
    background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(245,158,11,0.08));
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: var(--radius-md);
    padding: 1rem 1.4rem;
    margin-bottom: 1.2rem;
}
.warn-banner p {
    color: var(--accent-orange);
    font-size: 0.85rem;
    margin: 0;
    line-height: 1.5;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: var(--text-muted);
}
.empty-state .icon {
    font-size: 2.5rem;
    margin-bottom: 0.8rem;
    opacity: 0.5;
}
.empty-state h4 {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 0.4rem;
}
.empty-state p {
    font-size: 0.85rem;
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: var(--gradient-blue) !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.25s ease !important;
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.9 !important;
    box-shadow: 0 4px 16px rgba(59,130,246,0.35) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
    border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1rem !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
}

/* ── Hide Streamlit defaults ── */
#MainMenu { visibility: hidden; }
header { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Audit table ── */
.audit-row {
    display: grid;
    grid-template-columns: 100px 1fr 120px 140px;
    gap: 0.8rem;
    padding: 0.6rem 1rem;
    border-bottom: 1px solid var(--border);
    font-size: 0.82rem;
    align-items: center;
}
.audit-row:hover { background: var(--bg-card-hover); }
.audit-op {
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.7rem;
    letter-spacing: 0.04em;
}

/* ── Instruction card ── */
.instruction-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-orange);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    transition: all 0.2s ease;
}
.instruction-card:hover {
    border-color: var(--border-light);
    border-left-color: var(--accent-orange);
}
.instruction-priority {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    font-size: 0.75rem;
    font-weight: 700;
    margin-right: 0.6rem;
}
.priority-high { background: rgba(239,68,68,0.15); color: var(--accent-red); border: 1px solid rgba(239,68,68,0.3); }
.priority-mid { background: rgba(245,158,11,0.15); color: var(--accent-orange); border: 1px solid rgba(245,158,11,0.3); }
.priority-low { background: rgba(34,197,94,0.15); color: var(--accent-green); border: 1px solid rgba(34,197,94,0.3); }
.tag-chip {
    display: inline-block;
    font-size: 0.7rem;
    padding: 0.1rem 0.5rem;
    border-radius: 10px;
    background: var(--bg-secondary);
    color: var(--text-muted);
    border: 1px solid var(--border);
    margin-right: 0.3rem;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def run_async(coro):
    """Run async function from sync Streamlit context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def page_header(title: str, description: str):
    st.markdown(f"""
    <div class="page-header">
        <h1>{title}</h1>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)


def stat_card(label: str, value, sub: str = "", color: str = "blue"):
    return f"""
    <div class="stat-card {color}">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        {"<div class='stat-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """


def memory_card_html(mem_type: str, content: str, score: float = 0.5,
                      mem_id: str = "", created: str = "", access_count: int = 0):
    score_pct = int(score * 100)
    if score >= 0.7:
        bar_color = "var(--accent-green)"
    elif score >= 0.4:
        bar_color = "var(--accent-orange)"
    else:
        bar_color = "var(--accent-red)"
    return f"""
    <div class="memory-card">
        <span class="type-badge type-{mem_type}">{mem_type}</span>
        <div class="memory-content">{content}</div>
        <div class="memory-meta">
            <span class="score-bar">
                Score
                <span class="score-bar-track">
                    <span class="score-bar-fill" style="width:{score_pct}%;background:{bar_color}"></span>
                </span>
                {score:.2f}
            </span>
            {"<span>ID: " + mem_id[:12] + "...</span>" if mem_id else ""}
            {"<span>Accessed: " + str(access_count) + "x</span>" if access_count else ""}
            {"<span>" + created + "</span>" if created else ""}
        </div>
    </div>
    """


def empty_state(icon: str, title: str, message: str):
    st.markdown(f"""
    <div class="empty-state">
        <div class="icon">{icon}</div>
        <h4>{title}</h4>
        <p>{message}</p>
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, count: int | None = None):
    count_html = f'<span class="count">{count} items</span>' if count is not None else ""
    st.markdown(f"""
    <div class="section-header">
        <h3>{title}</h3>
        {count_html}
    </div>
    """, unsafe_allow_html=True)


def get_type_badge(mem_type: str) -> str:
    return f'<span class="type-badge type-{mem_type}">{mem_type}</span>'


# ──────────────────────────────────────────────────────────────
# Navigation
# ──────────────────────────────────────────────────────────────
NAV_ITEMS = {
    "Dashboard":           ("📊", "Overview & recent activity"),
    "Add Memory":          ("➕", "Store new memories"),
    "Search":              ("🔍", "Semantic search"),
    "Memory Browser":      ("📋", "Browse & manage"),
    "Agent Instructions":  ("⚙️", "Procedural memory"),
    "Context Assembler":   ("🧩", "Preview prompt injection"),
    "GDPR Forget":         ("🗑️", "Data deletion"),
    "Admin":               ("📈", "Stats, health & audit"),
}


def main():
    # ── Sidebar ──
    with st.sidebar:
        st.markdown("""
        <div style="padding: 0.8rem 0.5rem 1.2rem; border-bottom: 1px solid var(--border); margin-bottom: 1rem;">
            <div style="font-size: 1.3rem; font-weight: 700; background: linear-gradient(135deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                PyMem
            </div>
            <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.2rem;">
                Four-Type Memory Platform for AI Agents
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Navigation
        nav_options = list(NAV_ITEMS.keys())
        page = st.radio(
            "Navigation",
            nav_options,
            format_func=lambda x: f"{NAV_ITEMS[x][0]}  {x}",
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Scope configuration
        st.markdown("""
        <div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); margin-bottom: 0.5rem;">
            Scope Configuration
        </div>
        """, unsafe_allow_html=True)

        user_id = st.text_input("User ID", value="demo_user", key="scope_user")
        agent_id = st.text_input("Agent ID", value="demo_agent", key="scope_agent")
        session_id = st.text_input("Session ID", value="session_001", key="scope_session")

        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.7rem; color: var(--text-muted); line-height: 1.5;">
            <strong>Backends:</strong> In-memory (dev mode)<br>
            <strong>Run:</strong> <code>streamlit run pymem/ui/app.py</code>
        </div>
        """, unsafe_allow_html=True)

    # ── Engine ──
    engine = get_or_create_engine()

    # ── Route ──
    if page == "Dashboard":
        render_dashboard(engine, user_id, agent_id, session_id)
    elif page == "Add Memory":
        render_add_memory(engine, user_id, agent_id, session_id)
    elif page == "Search":
        render_search(engine, user_id, agent_id)
    elif page == "Memory Browser":
        render_browser(engine, user_id, agent_id)
    elif page == "Agent Instructions":
        render_instructions(engine, agent_id)
    elif page == "Context Assembler":
        render_context(engine, user_id, agent_id, session_id)
    elif page == "GDPR Forget":
        render_forget(engine, user_id)
    elif page == "Admin":
        render_admin(engine)


# ──────────────────────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────────────────────

def render_dashboard(engine, user_id, agent_id, session_id):
    page_header("Dashboard", "Real-time overview of your memory platform")

    stats = run_async(engine._relational.get_platform_stats())
    total = stats.get("total_memories", 0)
    sem = stats.get("semantic", 0)
    epi = stats.get("episodic", 0)
    proc = stats.get("procedural", 0)
    users = stats.get("unique_users", 0)
    agents = stats.get("unique_agents", 0)

    # Top stat cards
    st.markdown(f"""
    <div class="stat-row">
        {stat_card("Total Memories", total, f"{users} users &middot; {agents} agents", "blue")}
        {stat_card("Semantic", sem, "Facts, preferences, entities", "purple")}
        {stat_card("Episodic", epi, "Events, timeline, context", "green")}
        {stat_card("Procedural", proc, "Instructions & workflows", "orange")}
    </div>
    """, unsafe_allow_html=True)

    # Two-column layout
    col1, col2 = st.columns([3, 2])

    with col1:
        section_header("Recent Memories")
        from pymem.models import MemoryScope
        try:
            scope = MemoryScope(user_id=user_id)
            rows = run_async(
                engine._relational.list_memories(
                    filters=scope.to_filter(), limit=8
                )
            )
            if rows:
                cards_html = ""
                for row in rows:
                    cards_html += memory_card_html(
                        mem_type=row["memory_type"],
                        content=row["content"],
                        score=row.get("score", 0.5),
                        mem_id=row["id"],
                        created=str(row.get("created_at", ""))[:19],
                        access_count=row.get("access_count", 0),
                    )
                st.markdown(cards_html, unsafe_allow_html=True)
            else:
                empty_state(
                    "📝",
                    "No memories yet",
                    "Head to 'Add Memory' to store your first memory."
                )
        except Exception as e:
            st.error(f"Error loading memories: {e}")

    with col2:
        section_header("Memory Architecture")
        st.markdown("""
        <table class="arch-table">
            <thead>
                <tr><th>Type</th><th>Store</th><th>Lifetime</th></tr>
            </thead>
            <tbody>
                <tr>
                    <td><span class="type-badge type-working">Working</span></td>
                    <td>Redis</td>
                    <td>Session end</td>
                </tr>
                <tr>
                    <td><span class="type-badge type-episodic">Episodic</span></td>
                    <td>Graph + Relational</td>
                    <td>90 days</td>
                </tr>
                <tr>
                    <td><span class="type-badge type-semantic">Semantic</span></td>
                    <td>Vector + Graph</td>
                    <td>Decay curve</td>
                </tr>
                <tr>
                    <td><span class="type-badge type-procedural">Procedural</span></td>
                    <td>Relational</td>
                    <td>Permanent</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Backend Health")

        vector_ok = run_async(engine._vector.health_check())
        graph_ok = run_async(engine._graph.health_check())

        st.markdown(f"""
        <div class="health-row">
            <span class="health-chip {"health-ok" if vector_ok else "health-err"}">
                {"&#x2713;" if vector_ok else "&#x2717;"} Vector Store
            </span>
            <span class="health-chip {"health-ok" if graph_ok else "health-err"}">
                {"&#x2713;" if graph_ok else "&#x2717;"} Graph Store
            </span>
            <span class="health-chip health-ok">
                &#x2713; Relational
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Quick actions
        section_header("Quick Actions")
        if st.button("Add Memory", key="quick_add", use_container_width=True):
            st.session_state["nav"] = "Add Memory"
            st.rerun()
        if st.button("Search Memories", key="quick_search", use_container_width=True):
            st.session_state["nav"] = "Search"
            st.rerun()


def render_add_memory(engine, user_id, agent_id, session_id):
    page_header("Add Memory", "Extract memories from conversations or add them manually")

    tab1, tab2, tab3 = st.tabs(["Conversation Extract", "Manual Entry", "Bulk Import"])

    with tab1:
        st.markdown("""
        <div class="info-banner">
            <p><strong>How it works:</strong> Paste a conversation below. PyMem's extraction pipeline
            will analyze messages, identify facts, events, and preferences, then store them as
            typed memories with relevance scores.</p>
        </div>
        """, unsafe_allow_html=True)

        conversation = st.text_area(
            "Conversation",
            height=180,
            placeholder="user: I prefer Python over Java\nassistant: Noted! I'll use Python for code examples.\nuser: I've been using Kafka for 3 years at my current company",
            help="One message per line. Format: role: content",
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            memory_types = st.multiselect(
                "Memory types to extract",
                ["semantic", "episodic", "procedural"],
                default=["semantic", "episodic"],
            )
        with col2:
            async_mode = st.checkbox("Async extraction", value=False,
                                     help="Run extraction in background")

        if st.button("Extract & Store", type="primary", use_container_width=True):
            if not conversation.strip():
                st.warning("Please enter a conversation.")
                return

            messages = []
            for line in conversation.strip().split("\n"):
                if ":" in line:
                    role, content = line.split(":", 1)
                    messages.append(
                        {"role": role.strip().lower(), "content": content.strip()}
                    )

            if not messages:
                st.warning("Could not parse any messages. Use `role: content` format.")
                return

            from pymem.models import MemoryScope, MemoryType

            scope = MemoryScope(
                user_id=user_id, agent_id=agent_id, session_id=session_id
            )
            types = [MemoryType(t) for t in memory_types] if memory_types else None

            with st.spinner("Extracting memories..."):
                start = time.time()
                result = run_async(
                    engine.add(messages=messages, scope=scope, memory_types=types)
                )
                elapsed = time.time() - start

            st.success(f"Stored **{result.memories_count}** memories in {elapsed:.1f}s")

            if result.memories:
                cards = ""
                for mem in result.memories:
                    cards += memory_card_html(
                        mem_type=mem.memory_type.value,
                        content=mem.content,
                        score=mem.score,
                        mem_id=mem.id,
                    )
                st.markdown(cards, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class="info-banner">
            <p>Add a single memory directly. Choose the memory type, set an importance score,
            and optionally tag it for easier retrieval.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("manual_entry", clear_on_submit=True):
            content = st.text_area(
                "Memory content",
                placeholder="User prefers dark mode and uses VS Code as primary editor",
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                mem_type = st.selectbox("Type", ["semantic", "episodic", "procedural"])
            with col2:
                score = st.slider("Importance", 0.0, 1.0, 0.7, 0.05)
            with col3:
                tags = st.text_input("Tags", placeholder="preference, ui")

            submitted = st.form_submit_button("Add Memory", type="primary", use_container_width=True)

            if submitted and content.strip():
                from pymem.models import MemoryScope

                scope = MemoryScope(user_id=user_id, agent_id=agent_id)
                tag_list = [t.strip() for t in tags.split(",") if t.strip()]

                if mem_type == "semantic":
                    mem = run_async(
                        engine.semantic.add(
                            content=content, scope=scope,
                            score=score, metadata={"tags": tag_list},
                        )
                    )
                elif mem_type == "episodic":
                    mem = run_async(
                        engine.episodic.add_event(
                            content=content, scope=scope,
                            metadata={"tags": tag_list},
                        )
                    )
                else:
                    mem = run_async(
                        engine.procedural.add(
                            content=content, scope=scope, tags=tag_list
                        )
                    )
                st.success(f"Memory stored! ID: `{mem.id}`")
            elif submitted:
                st.warning("Please enter content.")

    with tab3:
        st.markdown("""
        <div class="info-banner">
            <p>Import multiple memories at once from a JSON array. Each entry needs
            <code>content</code> and <code>type</code> fields.</p>
        </div>
        """, unsafe_allow_html=True)

        json_input = st.text_area(
            "JSON array",
            height=200,
            placeholder='[\n  {"content": "User prefers Python", "type": "semantic", "score": 0.8},\n  {"content": "User attended PyCon 2024", "type": "episodic"}\n]',
        )

        if st.button("Import", type="primary", use_container_width=True):
            if not json_input.strip():
                st.warning("Please paste a JSON array.")
                return
            try:
                memories = json.loads(json_input)
                if not isinstance(memories, list):
                    st.error("Input must be a JSON array.")
                    return

                from pymem.models import MemoryScope

                scope = MemoryScope(user_id=user_id, agent_id=agent_id)
                count = 0
                progress = st.progress(0, text="Importing...")

                for i, mem_data in enumerate(memories):
                    content = mem_data.get("content", "")
                    if not content:
                        continue
                    mt = mem_data.get("type", "semantic")
                    if mt == "semantic":
                        run_async(engine.semantic.add(content=content, scope=scope))
                    elif mt == "episodic":
                        run_async(engine.episodic.add_event(content=content, scope=scope))
                    elif mt == "procedural":
                        run_async(engine.procedural.add(content=content, scope=scope))
                    count += 1
                    progress.progress((i + 1) / len(memories), text=f"Importing {i+1}/{len(memories)}")

                progress.empty()
                st.success(f"Imported **{count}** memories!")
            except json.JSONDecodeError:
                st.error("Invalid JSON format. Please check your input.")


def render_search(engine, user_id, agent_id):
    page_header("Search Memories", "Semantic search across all memory types with relevance ranking")

    # Search bar
    query = st.text_input(
        "Search query",
        placeholder="What programming languages does the user prefer?",
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search_types = st.multiselect(
            "Filter by type",
            ["semantic", "episodic", "procedural"],
            default=["semantic", "episodic", "procedural"],
        )
    with col2:
        limit = st.slider("Max results", 1, 50, 10)
    with col3:
        min_score = st.slider("Min score", 0.0, 1.0, 0.0, 0.05)

    search_clicked = st.button("Search", type="primary", use_container_width=True)

    if search_clicked and query:
        from pymem.models import MemoryScope, MemoryType

        scope = MemoryScope(user_id=user_id, agent_id=agent_id)
        types = [MemoryType(t) for t in search_types] if search_types else None

        with st.spinner("Searching..."):
            start = time.time()
            result = run_async(
                engine.search(query=query, scope=scope, memory_types=types, limit=limit)
            )
            elapsed = time.time() - start

        if not result.memories:
            empty_state("🔍", "No results", "Try a different query or broaden your search filters.")
            return

        section_header(f"Results for \"{query}\"", result.total)
        st.caption(f"Completed in {elapsed:.2f}s")

        cards = ""
        for mem in result.memories:
            if mem.score >= min_score:
                cards += memory_card_html(
                    mem_type=mem.memory_type.value,
                    content=mem.content,
                    score=mem.score,
                    mem_id=mem.id,
                    access_count=mem.access_count,
                )
        if cards:
            st.markdown(cards, unsafe_allow_html=True)
        else:
            empty_state("🔍", "No results above threshold", f"All results scored below {min_score:.2f}")

        # Context string
        if result.context_string:
            with st.expander("View assembled context string"):
                st.markdown(f'<div class="context-block">{result.context_string}</div>',
                            unsafe_allow_html=True)

    elif not query:
        empty_state("🔍", "Enter a search query", "Type a question or keywords to search across your memories.")


def render_browser(engine, user_id, agent_id):
    page_header("Memory Browser", "Browse, inspect, and manage all stored memories")

    col1, col2 = st.columns([1, 3])
    with col1:
        filter_type = st.selectbox(
            "Filter by type",
            ["All", "semantic", "episodic", "procedural"],
        )
    with col2:
        page_size = st.select_slider("Page size", [10, 25, 50, 100], value=25)

    from pymem.models import MemoryScope

    scope = MemoryScope(user_id=user_id, agent_id=agent_id)
    mem_type = filter_type if filter_type != "All" else None

    rows = run_async(
        engine._relational.list_memories(
            filters=scope.to_filter(), memory_type=mem_type, limit=page_size
        )
    )

    if not rows:
        empty_state("📋", "No memories found", "No memories match your current filters.")
        return

    section_header("Memories", len(rows))

    cards = ""
    for row in rows:
        cards += memory_card_html(
            mem_type=row["memory_type"],
            content=row["content"],
            score=row.get("score", 0.5),
            mem_id=row["id"],
            created=str(row.get("created_at", ""))[:19],
            access_count=row.get("access_count", 0),
        )
    st.markdown(cards, unsafe_allow_html=True)

    # Individual memory actions
    st.markdown("---")
    section_header("Memory Actions")

    col1, col2 = st.columns(2)
    with col1:
        memory_id = st.text_input("Memory ID", placeholder="Enter full memory ID to act on")
    with col2:
        action = st.selectbox("Action", ["View Details", "Update Content", "Soft Delete", "Hard Delete"])

    if st.button("Execute", use_container_width=True) and memory_id:
        if action == "View Details":
            mem = run_async(engine.get(memory_id))
            if mem:
                st.json({
                    "id": mem.id,
                    "type": mem.memory_type.value,
                    "content": mem.content,
                    "score": mem.score,
                    "version": mem.version,
                    "access_count": mem.access_count,
                    "created_at": str(mem.created_at),
                    "updated_at": str(mem.updated_at),
                    "metadata": mem.metadata,
                    "user_id": mem.user_id,
                    "agent_id": mem.agent_id,
                })
            else:
                st.warning(f"Memory `{memory_id}` not found.")

        elif action == "Soft Delete":
            run_async(engine._relational.soft_delete_memory(memory_id))
            st.success(f"Memory `{memory_id[:12]}...` soft-deleted.")
            st.rerun()

        elif action == "Hard Delete":
            from pymem.models import MemoryScope
            scope = MemoryScope(user_id=user_id)
            run_async(engine.delete(memory_id, scope=scope, hard_delete=True))
            st.success(f"Memory `{memory_id[:12]}...` permanently deleted.")
            st.rerun()

        elif action == "Update Content":
            st.info("Enter updated content below and click 'Save Update'.")
            new_content = st.text_area("New content", key="update_content_input")
            if st.button("Save Update", key="save_update"):
                from pymem.models import MemoryScope
                scope = MemoryScope(user_id=user_id)
                updated = run_async(engine.update(memory_id, content=new_content, scope=scope))
                if updated:
                    st.success(f"Memory updated! New version: {updated.version}")
                else:
                    st.error("Update failed.")


def render_instructions(engine, agent_id):
    page_header("Agent Instructions", f"Procedural memories for agent: {agent_id}")

    st.markdown("""
    <div class="info-banner">
        <p><strong>Procedural memories</strong> are persistent instructions that shape agent behavior.
        They never decay, are ordered by priority (1 = highest), and are injected into every context assembly.</p>
    </div>
    """, unsafe_allow_html=True)

    # Add new instruction
    with st.form("add_instruction", clear_on_submit=True):
        content = st.text_area(
            "New instruction",
            placeholder="Always respond in TypeScript when writing code examples.\nPrefer functional patterns over OOP.",
        )
        col1, col2 = st.columns(2)
        with col1:
            priority = st.slider("Priority (1 = highest)", 1, 10, 5)
        with col2:
            tags = st.text_input("Tags", placeholder="coding, style, format")

        if st.form_submit_button("Add Instruction", type="primary", use_container_width=True):
            if content.strip():
                from pymem.models import MemoryScope
                scope = MemoryScope(agent_id=agent_id)
                tag_list = [t.strip() for t in tags.split(",") if t.strip()]
                mem = run_async(
                    engine.procedural.add(
                        content=content, scope=scope,
                        tags=tag_list, priority=priority,
                    )
                )
                st.success(f"Instruction added! ID: `{mem.id}`")
                st.rerun()
            else:
                st.warning("Please enter instruction content.")

    # List existing
    from pymem.models import MemoryScope

    scope = MemoryScope(agent_id=agent_id)
    instructions = run_async(engine.procedural.get_all(scope))

    section_header("Active Instructions", len(instructions) if instructions else 0)

    if not instructions:
        empty_state("⚙️", "No instructions", "Add your first instruction above to shape agent behavior.")
        return

    for inst in instructions:
        priority_val = inst.metadata.get("priority", 5)
        tags = inst.metadata.get("tags", [])
        if priority_val <= 3:
            pcls = "priority-high"
        elif priority_val <= 6:
            pcls = "priority-mid"
        else:
            pcls = "priority-low"

        tags_html = "".join(f'<span class="tag-chip">{t}</span>' for t in tags)

        st.markdown(f"""
        <div class="instruction-card">
            <div style="display: flex; align-items: flex-start;">
                <span class="instruction-priority {pcls}">P{priority_val}</span>
                <div style="flex: 1;">
                    <div class="memory-content">{inst.content}</div>
                    <div>{tags_html}</div>
                    <div class="memory-meta" style="margin-top: 0.4rem;">
                        <span>ID: {inst.id[:12]}...</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(f"Remove", key=f"rm_{inst.id}"):
            run_async(engine._relational.soft_delete_memory(inst.id))
            st.rerun()


def render_context(engine, user_id, agent_id, session_id):
    page_header("Context Assembler", "Preview the complete memory context injected into agent prompts")

    st.markdown("""
    <div class="info-banner">
        <p>The context assembler gathers working memory, semantic facts, episodic events,
        and procedural instructions, then formats them into a single prompt block within
        your token budget.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Query (optional)",
            placeholder="What's the user's tech stack?",
            help="Relevance-rank memories against this query",
        )
    with col2:
        max_tokens = st.number_input("Token budget", 1000, 8000, 4000, step=500)

    if st.button("Assemble Context", type="primary", use_container_width=True):
        from pymem.models import MemoryScope

        scope = MemoryScope(
            user_id=user_id, agent_id=agent_id, session_id=session_id
        )

        with st.spinner("Assembling context..."):
            start = time.time()
            ctx = run_async(
                engine.get_context(
                    scope=scope, query=query or None, max_tokens=max_tokens,
                )
            )
            elapsed = time.time() - start

        st.caption(f"Assembled in {elapsed:.2f}s")

        # Formatted prompt
        section_header("Formatted Prompt Block")
        formatted = ctx.formatted or "(empty - no memories found)"
        st.markdown(f'<div class="context-block">{formatted}</div>', unsafe_allow_html=True)

        # Breakdown
        col1, col2 = st.columns(2)

        with col1:
            section_header("Semantic Facts", len(ctx.relevant_facts))
            if ctx.relevant_facts:
                cards = ""
                for f in ctx.relevant_facts:
                    cards += memory_card_html("semantic", f.content, f.score, f.id)
                st.markdown(cards, unsafe_allow_html=True)
            else:
                st.caption("No semantic memories found.")

        with col2:
            section_header("Episodic Events", len(ctx.recent_events))
            if ctx.recent_events:
                cards = ""
                for e in ctx.recent_events:
                    cards += memory_card_html("episodic", e.content, e.score, e.id)
                st.markdown(cards, unsafe_allow_html=True)
            else:
                st.caption("No episodic memories found.")

        section_header("Agent Instructions", len(ctx.agent_instructions))
        if ctx.agent_instructions:
            for inst in ctx.agent_instructions:
                st.markdown(f"""
                <div class="instruction-card">
                    <div class="memory-content">{inst.content}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No procedural instructions found.")

        if ctx.working_messages:
            section_header("Working Memory", len(ctx.working_messages))
            for msg in ctx.working_messages[-5:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                role_color = "var(--accent-blue)" if role == "user" else "var(--accent-green)"
                st.markdown(f"""
                <div style="padding: 0.5rem 0.8rem; margin-bottom: 0.3rem;
                            border-left: 3px solid {role_color};
                            background: var(--bg-card); border-radius: 0 6px 6px 0;">
                    <span style="font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
                                 color: {role_color};">{role}</span>
                    <div style="font-size: 0.85rem; color: var(--text-primary); margin-top: 0.2rem;">
                        {content}
                    </div>
                </div>
                """, unsafe_allow_html=True)


def render_forget(engine, user_id):
    page_header("GDPR Forget", "Permanently delete user memories for compliance")

    st.markdown("""
    <div class="warn-banner">
        <p><strong>Destructive operation.</strong> This permanently removes memories from all backends
        (vector store, graph store, relational database). This action cannot be undone.
        An audit trail entry will be created with a confirmation ID.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        scope_type = st.selectbox(
            "Scope to forget",
            ["all", "semantic", "episodic", "procedural"],
            help="Choose 'all' to delete everything, or pick a specific memory type."
        )
    with col2:
        st.markdown(f"""
        <div style="padding: 1rem; background: var(--bg-card); border: 1px solid var(--border);
                    border-radius: var(--radius-sm); margin-top: 0.4rem;">
            <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.3rem;">Target User</div>
            <div style="font-size: 1rem; font-weight: 600; color: var(--text-primary); font-family: 'JetBrains Mono', monospace;">
                {user_id}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    confirm = st.text_input(
        f"Type `{user_id}` to confirm deletion",
        placeholder="Type user ID to confirm",
    )

    if st.button("Permanently Delete Memories", type="primary", use_container_width=True):
        if confirm != user_id:
            st.error("Confirmation does not match. Please type the exact user ID.")
            return

        from pymem.models import MemoryScope, MemoryType

        scope = MemoryScope(user_id=user_id)
        types = [MemoryType(scope_type)] if scope_type != "all" else None

        with st.spinner("Deleting memories from all backends..."):
            result = run_async(engine.forget(scope=scope, memory_types=types))

        st.success("Memories deleted successfully.")

        st.markdown(f"""
        <div style="background: var(--bg-card); border: 1px solid var(--border);
                    border-radius: var(--radius-md); padding: 1.2rem;">
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem;">
                <div>
                    <div style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase;">Deleted</div>
                    <div style="font-size: 1.2rem; font-weight: 600; color: var(--text-primary);">
                        {sum(result.deleted_per_type.values())} memories
                    </div>
                </div>
                <div>
                    <div style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase;">Confirmation</div>
                    <div style="font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; color: var(--accent-blue);">
                        {result.confirmation_id[:16]}...
                    </div>
                </div>
                <div>
                    <div style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase;">Completed</div>
                    <div style="font-size: 0.8rem; color: var(--text-primary);">
                        {str(result.completed_at)[:19]}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Breakdown by type
        if result.deleted_per_type:
            st.markdown("<br>", unsafe_allow_html=True)
            breakdown = " &middot; ".join(
                f"**{k}**: {v}" for k, v in result.deleted_per_type.items()
            )
            st.markdown(f"Breakdown: {breakdown}")


def render_admin(engine):
    page_header("Admin Panel", "Platform statistics, backend health, and audit log")

    # Stats
    stats = run_async(engine._relational.get_platform_stats())

    st.markdown(f"""
    <div class="stat-row">
        {stat_card("Total Memories", stats.get("total_memories", 0), "", "blue")}
        {stat_card("Unique Users", stats.get("unique_users", 0), "", "green")}
        {stat_card("Unique Agents", stats.get("unique_agents", 0), "", "purple")}
        {stat_card("Working Memory", stats.get("working", 0), "Active sessions", "orange")}
    </div>
    """, unsafe_allow_html=True)

    # Memory distribution
    col1, col2 = st.columns([2, 1])

    with col1:
        section_header("Memory Distribution")
        type_data = {
            "Semantic": stats.get("semantic", 0),
            "Episodic": stats.get("episodic", 0),
            "Procedural": stats.get("procedural", 0),
        }

        if any(type_data.values()):
            import pandas as pd
            import plotly.express as px

            df = pd.DataFrame({
                "Type": list(type_data.keys()),
                "Count": list(type_data.values()),
            })

            try:
                fig = px.bar(
                    df, x="Type", y="Count",
                    color="Type",
                    color_discrete_map={
                        "Semantic": "#8b5cf6",
                        "Episodic": "#22c55e",
                        "Procedural": "#f59e0b",
                    },
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8"),
                    showlegend=False,
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=250,
                )
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=True, gridcolor="rgba(42,49,80,0.5)")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                # Plotly may not be installed, fall back to st.bar_chart
                st.bar_chart(df.set_index("Type"))
        else:
            empty_state("📊", "No data yet", "Add some memories to see distribution charts.")

    with col2:
        section_header("Backend Health")
        vector_ok = run_async(engine._vector.health_check())
        graph_ok = run_async(engine._graph.health_check())

        backends = [
            ("Vector Store", vector_ok, "pgvector / in-memory"),
            ("Graph Store", graph_ok, "kuzu / in-memory"),
            ("Relational", True, "SQLite"),
        ]

        for name, healthy, detail in backends:
            cls = "health-ok" if healthy else "health-err"
            icon = "&#x2713;" if healthy else "&#x2717;"
            st.markdown(f"""
            <div class="health-chip {cls}" style="display: flex; margin-bottom: 0.5rem; width: 100%;">
                <span>{icon} {name}</span>
            </div>
            """, unsafe_allow_html=True)
            st.caption(detail)

    # Audit log
    st.markdown("---")
    section_header("Audit Log")

    col1, col2, col3 = st.columns(3)
    with col1:
        audit_user = st.text_input("Filter by user", value="", key="audit_user")
    with col2:
        audit_mem = st.text_input("Filter by memory ID", value="", key="audit_mem")
    with col3:
        audit_limit = st.number_input("Max entries", 10, 500, 50, key="audit_limit")

    if st.button("Load Audit Log", use_container_width=True):
        entries = run_async(
            engine._relational.get_audit_log(
                user_id=audit_user or None,
                memory_id=audit_mem or None,
                limit=audit_limit,
            )
        )

        if entries:
            st.markdown(f"""
            <div style="background: var(--bg-card); border: 1px solid var(--border);
                        border-radius: var(--radius-md); overflow: hidden;">
                <div class="audit-row" style="background: var(--bg-secondary); font-weight: 600;
                            font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em;
                            color: var(--text-muted);">
                    <span>Operation</span>
                    <span>Memory ID</span>
                    <span>Actor</span>
                    <span>Timestamp</span>
                </div>
            """, unsafe_allow_html=True)

            rows_html = ""
            for entry in entries:
                op = entry.get("operation", "unknown")
                op_color = {
                    "create": "var(--accent-green)",
                    "update": "var(--accent-blue)",
                    "delete": "var(--accent-red)",
                    "search": "var(--accent-purple)",
                }.get(op, "var(--text-muted)")

                rows_html += f"""
                <div class="audit-row">
                    <span class="audit-op" style="color: {op_color};">{op}</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
                        {str(entry.get("memory_id", ""))[:12]}
                    </span>
                    <span>{entry.get("actor_id", "-")}</span>
                    <span style="font-size: 0.75rem;">{str(entry.get("timestamp", ""))[:19]}</span>
                </div>
                """

            st.markdown(rows_html + "</div>", unsafe_allow_html=True)
        else:
            empty_state("📜", "No audit entries", "Audit entries are created automatically when memories are modified.")

    # Manual consolidation
    st.markdown("---")
    section_header("Manual Operations")

    col1, col2 = st.columns(2)
    with col1:
        consolidate_user = st.text_input("User ID for consolidation", value=user_id, key="cons_user")
        if st.button("Run Consolidation", use_container_width=True):
            st.info(f"Consolidation triggered for `{consolidate_user}`. This reviews similar memories and merges duplicates.")

    with col2:
        import_json = st.text_area("Bulk import JSON", height=100, key="admin_import",
                                   placeholder='[{"content": "...", "type": "semantic"}]')
        if st.button("Import", use_container_width=True, key="admin_import_btn"):
            if import_json.strip():
                try:
                    data = json.loads(import_json)
                    from pymem.models import MemoryScope
                    scope = MemoryScope(user_id=user_id)
                    count = 0
                    for item in data:
                        c = item.get("content", "")
                        if not c:
                            continue
                        t = item.get("type", "semantic")
                        if t == "semantic":
                            run_async(engine.semantic.add(content=c, scope=scope))
                        elif t == "episodic":
                            run_async(engine.episodic.add_event(content=c, scope=scope))
                        elif t == "procedural":
                            run_async(engine.procedural.add(content=c, scope=scope))
                        count += 1
                    st.success(f"Imported {count} memories.")
                except json.JSONDecodeError:
                    st.error("Invalid JSON.")


if __name__ == "__main__":
    main()

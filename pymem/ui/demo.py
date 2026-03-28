"""PyMem Interactive Demo — Learn all 4 memory types with real examples.

Run: streamlit run pymem/ui/demo.py
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime

import streamlit as st

from pymem.ui.engine_factory import get_or_create_engine
from pymem.models import MemoryScope, MemoryType

# ── Page config ──
st.set_page_config(
    page_title="PyMem Demo — Learn Memory Types",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp { font-family: 'Inter', sans-serif; }

/* Cards */
.demo-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.demo-card h3 { margin: 0 0 0.5rem 0; color: #1e293b; }
.demo-card p { color: #475569; margin: 0; line-height: 1.6; }

/* Type headers */
.type-header {
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    color: white;
}
.type-header h2 { margin: 0; font-size: 1.4rem; }
.type-header p { margin: 0.3rem 0 0; opacity: 0.9; font-size: 0.9rem; }

.th-working { background: linear-gradient(135deg, #7c3aed, #a78bfa); }
.th-semantic { background: linear-gradient(135deg, #2563eb, #3b82f6); }
.th-episodic { background: linear-gradient(135deg, #059669, #10b981); }
.th-procedural { background: linear-gradient(135deg, #d97706, #f59e0b); }

/* Result card */
.result-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
    color: #1e293b;
}
.result-card .meta { color: #64748b; font-size: 0.78rem; margin-top: 0.4rem; }

/* Analogy box */
.analogy {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.analogy strong { color: #92400e; }
.analogy p { color: #78350f; margin: 0; }

/* Context block */
.ctx-block {
    background: #1e293b;
    color: #e2e8f0;
    border-radius: 10px;
    padding: 1.2rem;
    font-family: monospace;
    font-size: 0.82rem;
    white-space: pre-wrap;
    line-height: 1.6;
    max-height: 500px;
    overflow-y: auto;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #1e293b !important;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stRadio label:hover { background: #334155 !important; }

/* Hide defaults */
#MainMenu, header, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──
def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def show_result(content, mem_type="", score=None, mem_id=""):
    score_text = f" | Score: {score:.2f}" if score is not None else ""
    id_text = f" | ID: {mem_id[:10]}..." if mem_id else ""
    st.markdown(f"""
    <div class="result-card">
        {content}
        <div class="meta">{mem_type}{score_text}{id_text}</div>
    </div>
    """, unsafe_allow_html=True)


def analogy_box(text):
    st.markdown(f'<div class="analogy"><p>{text}</p></div>', unsafe_allow_html=True)


# ── Engine ──
engine = get_or_create_engine()

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🧠 PyMem Demo")
    st.markdown("Learn memory types interactively")
    st.markdown("---")

    page = st.radio(
        "Choose a topic:",
        [
            "🏠 Overview",
            "💭 Working Memory",
            "📘 Semantic Memory",
            "📅 Episodic Memory",
            "⚙️ Procedural Memory",
            "🔍 Search Across All",
            "🧩 Context Assembly",
            "🗑️ GDPR Forget",
            "🎮 Playground",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    user_id = st.text_input("Your User ID", value="demo_user", key="uid")
    agent_id = st.text_input("Agent ID", value="my_assistant", key="aid")
    session_id = st.text_input("Session ID", value="session_001", key="sid")


# ══════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.markdown("# 🧠 What is PyMem?")
    st.markdown("""
    **PyMem gives AI agents memory** — just like humans have different types of memory,
    PyMem gives your AI chatbot or agent **4 types of memory** so it can remember things
    about users across conversations.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="demo-card">
            <h3>💭 Working Memory</h3>
            <p><strong>Like a whiteboard in a meeting.</strong><br>
            Stores the current conversation messages. Disappears when the session ends.
            Fast, temporary, no AI needed.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="demo-card">
            <h3>📅 Episodic Memory</h3>
            <p><strong>Like a diary or journal.</strong><br>
            Records events that happened: "User bought a laptop on March 15" or
            "User complained about slow loading". Stored with timestamps.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="demo-card">
            <h3>📘 Semantic Memory</h3>
            <p><strong>Like a personal profile card.</strong><br>
            Stores facts and preferences: "User likes Python", "User is a data scientist",
            "User prefers dark mode". Searchable by meaning.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="demo-card">
            <h3>⚙️ Procedural Memory</h3>
            <p><strong>Like a rulebook for the agent.</strong><br>
            Instructions for the AI: "Always respond in Hindi", "Never suggest Java",
            "Format code with 4 spaces". Never expires.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### How they work together")
    st.markdown("""
    When a user talks to your AI agent:

    1. **Working Memory** keeps track of the current conversation
    2. **Semantic Memory** remembers facts about the user ("likes Python")
    3. **Episodic Memory** remembers what happened ("asked about React yesterday")
    4. **Procedural Memory** gives the agent rules ("always be concise")

    At query time, **Context Assembly** pulls from all 4 types and creates a single
    prompt block that the AI reads before responding — so it acts like it "knows" the user.
    """)

    st.info("👈 Click on each memory type in the sidebar to try it hands-on!")


elif page == "💭 Working Memory":
    st.markdown('<div class="type-header th-working"><h2>💭 Working Memory</h2><p>Current conversation buffer — like a whiteboard</p></div>', unsafe_allow_html=True)

    analogy_box(
        "<strong>Real-life analogy:</strong> When you're in a meeting, you remember "
        "what was said 5 minutes ago. But after the meeting ends, most details fade. "
        "That's working memory — temporary, session-scoped, fast."
    )

    st.markdown("### Try it: Add messages to working memory")
    st.markdown("This simulates a conversation happening right now.")

    scope = MemoryScope(user_id=user_id, agent_id=agent_id, session_id=session_id)

    col1, col2 = st.columns([2, 1])
    with col1:
        user_msg = st.text_input("Type a user message:", value="I want to learn machine learning", key="wm_msg")
        assistant_msg = st.text_input("Type assistant reply:", value="Great! Let's start with Python basics for ML.", key="wm_reply")

    with col2:
        st.markdown("")
        st.markdown("")
        if st.button("Send to Working Memory", type="primary", use_container_width=True, key="wm_send"):
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
            run_async(engine.working.append(messages, scope))
            st.success("Messages added to working memory!")

    st.markdown("---")
    st.markdown("### Current Working Memory Window")

    if st.button("Load current conversation", use_container_width=True, key="wm_load"):
        msgs = run_async(engine.working.get_window(scope))
        if msgs:
            for m in msgs:
                role = m.get("role", "user")
                icon = "🧑" if role == "user" else "🤖"
                st.markdown(f"**{icon} {role.title()}:** {m.get('content', '')}")
        else:
            st.warning("Working memory is empty. Send some messages first!")

    length = run_async(engine.working.get_length(session_id))
    st.caption(f"Messages in current session: {length}")

    st.markdown("---")
    st.markdown("### Key points")
    st.markdown("""
    - Stored per **session** (each conversation gets its own)
    - **Auto-expires** after 1 hour (configurable)
    - **No AI/embedding needed** — just stores raw messages
    - Used for: keeping context within a conversation
    """)


elif page == "📘 Semantic Memory":
    st.markdown('<div class="type-header th-semantic"><h2>📘 Semantic Memory</h2><p>Facts, preferences, knowledge about the user</p></div>', unsafe_allow_html=True)

    analogy_box(
        "<strong>Real-life analogy:</strong> You know that your friend likes pizza, "
        "works at Google, and speaks Hindi. These are facts you just 'know' — you don't "
        "remember exactly when you learned them. That's semantic memory."
    )

    scope = MemoryScope(user_id=user_id, agent_id=agent_id)

    st.markdown("### Try it: Store facts about a user")

    presets = {
        "Custom": "",
        "User likes Python": "User prefers Python over JavaScript for backend development",
        "User is vegetarian": "User is vegetarian and prefers Indian cuisine",
        "User works at startup": "User works at an AI startup as a machine learning engineer",
        "User learning React": "User is currently learning React for frontend development",
    }

    preset = st.selectbox("Pick an example or write your own:", list(presets.keys()), key="sem_preset")
    fact = st.text_input("Fact to remember:", value=presets[preset], key="sem_fact")

    if st.button("Store this fact", type="primary", key="sem_add"):
        if fact.strip():
            result = run_async(engine.semantic.add(content=fact, scope=scope, score=0.8))
            st.success(f"Stored! Memory ID: `{result.id[:12]}...`")

    st.markdown("---")
    st.markdown("### Search semantic memories")
    st.markdown("Search by **meaning**, not exact words. Try: 'food preferences' or 'programming language'")

    query = st.text_input("Search query:", value="what programming language", key="sem_query")
    if st.button("Search", type="primary", key="sem_search"):
        if query.strip():
            results = run_async(engine.search(query, scope, memory_types=[MemoryType.SEMANTIC], limit=5))
            if results.memories:
                for m in results.memories:
                    show_result(m.content, "semantic", m.score, m.id)
            else:
                st.warning("No results. Store some facts first!")

    st.markdown("---")
    st.markdown("### Key points")
    st.markdown("""
    - Stores **facts, preferences, and knowledge** about users
    - **Searchable by meaning** (vector similarity) — "food" finds "vegetarian"
    - Gets **auto-extracted** from conversations by AI when using `engine.add()`
    - **Decays over time** if not accessed (half-life ~100 days)
    - Used for: personalizing responses, remembering user preferences
    """)


elif page == "📅 Episodic Memory":
    st.markdown('<div class="type-header th-episodic"><h2>📅 Episodic Memory</h2><p>What happened, when, and in what context</p></div>', unsafe_allow_html=True)

    analogy_box(
        "<strong>Real-life analogy:</strong> You remember that last Tuesday you had "
        "a job interview, and it went well. You remember the sequence of events. "
        "That's episodic memory — timestamped events."
    )

    scope = MemoryScope(user_id=user_id, agent_id=agent_id)

    st.markdown("### Try it: Record events")

    presets = {
        "Custom": "",
        "User asked about pricing": "User asked about enterprise pricing plans and was interested in the team plan",
        "User completed onboarding": "User successfully completed the onboarding tutorial for the dashboard",
        "User reported a bug": "User reported that the export feature crashes when exporting more than 1000 rows",
        "User upgraded plan": "User upgraded from free tier to professional plan",
    }

    preset = st.selectbox("Pick an example or write your own:", list(presets.keys()), key="epi_preset")
    event = st.text_input("Event to record:", value=presets[preset], key="epi_event")

    if st.button("Record this event", type="primary", key="epi_add"):
        if event.strip():
            result = run_async(engine.episodic.add_event(
                content=event, scope=scope,
                metadata={"timestamp": datetime.utcnow().isoformat()}
            ))
            st.success(f"Event recorded! Memory ID: `{result.id[:12]}...`")

    st.markdown("---")
    st.markdown("### Recent events timeline")

    if st.button("Show recent events", type="primary", key="epi_recent"):
        events = run_async(engine.episodic.get_recent(scope, limit=10))
        if events:
            for e in events:
                ts = e.metadata.get("created_at", e.metadata.get("timestamp", ""))
                show_result(e.content, "episodic", e.score, e.id)
        else:
            st.warning("No events recorded yet. Add some above!")

    st.markdown("---")
    st.markdown("### Key points")
    st.markdown("""
    - Records **events with timestamps** — what happened and when
    - Stored in a **graph** (events can be linked in chains)
    - **Decays slowly** over ~200 days half-life
    - Default TTL: 90 days (configurable)
    - Used for: "Last time you asked about X...", understanding user journey
    """)


elif page == "⚙️ Procedural Memory":
    st.markdown('<div class="type-header th-procedural"><h2>⚙️ Procedural Memory</h2><p>Rules, instructions, and learned behaviors for the agent</p></div>', unsafe_allow_html=True)

    analogy_box(
        "<strong>Real-life analogy:</strong> You know how to ride a bicycle — you "
        "don't think about the steps, you just do it. For AI agents, procedural memory "
        "is like a rulebook: 'always greet in Hindi', 'use 4-space indentation'."
    )

    scope = MemoryScope(user_id=user_id, agent_id=agent_id)

    st.markdown("### Try it: Add agent instructions")

    presets = {
        "Custom": ("", []),
        "Respond in Hindi": ("Always respond to the user in Hindi language", ["language", "hindi"]),
        "Code style": ("Always use 4-space indentation and type hints in Python code", ["code", "style"]),
        "Be concise": ("Keep responses under 3 sentences unless the user asks for detail", ["style", "brevity"]),
        "No Java suggestions": ("Never suggest Java solutions. User strongly prefers Python.", ["language", "preference"]),
    }

    preset = st.selectbox("Pick an example or write your own:", list(presets.keys()), key="proc_preset")
    default_content, default_tags = presets[preset]

    instruction = st.text_input("Instruction:", value=default_content, key="proc_inst")
    tags_str = st.text_input("Tags (comma-separated):", value=", ".join(default_tags), key="proc_tags")
    priority = st.slider("Priority (1=highest, 10=lowest):", 1, 10, 5, key="proc_pri")

    if st.button("Add instruction", type="primary", key="proc_add"):
        if instruction.strip():
            tags = [t.strip() for t in tags_str.split(",") if t.strip()]
            result = run_async(engine.procedural.add(
                content=instruction, scope=scope, tags=tags, priority=priority
            ))
            st.success(f"Instruction stored! ID: `{result.id[:12]}...`")

    st.markdown("---")
    st.markdown("### All active instructions")

    if st.button("Show all instructions", type="primary", key="proc_all"):
        instructions = run_async(engine.procedural.get_all(scope))
        if instructions:
            for i, mem in enumerate(instructions, 1):
                tags = mem.metadata.get("tags", [])
                pri = mem.metadata.get("priority", 5)
                tags_html = " ".join(f"`{t}`" for t in tags) if tags else ""
                st.markdown(f"**{i}.** {mem.content}  \nPriority: **{pri}** {tags_html}")
        else:
            st.warning("No instructions yet. Add some above!")

    st.markdown("---")
    st.markdown("### Key points")
    st.markdown("""
    - Stores **rules and instructions** for the AI agent
    - **Never expires** — manual deletion only
    - **Priority-ordered** (1 = most important)
    - **Tag-based retrieval** — filter by category
    - Score is always 1.0 (never decays)
    - Used for: agent customization, user preferences for behavior
    """)


elif page == "🔍 Search Across All":
    st.markdown("# 🔍 Multi-Type Search")
    st.markdown("Search across **all memory types at once**. Results are ranked by relevance score.")

    analogy_box(
        "<strong>How it works:</strong> PyMem searches semantic (by meaning), "
        "episodic (by timeline), and procedural (by keyword) in parallel, "
        "then merges and ranks the results by score."
    )

    scope = MemoryScope(user_id=user_id, agent_id=agent_id)

    query = st.text_input("Search all memories:", value="programming language preference", key="search_all")

    col1, col2, col3 = st.columns(3)
    with col1:
        inc_semantic = st.checkbox("Semantic", value=True, key="s_sem")
    with col2:
        inc_episodic = st.checkbox("Episodic", value=True, key="s_epi")
    with col3:
        inc_procedural = st.checkbox("Procedural", value=True, key="s_proc")

    if st.button("Search", type="primary", key="search_go"):
        types = []
        if inc_semantic:
            types.append(MemoryType.SEMANTIC)
        if inc_episodic:
            types.append(MemoryType.EPISODIC)
        if inc_procedural:
            types.append(MemoryType.PROCEDURAL)

        if query.strip() and types:
            results = run_async(engine.search(query, scope, memory_types=types, limit=10))
            if results.memories:
                st.markdown(f"**Found {results.total} memories:**")
                for m in results.memories:
                    show_result(m.content, m.memory_type.value, m.score, m.id)
            else:
                st.warning("No results. Add some memories first using the other pages!")
        else:
            st.warning("Enter a query and select at least one memory type.")


elif page == "🧩 Context Assembly":
    st.markdown("# 🧩 Context Assembly")
    st.markdown("""
    This is the **most powerful feature**. It pulls from all 4 memory types and creates
    a single text block that you inject into your AI agent's system prompt.

    Your agent reads this block and **acts like it knows the user**.
    """)

    analogy_box(
        "<strong>Think of it like:</strong> Before a meeting, your assistant gives you "
        "a briefing sheet: who the person is, what they like, what happened last time, "
        "and your rules. That's what Context Assembly does for your AI agent."
    )

    scope = MemoryScope(user_id=user_id, agent_id=agent_id, session_id=session_id)

    query = st.text_input(
        "Optional: what is the user asking about? (improves relevance)",
        value="help with Python project",
        key="ctx_query",
    )

    if st.button("Assemble Context", type="primary", key="ctx_go"):
        context = run_async(engine.get_context(scope, query=query if query.strip() else None))

        if context.formatted:
            st.markdown("### Generated prompt block")
            st.markdown("This text gets injected into the agent's system prompt:")
            st.markdown(f'<div class="ctx-block">{context.formatted}</div>', unsafe_allow_html=True)

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Facts loaded", len(context.relevant_facts))
                st.metric("Events loaded", len(context.recent_events))
            with col2:
                st.metric("Instructions loaded", len(context.agent_instructions))
                st.metric("Working messages", len(context.working_messages))
        else:
            st.warning("No memories found. Add some memories using the other pages first!")

    st.markdown("---")
    st.markdown("### How to use this in your app")
    st.code("""
# In your AI agent code:
context = await engine.get_context(
    scope=MemoryScope(user_id="user123", agent_id="my_bot"),
    query="user's current question",
    max_tokens=4000,
)

# Inject into system prompt:
system_prompt = f\"\"\"
You are a helpful assistant.

{context.formatted}
\"\"\"
    """, language="python")


elif page == "🗑️ GDPR Forget":
    st.markdown("# 🗑️ GDPR Forget")
    st.markdown("""
    **Delete ALL memories** for a user — across all memory types.
    This is a compliance feature for GDPR "right to be forgotten".
    """)

    analogy_box(
        "<strong>What it does:</strong> Wipes semantic facts, episodic events, "
        "procedural instructions, working memory, AND graph relationships — "
        "everything for the specified user. Returns a confirmation ID for audit."
    )

    scope = MemoryScope(user_id=user_id)

    st.warning(f"This will delete ALL memories for user: **{user_id}**")

    col1, col2 = st.columns(2)
    with col1:
        types_to_forget = st.multiselect(
            "Memory types to delete:",
            ["semantic", "episodic", "procedural", "working"],
            default=["semantic", "episodic", "procedural", "working"],
            key="forget_types",
        )
    with col2:
        st.markdown("")

    confirm = st.checkbox("I understand this is irreversible", key="forget_confirm")

    if st.button("Forget User", type="primary", key="forget_go", disabled=not confirm):
        type_map = {
            "semantic": MemoryType.SEMANTIC,
            "episodic": MemoryType.EPISODIC,
            "procedural": MemoryType.PROCEDURAL,
            "working": MemoryType.WORKING,
        }
        mtypes = [type_map[t] for t in types_to_forget]
        result = run_async(engine.forget(scope, memory_types=mtypes))
        st.success(f"Done! Confirmation ID: `{result.confirmation_id}`")
        st.json(result.deleted_per_type)


elif page == "🎮 Playground":
    st.markdown("# 🎮 Playground — Full Workflow")
    st.markdown("Try a complete workflow: add a conversation, then see what PyMem extracts and remembers.")

    scope = MemoryScope(user_id=user_id, agent_id=agent_id, session_id=session_id)

    st.markdown("### Step 1: Simulate a conversation")

    default_convo = [
        {"role": "user", "content": "Hi! I'm Arjun, I work at Infosys as a backend developer."},
        {"role": "assistant", "content": "Hello Arjun! Nice to meet you. How can I help you today?"},
        {"role": "user", "content": "I want to learn FastAPI. I already know Django and Flask. I prefer video tutorials."},
        {"role": "assistant", "content": "Great background! Since you know Flask, FastAPI will feel familiar. I recommend the official tutorial first."},
    ]

    convo_json = st.text_area(
        "Conversation (JSON array):",
        value=json.dumps(default_convo, indent=2),
        height=250,
        key="play_convo",
    )

    if st.button("Process this conversation", type="primary", key="play_go"):
        try:
            messages = json.loads(convo_json)
        except json.JSONDecodeError:
            st.error("Invalid JSON!")
            messages = None

        if messages:
            with st.spinner("Processing..."):
                # Add to working memory
                run_async(engine.working.append(messages, scope))

                # Simple extraction (no LLM needed)
                result = run_async(engine.add(messages, scope))

            st.success(f"Processed! Stored {result.memories_count} memories.")

            if result.memories:
                st.markdown("### Extracted memories:")
                for m in result.memories:
                    show_result(m.content, m.memory_type.value, m.score, m.id)

            st.markdown("---")
            st.markdown("### Step 2: See the assembled context")
            context = run_async(engine.get_context(scope, query="FastAPI learning"))
            if context.formatted:
                st.markdown(f'<div class="ctx-block">{context.formatted}</div>', unsafe_allow_html=True)
            else:
                st.info("No context assembled yet (try adding more diverse memories).")

    st.markdown("---")
    st.markdown("### Step 3: Search your memories")
    q = st.text_input("Search:", value="what does the user want to learn", key="play_search")
    if st.button("Search", key="play_search_go"):
        results = run_async(engine.search(q, scope, limit=5))
        if results.memories:
            for m in results.memories:
                show_result(m.content, m.memory_type.value, m.score, m.id)
        else:
            st.warning("No results. Process a conversation first!")


# ── Main guard ──
if __name__ == "__main__":
    pass

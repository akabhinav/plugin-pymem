"""PyMem Streamlit UI — visual dashboard for memory management.

Run: streamlit run pymem/ui/app.py
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime

import streamlit as st

from pymem.ui.engine_factory import get_or_create_engine

# Page config
st.set_page_config(
    page_title="PyMem — Memory Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def run_async(coro):
    """Run async function from sync Streamlit context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def main():
    # Sidebar navigation
    st.sidebar.title("PyMem")
    st.sidebar.caption("Four-Type Memory Platform for AI Agents")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "Add Memory",
            "Search Memories",
            "Memory Browser",
            "Agent Instructions",
            "Context Assembler",
            "GDPR Forget",
            "Platform Stats",
        ],
    )

    # Scope configuration in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Scope")
    user_id = st.sidebar.text_input("User ID", value="demo_user")
    agent_id = st.sidebar.text_input("Agent ID", value="demo_agent")
    session_id = st.sidebar.text_input("Session ID", value="session_001")

    engine = get_or_create_engine()

    if page == "Dashboard":
        render_dashboard(engine, user_id)
    elif page == "Add Memory":
        render_add_memory(engine, user_id, agent_id, session_id)
    elif page == "Search Memories":
        render_search(engine, user_id, agent_id)
    elif page == "Memory Browser":
        render_browser(engine, user_id, agent_id)
    elif page == "Agent Instructions":
        render_instructions(engine, agent_id)
    elif page == "Context Assembler":
        render_context(engine, user_id, agent_id, session_id)
    elif page == "GDPR Forget":
        render_forget(engine, user_id)
    elif page == "Platform Stats":
        render_stats(engine)


def render_dashboard(engine, user_id):
    st.title("PyMem Dashboard")
    st.markdown("**Four-type cognitive memory platform for AI agents**")

    col1, col2, col3, col4 = st.columns(4)

    stats = run_async(engine._relational.get_platform_stats())

    with col1:
        st.metric("Total Memories", stats.get("total_memories", 0))
    with col2:
        st.metric("Semantic", stats.get("semantic", 0))
    with col3:
        st.metric("Episodic", stats.get("episodic", 0))
    with col4:
        st.metric("Procedural", stats.get("procedural", 0))

    st.markdown("---")

    # Memory type explanation
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Memory Types")
        st.markdown("""
| Type | Purpose | Store | TTL |
|------|---------|-------|-----|
| **Working** | In-context sliding window | Redis | Session end |
| **Episodic** | What happened, when, with whom | Graph + Relational | 90 days |
| **Semantic** | Facts, preferences, entities | Vector + Graph | Decay curve |
| **Procedural** | Instructions, workflows | Relational | Never |
        """)

    with col2:
        st.subheader("Architecture")
        st.markdown("""
- **Storage**: Pluggable backends (pgvector, Qdrant, Kuzu, Neo4j)
- **Intelligence**: LLM extraction via PyGate
- **Scoping**: user_id / agent_id / session_id / org_id
- **GDPR**: First-class forget operation
- **Versioning**: Every write creates a version
        """)

    # Recent activity
    st.subheader("Recent Memories")
    from pymem.models import MemoryScope

    try:
        scope = MemoryScope(user_id=user_id)
        rows = run_async(
            engine._relational.list_memories(
                filters=scope.to_filter(), limit=10
            )
        )
        if rows:
            for row in rows:
                with st.expander(
                    f"[{row['memory_type'].upper()}] {row['content'][:80]}..."
                    if len(row["content"]) > 80
                    else f"[{row['memory_type'].upper()}] {row['content']}"
                ):
                    st.json(
                        {
                            "id": row["id"],
                            "type": row["memory_type"],
                            "content": row["content"],
                            "score": row.get("score", 0.5),
                            "created_at": row.get("created_at", ""),
                        }
                    )
        else:
            st.info("No memories yet. Use 'Add Memory' to get started.")
    except Exception as e:
        st.error(f"Error loading memories: {e}")


def render_add_memory(engine, user_id, agent_id, session_id):
    st.title("Add Memory")

    tab1, tab2, tab3 = st.tabs(
        ["From Conversation", "Manual Entry", "Bulk Import"]
    )

    with tab1:
        st.subheader("Extract memories from a conversation")
        conversation = st.text_area(
            "Paste conversation (one message per line, format: role: content)",
            height=200,
            placeholder="user: I prefer Python over Java\nassistant: Noted! I'll use Python for code examples.\nuser: I've been using Kafka for 3 years",
        )

        memory_types = st.multiselect(
            "Memory types to extract",
            ["semantic", "episodic", "procedural"],
            default=["semantic", "episodic"],
        )

        if st.button("Extract & Store", type="primary"):
            if not conversation.strip():
                st.warning("Please enter a conversation")
                return

            messages = []
            for line in conversation.strip().split("\n"):
                if ":" in line:
                    role, content = line.split(":", 1)
                    messages.append(
                        {"role": role.strip().lower(), "content": content.strip()}
                    )

            if not messages:
                st.warning("Could not parse messages")
                return

            from pymem.models import MemoryScope, MemoryType

            scope = MemoryScope(
                user_id=user_id, agent_id=agent_id, session_id=session_id
            )
            types = [MemoryType(t) for t in memory_types] if memory_types else None

            with st.spinner("Extracting memories..."):
                result = run_async(
                    engine.add(messages=messages, scope=scope, memory_types=types)
                )

            st.success(f"Stored {result.memories_count} memories!")
            for mem in result.memories:
                st.markdown(
                    f"- **[{mem.memory_type.value}]** {mem.content} (score: {mem.score:.2f})"
                )

    with tab2:
        st.subheader("Add a single memory manually")
        content = st.text_area("Memory content", placeholder="User prefers dark mode")
        mem_type = st.selectbox(
            "Memory type", ["semantic", "episodic", "procedural"]
        )
        score = st.slider("Importance score", 0.0, 1.0, 0.7)
        tags = st.text_input("Tags (comma-separated)", placeholder="preference, ui")

        if st.button("Add Memory"):
            if not content.strip():
                st.warning("Please enter content")
                return

            from pymem.models import MemoryScope

            scope = MemoryScope(user_id=user_id, agent_id=agent_id)
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]

            with st.spinner("Storing memory..."):
                if mem_type == "semantic":
                    mem = run_async(
                        engine.semantic.add(
                            content=content,
                            scope=scope,
                            score=score,
                            metadata={"tags": tag_list},
                        )
                    )
                elif mem_type == "episodic":
                    mem = run_async(
                        engine.episodic.add_event(
                            content=content,
                            scope=scope,
                            metadata={"tags": tag_list},
                        )
                    )
                else:
                    mem = run_async(
                        engine.procedural.add(
                            content=content, scope=scope, tags=tag_list
                        )
                    )

            st.success(f"Memory stored! ID: {mem.id}")

    with tab3:
        st.subheader("Bulk import memories")
        json_input = st.text_area(
            "JSON array of memories",
            height=200,
            placeholder='[{"content": "...", "type": "semantic", "score": 0.8}]',
        )

        if st.button("Import"):
            try:
                memories = json.loads(json_input)
                from pymem.models import MemoryScope

                scope = MemoryScope(user_id=user_id, agent_id=agent_id)
                count = 0
                for mem_data in memories:
                    content = mem_data.get("content", "")
                    if not content:
                        continue
                    mem_type = mem_data.get("type", "semantic")
                    if mem_type == "semantic":
                        run_async(
                            engine.semantic.add(content=content, scope=scope)
                        )
                    elif mem_type == "episodic":
                        run_async(
                            engine.episodic.add_event(content=content, scope=scope)
                        )
                    elif mem_type == "procedural":
                        run_async(
                            engine.procedural.add(content=content, scope=scope)
                        )
                    count += 1
                st.success(f"Imported {count} memories!")
            except json.JSONDecodeError:
                st.error("Invalid JSON format")


def render_search(engine, user_id, agent_id):
    st.title("Search Memories")

    query = st.text_input("Search query", placeholder="What does the user prefer?")
    col1, col2 = st.columns(2)
    with col1:
        search_types = st.multiselect(
            "Memory types",
            ["semantic", "episodic", "procedural"],
            default=["semantic", "episodic", "procedural"],
        )
    with col2:
        limit = st.slider("Max results", 1, 50, 10)

    if st.button("Search", type="primary") or query:
        if not query:
            st.info("Enter a search query")
            return

        from pymem.models import MemoryScope, MemoryType

        scope = MemoryScope(user_id=user_id, agent_id=agent_id)
        types = [MemoryType(t) for t in search_types] if search_types else None

        with st.spinner("Searching..."):
            result = run_async(
                engine.search(query=query, scope=scope, memory_types=types, limit=limit)
            )

        if not result.memories:
            st.info("No memories found matching your query")
            return

        st.subheader(f"Found {result.total} memories")
        for i, mem in enumerate(result.memories, 1):
            col1, col2, col3 = st.columns([1, 6, 1])
            with col1:
                st.markdown(f"**#{i}**")
                type_color = {
                    "semantic": "blue",
                    "episodic": "green",
                    "procedural": "orange",
                }.get(mem.memory_type.value, "gray")
                st.markdown(
                    f":{type_color}[{mem.memory_type.value}]"
                )
            with col2:
                st.markdown(mem.content)
            with col3:
                st.markdown(f"Score: **{mem.score:.2f}**")
            st.divider()


def render_browser(engine, user_id, agent_id):
    st.title("Memory Browser")

    filter_type = st.selectbox(
        "Filter by type",
        ["All", "semantic", "episodic", "procedural"],
    )

    from pymem.models import MemoryScope

    scope = MemoryScope(user_id=user_id, agent_id=agent_id)
    mem_type = filter_type if filter_type != "All" else None

    rows = run_async(
        engine._relational.list_memories(
            filters=scope.to_filter(), memory_type=mem_type, limit=50
        )
    )

    if not rows:
        st.info("No memories found")
        return

    st.subheader(f"{len(rows)} memories")

    for row in rows:
        col1, col2, col3 = st.columns([1, 5, 2])
        with col1:
            type_emoji = {
                "semantic": "💡",
                "episodic": "📅",
                "procedural": "⚙️",
                "working": "💬",
            }.get(row["memory_type"], "📝")
            st.markdown(f"{type_emoji} **{row['memory_type']}**")
        with col2:
            st.markdown(row["content"])
        with col3:
            st.caption(f"Score: {row.get('score', 0.5):.2f}")
            st.caption(f"ID: {row['id'][:8]}...")
            if st.button("Delete", key=f"del_{row['id']}"):
                run_async(engine._relational.soft_delete_memory(row["id"]))
                st.rerun()
        st.divider()


def render_instructions(engine, agent_id):
    st.title("Agent Instructions")
    st.markdown(f"**Managing instructions for agent: `{agent_id}`**")

    # Add new instruction
    with st.form("add_instruction"):
        st.subheader("Add New Instruction")
        content = st.text_area(
            "Instruction",
            placeholder="Always respond in TypeScript when writing code examples",
        )
        col1, col2 = st.columns(2)
        with col1:
            priority = st.slider("Priority (1=highest)", 1, 10, 5)
        with col2:
            tags = st.text_input("Tags", placeholder="coding, style")

        if st.form_submit_button("Add Instruction", type="primary"):
            if content:
                from pymem.models import MemoryScope

                scope = MemoryScope(agent_id=agent_id)
                tag_list = [t.strip() for t in tags.split(",") if t.strip()]
                mem = run_async(
                    engine.procedural.add(
                        content=content,
                        scope=scope,
                        tags=tag_list,
                        priority=priority,
                    )
                )
                st.success(f"Instruction added! ID: {mem.id}")
                st.rerun()

    # List existing instructions
    st.markdown("---")
    st.subheader("Current Instructions")

    from pymem.models import MemoryScope

    scope = MemoryScope(agent_id=agent_id)
    instructions = run_async(engine.procedural.get_all(scope))

    if not instructions:
        st.info("No instructions yet")
        return

    for inst in instructions:
        priority = inst.metadata.get("priority", 5)
        tags = inst.metadata.get("tags", [])
        with st.container():
            col1, col2, col3 = st.columns([1, 6, 1])
            with col1:
                st.markdown(f"**P{priority}**")
            with col2:
                st.markdown(inst.content)
                if tags:
                    st.caption(f"Tags: {', '.join(tags)}")
            with col3:
                if st.button("Remove", key=f"rm_{inst.id}"):
                    run_async(engine._relational.soft_delete_memory(inst.id))
                    st.rerun()
            st.divider()


def render_context(engine, user_id, agent_id, session_id):
    st.title("Context Assembler")
    st.markdown(
        "Preview the full context that would be injected into an agent's system prompt"
    )

    query = st.text_input(
        "Query context (optional)",
        placeholder="What's the user's tech stack?",
    )
    max_tokens = st.slider("Max tokens", 1000, 8000, 4000)

    if st.button("Assemble Context", type="primary"):
        from pymem.models import MemoryScope

        scope = MemoryScope(
            user_id=user_id, agent_id=agent_id, session_id=session_id
        )

        with st.spinner("Assembling context..."):
            ctx = run_async(
                engine.get_context(
                    scope=scope,
                    query=query or None,
                    max_tokens=max_tokens,
                )
            )

        # Display formatted context
        st.subheader("Formatted Prompt Block")
        st.code(ctx.formatted or "(empty — no memories found)", language="markdown")

        # Breakdown
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Relevant Facts")
            for fact in ctx.relevant_facts:
                st.markdown(f"- {fact.content}")
            if not ctx.relevant_facts:
                st.caption("None")

        with col2:
            st.subheader("Recent Events")
            for event in ctx.recent_events:
                st.markdown(f"- {event.content}")
            if not ctx.recent_events:
                st.caption("None")

        st.subheader("Agent Instructions")
        for inst in ctx.agent_instructions:
            st.markdown(f"- {inst.content}")
        if not ctx.agent_instructions:
            st.caption("None")

        if ctx.working_messages:
            st.subheader("Working Memory")
            for msg in ctx.working_messages[-5:]:
                st.markdown(f"**{msg.get('role', 'user')}**: {msg.get('content', '')}")


def render_forget(engine, user_id):
    st.title("GDPR Forget")
    st.warning(
        "This operation permanently deletes all memories for the specified scope. "
        "This action cannot be undone."
    )

    scope_type = st.selectbox(
        "What to forget",
        ["all", "semantic", "episodic", "procedural"],
    )

    st.markdown(f"**User:** `{user_id}`")
    st.markdown(f"**Scope:** `{scope_type}`")

    confirm = st.text_input(
        f'Type "{user_id}" to confirm deletion',
        placeholder="Type user ID to confirm",
    )

    if st.button("Forget", type="primary"):
        if confirm != user_id:
            st.error("Confirmation does not match user ID")
            return

        from pymem.models import MemoryScope, MemoryType

        scope = MemoryScope(user_id=user_id)
        types = [MemoryType(scope_type)] if scope_type != "all" else None

        with st.spinner("Deleting memories..."):
            result = run_async(engine.forget(scope=scope, memory_types=types))

        st.success("Memories deleted successfully!")
        st.json(
            {
                "deleted_per_type": result.deleted_per_type,
                "confirmation_id": result.confirmation_id,
                "completed_at": str(result.completed_at),
            }
        )


def render_stats(engine):
    st.title("Platform Statistics")

    stats = run_async(engine._relational.get_platform_stats())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Memories", stats.get("total_memories", 0))
    with col2:
        st.metric("Unique Users", stats.get("unique_users", 0))
    with col3:
        st.metric("Unique Agents", stats.get("unique_agents", 0))

    st.markdown("---")

    # Memory type distribution
    st.subheader("Memory Type Distribution")
    type_data = {
        "Semantic": stats.get("semantic", 0),
        "Episodic": stats.get("episodic", 0),
        "Procedural": stats.get("procedural", 0),
        "Working": stats.get("working", 0),
    }

    if any(type_data.values()):
        import pandas as pd

        df = pd.DataFrame(
            {"Type": list(type_data.keys()), "Count": list(type_data.values())}
        )
        st.bar_chart(df.set_index("Type"))
    else:
        st.info("No data yet")

    # Backend health
    st.subheader("Backend Health")
    vector_ok = run_async(engine._vector.health_check())
    graph_ok = run_async(engine._graph.health_check())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"Vector Store: {'✅' if vector_ok else '❌'}")
    with col2:
        st.markdown(f"Graph Store: {'✅' if graph_ok else '❌'}")
    with col3:
        st.markdown("Relational: ✅")


if __name__ == "__main__":
    main()

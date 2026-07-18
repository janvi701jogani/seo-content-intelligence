from modules.community.reddit import (
    RedditCollector,
    run_community_intelligence,
)
from modules.extractor import run_intelligence_engine
import pandas as pd
import streamlit as st
from openai import OpenAI

from modules.project.project_manager import (
    initialize,
    list_projects,
)

from modules.serp.serper import get_serp
from modules.scraping.collector import collect_competitors


st.set_page_config(
    page_title="SEO Intelligence Platform",
    page_icon="📈",
    layout="wide"
)

initialize()

st.title("SEO Intelligence Platform")


def render_table(rows, empty_message):
    if rows:
        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(empty_message)

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("API Keys")

openai_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password"
)

serper_key = st.sidebar.text_input(
    "Serper API Key",
    type="password"
)

client = None

if openai_key:
    client = OpenAI(api_key=openai_key)

st.sidebar.header("Projects")

projects = list_projects()

if projects:
    st.sidebar.selectbox(
        "Existing Projects",
        projects
    )

reddit_client_id = st.sidebar.text_input(
    "Reddit Client ID",
    type="password"
)

reddit_client_secret = st.sidebar.text_input(
    "Reddit Client Secret",
    type="password"
)

reddit_user_agent = st.sidebar.text_input(
    "Reddit User Agent"
)

# -----------------------------
# Inputs
# -----------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    keyword = st.text_input(
        "Keyword"
    )

with col2:
    country = st.selectbox(
        "Country",
        {
            "India": "in",
            "United States": "us",
            "United Kingdom": "uk",
            "Australia": "au",
            "Canada": "ca"
        }
    )

with col3:
    language = st.selectbox(
        "Language",
        {
            "English": "en",
            "French": "fr",
            "German": "de",
            "Spanish": "es"
        }
    )

with col4:
    num_results = st.slider(
        "Competitors",
        5,
        20,
        10
    )

# -----------------------------
# Run
# -----------------------------

if st.button("Run Competitor Intelligence"):

    if not serper_key:
        st.error("Please enter Serper API Key.")
        st.stop()

    if not keyword:
        st.error("Please enter a keyword.")
        st.stop()

    project_name = keyword.lower().strip().replace(" ", "-")

    with st.spinner("Searching Google..."):

        organic_results, serp_summary = get_serp(
            keyword=keyword,
            serper_key=serper_key,
            country=country,
            language=language,
            num_results=num_results
        )

    with st.spinner("Scraping competitors..."):

        competitors = collect_competitors(
            keyword=keyword,
            serper_key=serper_key,
            project_name=project_name,
            country=country,
            language=language,
            num_results=num_results
        )

    with st.spinner("Extracting entities..."):

        results = run_intelligence_engine(
            competitors
        )

        competitors = results["competitors"]
        entities = results["entities"]
        topics = results["topics"]

    reddit_threads = []

    if reddit_client_id and reddit_client_secret and reddit_user_agent:
        with st.spinner("Fetching Reddit discussions..."):
            collector = RedditCollector(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
                user_agent=reddit_user_agent,
            )
            reddit_threads = collector.search_via_google(
                keyword=keyword,
                serper_key=serper_key,
                limit=10,
                country=country,
                language=language,
            )

    community = {}

    if reddit_threads:
        with st.spinner("Processing community intelligence..."):
            community = run_community_intelligence(
                reddit_threads,
                competitor_entities=entities,
                competitor_topics=topics,
            )

    tabs = st.tabs([
        "SERP",
        "Competitors",
        "Topics",
        "Entities",
        "Community"
    ])

    # -----------------------------
    # SERP
    # -----------------------------

    with tabs[0]:

        st.subheader("Organic Results")

        for result in organic_results:

            st.markdown(f"### {result.get('title','')}")

            st.write(result.get("link", ""))

            st.write(result.get("snippet", ""))

            st.divider()

    # -----------------------------
    # Competitors
    # -----------------------------

    with tabs[1]:

        st.subheader("Competitor Intelligence")

        for competitor in competitors:

            with st.expander(competitor["title"]):

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Position:**", competitor["position"])
                    st.write("**URL:**", competitor["url"])

                with col2:
                    st.write("**Credits:**", competitor["credits"])
                    st.write("**Words:**", len(competitor["text"].split()))

                st.write("### Metadata")
                st.json(competitor["metadata"])

                st.write("### Content")

                st.text_area(
                    f"content_{competitor['position']}",
                    competitor["text"],
                    height=350
                )

    # -----------------------------
    # Topics
    # -----------------------------

    with tabs[2]:

        st.subheader("Topic Intelligence")

        if topics:
            topics_df = pd.DataFrame(topics)
            topic_columns = [
                "topic",
                "coverage",
                "competitors_using",
                "importance",
                "heading_hits",
                "sources",
                "urls",
                "average_word_count",
                "related_phrases",
                "related_entities",
            ]
            topic_columns = [c for c in topic_columns if c in topics_df.columns]
            st.dataframe(
                topics_df[topic_columns],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No topics found.")

    # -----------------------------
    # Entities
    # -----------------------------

    with tabs[3]:

        st.subheader("Entity Intelligence")

        if entities:
            entities_df = pd.DataFrame(entities)
            entity_columns = [
                "entity",
                "type",
                "mentions",
                "competitors_using",
                "coverage",
                "average_mentions",
                "importance",
                "confidence",
                "extractor_sources",
            ]
            entity_columns = [c for c in entity_columns if c in entities_df.columns]
            st.dataframe(
                entities_df[entity_columns],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No entities found.")

    # -----------------------------
    # Community
    # -----------------------------

    with tabs[4]:

        st.subheader("Community Intelligence")

        if not reddit_threads:
            st.info("No Reddit threads collected.")
        else:
            # Overview
            stats = community.get("statistics", {})
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Threads analyzed", stats.get("threads", 0))
            m2.metric("Comments analyzed", stats.get("comments", 0))
            m3.metric("Subreddits", stats.get("subreddits", 0))
            m4.metric("Avg. upvotes", stats.get("average_upvotes", 0))
            m5.metric("Time span", stats.get("time_span", "-"))
            if stats.get("subreddit_names"):
                st.caption(
                    "Subreddits: "
                    + ", ".join(f"r/{name}" for name in stats["subreddit_names"])
                )

            st.divider()
            st.subheader("Questions")
            render_table(community.get("questions", []), "No questions found.")

            st.divider()
            st.subheader("Pain Points")
            render_table(community.get("pain_points", []), "No pain points found.")

            st.divider()
            st.subheader("Recommendations")
            render_table(
                community.get("recommendations", []),
                "No recommendations found."
            )

            st.divider()
            st.subheader("Brands & Products")
            render_table(community.get("brands", []), "No brands found.")

            st.divider()
            st.subheader("Features Users Care About")
            render_table(community.get("features", []), "No features found.")

            st.divider()
            st.subheader("Decision Factors")
            render_table(
                community.get("decision_factors", []),
                "No decision factors found."
            )

            st.divider()
            st.subheader("Community Terminology")
            render_table(community.get("vocabulary", []), "No terminology found.")

            st.divider()
            st.subheader("Common Mistakes")
            render_table(community.get("mistakes", []), "No mistakes found.")

            st.divider()
            st.subheader("Myths")
            render_table(community.get("myths", []), "No myths found.")

            st.divider()
            st.subheader("Real Experiences")
            render_table(community.get("experiences", []), "No experiences found.")

            st.divider()
            st.subheader("Content Opportunities")
            render_table(
                community.get("gaps", []),
                "No content opportunities computed."
            )

            st.divider()
            st.subheader("Threads")
            st.write(f"Threads found: {len(reddit_threads)}")
            for thread in reddit_threads:
                with st.expander(thread.title):
                    st.write(f"**Subreddit:** r/{thread.subreddit}")
                    st.write(f"**Score:** {thread.score}")
                    st.write(f"**Comments:** {thread.num_comments}")
                    st.write(thread.selftext)
                    st.markdown(thread.permalink)
                    st.subheader("Top Comments")
                    for comment in thread.comments[:10]:
                        st.write(comment.body)
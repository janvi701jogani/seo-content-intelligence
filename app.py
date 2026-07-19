from modules.community.reddit import (
    RedditCollector,
    run_community_intelligence,
)
from modules.research.literature import (
    ResearchCollector,
    run_research_intelligence,
)
from modules.search.intelligence import run_search_intelligence
from modules.strategy.funnel import run_funnel_strategy
from modules.brief.content_brief import run_content_brief
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

    research = {}

    with st.spinner("Scanning research literature..."):
        research_papers = ResearchCollector().search(
            keyword=keyword,
            limit=25,
        )
        if research_papers:
            research = run_research_intelligence(
                research_papers,
                competitor_entities=entities,
                competitor_topics=topics,
            )

    with st.spinner("Analyzing search intent..."):
        search_intel = run_search_intelligence(
            keyword=keyword,
            serper_key=serper_key,
            country=country,
            language=language,
            competitors=competitors,
            community=community,
        )

    strategy = {}

    if client is not None:
        with st.spinner("Generating funnel content strategy..."):
            strategy = run_funnel_strategy(
                client,
                keyword,
                topics=topics,
                entities=entities,
                community=community,
                search_intel=search_intel,
                competitors=competitors,
            )

    brief = {}

    if client is not None:
        with st.spinner("Researching all tabs and drafting content brief..."):
            brief = run_content_brief(
                client,
                keyword,
                bundle={
                    "keyword": keyword,
                    "topics": topics,
                    "entities": entities,
                    "search_intel": search_intel,
                    "community": community,
                    "reddit_threads": reddit_threads,
                    "research": research,
                    "research_papers": research_papers,
                    "competitors": competitors,
                    "strategy": strategy,
                },
            )

    tabs = st.tabs([
        "SERP",
        "Search",
        "Competitors",
        "Topics",
        "Entities",
        "Community",
        "Research",
        "Recommendations",
        "Content Brief"
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
    # Search Intelligence
    # -----------------------------

    with tabs[1]:

        st.subheader("Search Intelligence")

        st.write("### Signal Matrix")
        st.caption(
            "Questions ranked by opportunity: strong Google presence + "
            "Reddit demand + low competitor coverage rise to the top."
        )
        render_table(
            search_intel.get("signal_matrix", []),
            "No signal matrix computed."
        )

        st.divider()
        st.write("### Question Intelligence")
        st.caption("Merged and clustered across all sources below.")
        render_table(
            search_intel.get("question_intelligence", []),
            "No questions found."
        )

        st.divider()
        st.write("### People Also Ask")
        render_table(search_intel.get("paa", []), "No PAA results.")

        st.divider()
        st.write("### Related Searches")
        render_table(
            search_intel.get("related_searches", []),
            "No related searches."
        )

        st.divider()
        st.write("### Autosuggest")
        render_table(search_intel.get("autosuggest", []), "No autosuggest results.")

        st.divider()
        st.write("### Competitor FAQs")
        render_table(search_intel.get("faqs", []), "No FAQs extracted.")

    # -----------------------------
    # Competitors
    # -----------------------------

    with tabs[2]:

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

    with tabs[3]:

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

    with tabs[4]:

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

    with tabs[5]:

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

    # -----------------------------
    # Research
    # -----------------------------

    with tabs[6]:

        st.subheader("Research Insights")

        if not research:
            st.info("No research literature found.")
        else:
            # Overview
            rstats = research.get("statistics", {})
            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("Papers scanned", rstats.get("papers", 0))
            r2.metric("Reviews", rstats.get("reviews", 0))
            r3.metric("Journals", rstats.get("journals", 0))
            r4.metric("Avg. citations", rstats.get("average_citations", 0))
            r5.metric("Year span", rstats.get("year_span", "-"))

            st.divider()
            st.subheader("Information Gain")
            st.caption(
                "Research concepts ranked by lowest competitor coverage "
                "and highest research support. These are the unique, "
                "citable angles competitors miss."
            )
            render_table(
                research.get("information_gain", []),
                "No information-gain concepts computed."
            )

            st.divider()
            st.subheader("Data Points to Cite")
            render_table(
                research.get("data_points", []),
                "No numeric findings extracted."
            )

            st.divider()
            st.subheader("Research Concepts")
            render_table(
                research.get("concepts", []),
                "No research concepts found."
            )

            st.divider()
            st.subheader("Papers (recent and reviews first)")
            render_table(
                research.get("papers", []),
                "No papers found."
            )

    # -----------------------------
    # Recommendations (funnel strategy)
    # -----------------------------

    with tabs[7]:

        st.subheader("Content Recommendations")

        if client is None:
            st.info("Enter an OpenAI API key in the sidebar to generate "
                    "funnel content recommendations.")
        elif not strategy:
            st.info("No recommendations generated.")
        else:
            stage = strategy.get("stage", "")
            rationale = strategy.get("stage_rationale", "")
            st.write(f"**Detected funnel stage:** {stage}")
            if rationale:
                st.caption(rationale)
            st.caption(
                "Educational, brand-neutral angles for the stages below "
                "this keyword in the funnel."
            )

            recommendations = strategy.get("recommendations", {}) or {}

            for stage_key in ("MOFU", "BOFU"):
                items = recommendations.get(stage_key, [])
                if items:
                    st.divider()
                    st.subheader(f"{stage_key} Content Ideas")
                    render_table(items, f"No {stage_key} recommendations.")

            faqs = recommendations.get("FAQs", [])
            if faqs:
                st.divider()
                st.subheader("FAQs")
                render_table(faqs, "No FAQ recommendations.")

    # -----------------------------
    # Content Brief (research agent)
    # -----------------------------

    with tabs[8]:

        st.subheader("Content Brief")

        if client is None:
            st.info("Enter an OpenAI API key in the sidebar to generate a "
                    "content brief.")
        elif not brief or not brief.get("brief"):
            st.info("No content brief generated.")
        else:
            st.download_button(
                "Download brief (Markdown)",
                brief["brief"],
                file_name=f"{project_name}-content-brief.md",
                mime="text/markdown",
            )
            st.markdown(brief["brief"])

            with st.expander("All sources"):
                sources = brief.get("sources", {})
                st.write("**Competitors**")
                render_table(sources.get("competitors", []), "None.")
                st.write("**Reddit threads**")
                render_table(sources.get("reddit_threads", []), "None.")
                st.write("**Papers**")
                render_table(sources.get("papers", []), "None.")
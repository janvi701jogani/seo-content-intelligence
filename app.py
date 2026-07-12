from modules.extractor import run_intelligence_engine
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

        from modules.extractor import run_intelligence_engine

        results = run_intelligence_engine(
           competitors
        )

        competitors = results["competitors"]

        entities = results["entities"]

        topics = results["topics"]

        coverage = results["coverage"]
        statistics = results["statistics"]

    tabs = st.tabs([
        "SERP",
        "Competitors",
        "Entities",
        "Topics",
        "Coverage",
        "Statistics"
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
    # Entities
    # -----------------------------

    with tabs[2]:

        st.subheader("Entity Intelligence")

        if entities:

            st.dataframe(
                entities,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info("No entities found.")

    # -----------------------------
    # Topics
    # -----------------------------

    with tabs[3]:
        st.subheader("Topic Intelligence")
        topics = results.get("topics", [])
        if topics:
            st.dataframe(
                topics,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No topics found.")

    # -----------------------------
    # Coverage
    # -----------------------------

    with tabs[4]:
        st.subheader("Coverage")
        if coverage:
            st.dataframe(
                coverage,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No coverage data found.")

    # -----------------------------
    # Statistics
    # -----------------------------

    with tabs[5]:
        st.subheader("Statistics")
        if statistics:
            st.dataframe(
                statistics,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No statistics found.")
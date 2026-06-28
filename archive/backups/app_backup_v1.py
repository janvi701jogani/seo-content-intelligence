import streamlit as st
from openai import OpenAI

from modules.serp.serper import get_serp
from modules.content.brief import generate_brief
from modules.content.faq import generate_faqs
from modules.content.eeat import generate_eeat
from modules.content.info_gain import generate_info_gain

st.set_page_config(
    page_title="SEO Content Optimization Workspace",
    layout="wide"
)

st.title("SEO Content Optimization Workspace")

st.sidebar.header("API Configuration")

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

st.header("Content Brief Generator")

col1, col2 = st.columns(2)

with col1:
    keyword = st.text_input("Primary Keyword")

with col2:
    geography = st.selectbox(
        "Target Geography",
        ["India", "US", "UK", "Australia"]
    )

if st.button("Generate Analysis"):

    if not openai_key:
        st.error("Please enter OpenAI API Key")

    elif not serper_key:
        st.error("Please enter Serper API Key")

    elif not keyword:
        st.error("Please enter a keyword")

    else:

        organic_results, serp_summary = get_serp(
            keyword,
            serper_key
        )

        tabs = st.tabs([
            "SERP",
            "Competitors",
            "Content Brief",
            "FAQs",
            "EEAT",
            "Information Gain"
        ])

        with tabs[0]:

            st.subheader("Top Ranking Pages")

            for result in organic_results:

                st.markdown(
                    f"""
### {result.get('title')}

**URL:** {result.get('link')}

**Snippet:**
{result.get('snippet')}
"""
                )

        with tabs[1]:

            st.subheader("Competitor Analysis")

            st.info(
                "Competitor scraping will be connected next."
            )

        with tabs[2]:

            brief = generate_brief(
                client,
                keyword,
                geography,
                serp_summary
            )

            st.markdown(brief)

        with tabs[3]:

            faqs = generate_faqs(
                client,
                keyword,
                serp_summary
            )

            st.markdown(faqs)

        with tabs[4]:

            eeat = generate_eeat(
                client,
                keyword,
                serp_summary
            )

            st.markdown(eeat)

        with tabs[5]:

            info_gain = generate_info_gain(
                client,
                keyword,
                serp_summary
            )

            st.markdown(info_gain)
import requests
import streamlit as st
from openai import OpenAI

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="SEO Content Optimization Workspace",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================

st.title("SEO Content Optimization Workspace")

# =====================================================
# API CONFIGURATION
# =====================================================

st.sidebar.header("API Configuration")

openai_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password"
)

serper_key = st.sidebar.text_input(
    "Serper API Key",
    type="password"
)

reddit_key = st.sidebar.text_input(
    "Reddit API Key",
    type="password"
)

tavily_key = st.sidebar.text_input(
    "Tavily API Key",
    type="password"
)

gsc_credentials = st.sidebar.file_uploader(
    "Upload GSC Service Account JSON",
    type=["json"]
)

# =====================================================
# OPENAI CLIENT
# =====================================================

client = None

if openai_key:
    client = OpenAI(api_key=openai_key)

# =====================================================
# TASK SELECTION
# =====================================================

st.sidebar.header("Task Selection")

task = st.sidebar.radio(
    "Choose Task",
    [
        "Content Brief Generator",
        "Reoptimization",
        "FAQ Optimization",
        "GSC Opportunity Mining",
        "AI Overview Analysis"
    ]
)

# =====================================================
# CONTENT BRIEF GENERATOR
# =====================================================

if task == "Content Brief Generator":

    st.header("Content Brief Generator")

    col1, col2 = st.columns(2)

    with col1:

        keyword = st.text_input("Primary Keyword")

        geography = st.selectbox(
            "Target Geography",
            ["India", "US", "UK", "Australia"]
        )

    with col2:

        language = st.selectbox(
            "Language",
            ["English"]
        )

        device = st.selectbox(
            "Device",
            ["Desktop", "Mobile"]
        )

    st.subheader("Competitor Analysis")

    competitors = st.text_area(
        "Competitor URLs (one per line)"
    )

    st.subheader("Optional Analysis")

    col3, col4, col5 = st.columns(3)

    with col3:
        aio = st.checkbox("AI Overview Analysis")
        faq = st.checkbox("FAQ Suggestions")
        info_gain = st.checkbox("Information Gain")

    with col4:
        eeat = st.checkbox("EEAT Suggestions")
        reddit = st.checkbox("Reddit Pulse")
        quora = st.checkbox("Quora Pulse")

    with col5:
        research = st.checkbox("Research Paper Analysis")
        entities = st.checkbox("Semantic Entities")
        formatting = st.checkbox("SERP Formatting")

    if st.button("Generate Content Brief"):

        if not openai_key:
            st.error("Please enter OpenAI API Key")

        elif not serper_key:
            st.error("Please enter Serper API Key")

        else:

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "SERP Analysis",
                "Content Brief",
                "Information Gain",
                "FAQs",
                "EEAT"
            ])

            # =====================================================
            # FETCH REAL SERP DATA
            # =====================================================

            with st.spinner("Fetching real SERP data..."):

                url = "https://google.serper.dev/search"

                payload = {
                    "q": keyword,
                    "gl": "in",
                    "hl": "en",
                    "num": 10
                }

                headers = {
                    "X-API-KEY": serper_key,
                    "Content-Type": "application/json"
                }

                response = requests.post(
                    url,
                    json=payload,
                    headers=headers
                )

                data = response.json()

                organic_results = data.get("organic", [])

                serp_summary = ""

                for result in organic_results:

                    serp_summary += f"""
                    Title: {result.get('title')}
                    Snippet: {result.get('snippet')}
                    URL: {result.get('link')}
                    """

            # =====================================================
            # SERP ANALYSIS
            # =====================================================

            with tab1:

                st.subheader("Top Ranking Pages")

                for result in organic_results:

                    st.markdown(f"""
### {result.get('title')}

**URL:** {result.get('link')}

**Snippet:**  
{result.get('snippet')}
                    """)

                st.subheader("AI SERP Insights")

                with st.spinner("Analyzing SERP patterns..."):

                    response = client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an advanced SEO strategist."
                            },
                            {
                                "role": "user",
                                "content": f"""
                                Analyze these REAL SERP results.

                                Keyword:
                                {keyword}

                                Geography:
                                {geography}

                                SERP DATA:
                                {serp_summary}

                                Identify:
                                - dominant search intent
                                - ranking patterns
                                - content depth trends
                                - content structure trends
                                - likely AI Overview opportunities
                                - formatting patterns
                                - opportunities competitors are missing
                                """
                            }
                        ]
                    )

                    st.markdown(
                        response.choices[0].message.content
                    )

            # =====================================================
            # CONTENT BRIEF
            # =====================================================

            with tab2:

                st.subheader("Content Brief")

                with st.spinner("Generating content brief..."):

                    response = client.chat.completions.create(
                        model="gpt-4.1",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an advanced SEO content strategist."
                            },
                            {
                                "role": "user",
                                "content": f"""
                                Create a comprehensive SEO content brief.

                                Keyword:
                                {keyword}

                                Geography:
                                {geography}

                                REAL SERP DATA:
                                {serp_summary}

                                Include:
                                - search intent
                                - recommended H1
                                - detailed H2/H3 structure
                                - topical coverage
                                - semantic entities
                                - content flow
                                - formatting suggestions
                                - AI Overview optimization
                                - user psychology considerations
                                - content differentiation
                                - opportunities competitors missed
                                """
                            }
                        ]
                    )

                    st.markdown(
                        response.choices[0].message.content
                    )

            # =====================================================
            # INFORMATION GAIN
            # =====================================================

            with tab3:

                st.subheader("Information Gain")

                with st.spinner("Generating information gain suggestions..."):

                    response = client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an information gain SEO expert."
                            },
                            {
                                "role": "user",
                                "content": f"""
                                Analyze these REAL SERP results.

                                Keyword:
                                {keyword}

                                SERP DATA:
                                {serp_summary}

                                Suggest:
                                - overlooked angles
                                - unique insights
                                - emotional/user psychology gaps
                                - original research opportunities
                                - expert insight opportunities
                                - comparison opportunities
                                - community pain points
                                """
                            }
                        ]
                    )

                    st.markdown(
                        response.choices[0].message.content
                    )

            # =====================================================
            # FAQS
            # =====================================================

            with tab4:

                st.subheader("FAQ Suggestions")

                with st.spinner("Generating FAQs..."):

                    response = client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an SEO FAQ strategist."
                            },
                            {
                                "role": "user",
                                "content": f"""
                                Generate advanced FAQs for:

                                {keyword}

                                Based on these SERP results:
                                {serp_summary}

                                Include:
                                - informational FAQs
                                - transactional FAQs
                                - comparison FAQs
                                - objection handling FAQs
                                - AI Overview friendly FAQs
                                - trust/reassurance FAQs
                                """
                            }
                        ]
                    )

                    st.markdown(
                        response.choices[0].message.content
                    )

            # =====================================================
            # EEAT
            # =====================================================

            with tab5:

                st.subheader("EEAT Suggestions")

                with st.spinner("Generating EEAT recommendations..."):

                    response = client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an EEAT optimization expert."
                            },
                            {
                                "role": "user",
                                "content": f"""
                                Suggest EEAT improvements for:

                                {keyword}

                                Based on:
                                {serp_summary}

                                Include:
                                - expert recommendations
                                - trust signals
                                - authority signals
                                - statistics suggestions
                                - citation opportunities
                                - credibility improvements
                                - research opportunities
                                """
                            }
                        ]
                    )

                    st.markdown(
                        response.choices[0].message.content
                    )

# =====================================================
# REOPTIMIZATION
# =====================================================

elif task == "Reoptimization":

    st.header("Reoptimization Workspace")

    url = st.text_input("Existing URL")

    existing_content = st.text_area(
        "OR paste existing content"
    )

    competitor_urls = st.text_area(
        "Competitor URLs"
    )

    st.subheader("Optimization Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        gap = st.checkbox("Gap Analysis")
        faqs = st.checkbox("FAQ Optimization")
        aio = st.checkbox("AI Overview Optimization")

    with col2:
        gsc = st.checkbox("GSC Opportunities")
        sentiment = st.checkbox("Brand Sentiment")
        info_gain = st.checkbox("Information Gain")

    with col3:
        section = st.checkbox("Section Optimization")
        reddit = st.checkbox("Reddit Pulse")
        quora = st.checkbox("Quora Pulse")

    if st.button("Run Reoptimization"):

        if not openai_key:
            st.error("Please enter OpenAI API Key")

        else:

            with st.spinner("Running reoptimization analysis..."):

                response = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an advanced SEO reoptimization expert."
                        },
                        {
                            "role": "user",
                            "content": f"""
                            Analyze this content for reoptimization.

                            URL:
                            {url}

                            CONTENT:
                            {existing_content}

                            Give:
                            - missing subtopics
                            - missing entities
                            - weak sections
                            - optimization opportunities
                            - FAQ opportunities
                            - information gain suggestions
                            - AI Overview opportunities
                            """
                        }
                    ]
                )

                st.markdown(
                    response.choices[0].message.content
                )

# =====================================================
# FAQ OPTIMIZATION
# =====================================================

elif task == "FAQ Optimization":

    st.header("FAQ Optimization")

    keyword = st.text_input("Keyword")

    url = st.text_input("URL")

    if st.button("Generate FAQs"):

        if not openai_key:
            st.error("Please enter OpenAI API Key")

        else:

            with st.spinner("Generating FAQs..."):

                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an FAQ generation expert."
                        },
                        {
                            "role": "user",
                            "content": f"""
                            Generate advanced SEO FAQs for:

                            {keyword}

                            Include:
                            - informational
                            - transactional
                            - objection handling
                            - comparison
                            - AI Overview
                            """
                        }
                    ]
                )

                st.markdown(
                    response.choices[0].message.content
                )

# =====================================================
# GSC OPPORTUNITIES
# =====================================================

elif task == "GSC Opportunity Mining":

    st.header("GSC Opportunity Mining")

    property_name = st.text_input("GSC Property")

    page_url = st.text_input("Page URL")

    if st.button("Analyze GSC Opportunities"):

        st.info("GSC integration coming next")

# =====================================================
# AI OVERVIEW ANALYSIS
# =====================================================

elif task == "AI Overview Analysis":

    st.header("AI Overview Analysis")

    keyword = st.text_input("Keyword")

    geography = st.selectbox(
        "Target Geography",
        ["India", "US", "UK", "Australia"]
    )

    if st.button("Analyze AI Overview"):

        if not openai_key:
            st.error("Please enter OpenAI API Key")

        else:

            with st.spinner("Analyzing AI Overview opportunities..."):

                response = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an AI Overview optimization expert."
                        },
                        {
                            "role": "user",
                            "content": f"""
                            Analyze AI Overview opportunities for:

                            {keyword}

                            Geography:
                            {geography}

                            Include:
                            - answer formatting
                            - compression patterns
                            - factual density
                            - AI Overview optimization
                            - paragraph recommendations
                            - trust recommendations
                            """
                        }
                    ]
                )

                st.markdown(
                    response.choices[0].message.content
                )
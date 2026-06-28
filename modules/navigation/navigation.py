import streamlit as st


PAGES = [
    "Dashboard",
    "Projects",
    "Data Collection",
    "Competitor Intelligence",
    "Entity Intelligence",
    "Topic Intelligence",
    "Research Intelligence",
    "Brand Intelligence",
    "AI Outputs",
    "Exports",
    "Settings"
]


def sidebar():

    st.sidebar.title("SEO Intelligence Platform")

    page = st.sidebar.radio(
        "Navigation",
        PAGES
    )

    return page
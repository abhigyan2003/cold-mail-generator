import sys
import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text

# Print the current Python executable for debugging
print("Python executable in use:", sys.executable)

# Optional: Set USER_AGENT to silence chromadb warnings
os.environ["USER_AGENT"] = "cold-email-generator"


def create_streamlit_app(llm, portfolio, clean_text):
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    st.title("📧 Cold Mail Generator")

    # UI inputs
    url_input = st.text_input(
        "Enter a URL:",
        value="https://remoteok.com/remote-jobs/remote-ai-research-engineer-job-223000",
        key="url_input"
    )
    submit_button = st.button("Submit", key="submit_button")

    if submit_button:
        try:
            st.write("🔄 Fetching job description from URL...")
            loader = WebBaseLoader([url_input])
            content = loader.load().pop().page_content
            st.write("✅ Page fetched successfully.")

            cleaned_data = clean_text(content)
            st.write("🧹 Cleaned job description:", cleaned_data[:500])  # show partial

            st.write("📂 Loading portfolio...")
            portfolio.load_portfolio()
            st.write("✅ Portfolio loaded.")

            st.write("🧠 Extracting jobs from description...")
            jobs = llm.extract_jobs(cleaned_data)
            st.write("📝 Jobs extracted:", jobs)

            if not jobs:
                st.warning("No jobs extracted. Check the job description or extraction logic.")

            for job in jobs:
                st.write("🔍 Working on job:", job)
                skills = job.get('skills', [])
                links = portfolio.query_links(skills)
                st.write("🔗 Found links:", links)

                email = llm.write_mail(job, links)
                st.code(email, language='markdown')

        except Exception as e:
            st.error(f"An error occurred while processing: {e}")
            st.exception(e)


if __name__ == "__main__":
    # Instantiate core classes
    chain = Chain()
    portfolio = Portfolio()

    # Launch Streamlit app
    create_streamlit_app(chain, portfolio, clean_text)

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI


# ---------------- LOAD ENVIRONMENT ----------------

load_dotenv()


# ---------------- MODEL ----------------

model = ChatMistralAI(
    model="mistral-small-latest"
)


# ---------------- PROMPT TEMPLATE ----------------

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a movie information extraction assistant.

Analyze the movie paragraph provided by the user and extract the
most useful information.

Return the information in exactly this format:

Movie Title:
Release Year:
Genre:
Director:
Main Cast:
Plot:
IMDb Rating:
Key Themes:
Notable Elements:
Quick Summary:

Instructions:
- Extract information only from the given paragraph.
- Do not invent or assume information that is not mentioned.
- If a field is not available, write "Not mentioned".
- Keep the Plot concise (2-3 sentences).
- Keep the Quick Summary very short (1-2 sentences).
- For Main Cast, list the important actors mentioned.
- For Key Themes, identify the major themes discussed.
- For Notable Elements, mention important details such as:
  soundtrack, visual style, scientific accuracy, critical reception,
  or cultural significance if mentioned.
- Keep the overall response concise and easy to read.
"""
    ),
    (
        "human",
        """
Here is the movie paragraph:

{movie_paragraph}
"""
    )
])


# ---------------- STREAMLIT CONFIG ----------------

st.set_page_config(
    page_title="Movie Information Extractor",
    page_icon="🎬",
    layout="centered"
)


# ---------------- HEADER ----------------

st.title("🎬 Movie Information Extractor")

st.caption(
    "Extract useful information from any movie paragraph using Mistral AI"
)


# ---------------- INPUT ----------------

movie_paragraph = st.text_area(
    "Enter Movie Paragraph",
    height=220,
    placeholder="Paste your movie paragraph here..."
)


# ---------------- BUTTON ----------------

if st.button(
    "🎯 Extract Information",
    use_container_width=True
):

    if not movie_paragraph.strip():

        st.warning("Please enter a movie paragraph first.")

    else:

        with st.spinner("Analyzing movie..."):

            try:

                # Create prompt
                final_prompt = prompt_template.invoke({
                    "movie_paragraph": movie_paragraph
                })

                # Get response
                response = model.invoke(final_prompt)

                # Store result in session state
                st.session_state["movie_result"] = response.text

            except Exception as e:

                st.error(
                    "Something went wrong while contacting Mistral."
                )

                st.caption(str(e))


# ---------------- RESULT ----------------

if "movie_result" in st.session_state:

    st.subheader("📋 Movie Information")

    st.text(st.session_state["movie_result"])
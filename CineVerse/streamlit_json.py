import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from pydantic import BaseModel, Field
from typing import List, Optional

from langchain_core.output_parsers import PydanticOutputParser


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# MODEL
# ============================================================

model = ChatMistralAI(
    model="mistral-small-latest"
)


# ============================================================
# PYDANTIC MODEL
# ============================================================

class Movie(BaseModel):

    title: str = Field(
        description="Movie title"
    )

    year: int = Field(
        description="Release year"
    )

    director: str = Field(
        description="Movie director"
    )

    genre: str = Field(
        description="Movie genre"
    )

    rating: float | None = Field(
        description="IMDb rating"
    )

    cast: List[str] = Field(
        description="Main cast"
    )

    themes: Optional[List[str]] = Field(
        description="Key themes"
    )


# ============================================================
# OUTPUT PARSER
# ============================================================

parser = PydanticOutputParser(
    pydantic_object=Movie
)


# ============================================================
# PROMPT TEMPLATE
# ============================================================

prompt_template = ChatPromptTemplate.from_messages([

    (
        "system",
        """
You are a movie information extraction assistant.

Extract the following information from the movie paragraph
provided by the user.

{format_instructions}

Important rules:

- Extract information only from the paragraph.
- Do not invent information.
- If IMDb rating is not mentioned, return null.
- If themes are not mentioned, return an empty list.
- Make sure the output follows the required format exactly.
"""
    ),

    (
        "human",
        """
Movie Paragraph:

{paragraph}
"""
    )

])


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Movie Information Extractor",
    page_icon="🎬",
    layout="centered"
)


# ============================================================
# HEADER
# ============================================================

st.title("🎬 Movie Information Extractor")

st.write(
    "Extract structured movie information using "
    "Mistral AI + Pydantic."
)


# ============================================================
# INPUT
# ============================================================

paragraph = st.text_area(
    "Enter Movie Paragraph",
    height=250,
    placeholder="Paste your movie paragraph here..."
)


# ============================================================
# BUTTON
# ============================================================

if st.button(
    "🎯 Extract Movie Information",
    use_container_width=True
):

    if not paragraph.strip():

        st.warning(
            "Please enter a movie paragraph first."
        )

    else:

        with st.spinner(
            "Extracting movie information..."
        ):

            try:

                # Create prompt
                final_prompt = prompt_template.invoke({
                    "paragraph": paragraph,
                    "format_instructions":
                        parser.get_format_instructions()
                })

                # Call Mistral
                response = model.invoke(
                    final_prompt
                )

                # Convert LLM response into Pydantic object
                movie = parser.parse(
                    response.text
                )

                # Save movie object
                st.session_state["movie"] = movie

            except Exception as e:

                st.error(
                    "Something went wrong while processing the movie."
                )

                st.exception(e)


# ============================================================
# DISPLAY JSON
# ============================================================

if "movie" in st.session_state:

    movie = st.session_state["movie"]

    st.divider()

    st.subheader("📦 Structured JSON Output")

    # Convert Pydantic object → Python dictionary
    movie_data = movie.model_dump()

    # Display dictionary as JSON
    st.json(movie_data)


    # ========================================================
    # OPTIONAL: INDIVIDUAL INFORMATION
    # ========================================================

    st.divider()

    st.subheader("🎬 Movie Details")

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Title:**", movie.title)

        st.write(
            "**Release Year:**",
            movie.year
        )

        st.write(
            "**Genre:**",
            movie.genre
        )

        st.write(
            "**Director:**",
            movie.director
        )

    with col2:

        st.write(
            "**IMDb Rating:**",
            movie.rating
            if movie.rating is not None
            else "Not mentioned"
        )


    st.subheader("👥 Main Cast")

    for actor in movie.cast:

        st.write(f"• {actor}")


    st.subheader("🧠 Key Themes")

    if movie.themes:

        for theme in movie.themes:

            st.write(f"• {theme}")

    else:

        st.write("Not mentioned")
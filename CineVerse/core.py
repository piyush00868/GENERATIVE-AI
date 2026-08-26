from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

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


# ---------------- USER INPUT ----------------

para = input("Enter the movie paragraph: ")


# ---------------- CREATE PROMPT ----------------

final_prompt = prompt_template.invoke({
    "movie_paragraph": para
})


# ---------------- GET RESPONSE ----------------

response = model.invoke(final_prompt)


# ---------------- DISPLAY RESULT ----------------

print("\n" + "=" * 60)
print("MOVIE INFORMATION")
print("=" * 60)

print(response.text)
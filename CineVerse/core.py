from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from  langchain_mistralai import ChatMistralAI

load_dotenv()  # Load environment variables from .env file  

model = ChatMistralAI(model="mistral-small-latest")
# ---------------- PROMPT TEMPLATE ----------------
prompt_template = PromptTemplate.from_template("""
You are a movie information extraction assistant.

Analyze the movie paragraph provided below and extract the most useful information.

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
- For Key Themes, identify the major themes discussed in the paragraph.
- For Notable Elements, mention important details such as awards, soundtrack, visual style, scientific accuracy, critical reception, or cultural significance if mentioned.
- Keep the overall response concise and easy to read.

Movie Paragraph:

{movie_paragraph}
""")

movie_paragraph = """
Interstellar is a visually stunning science fiction epic directed by Christopher Nolan.
Released in 2014, the film stars Matthew McConaughey, Anne Hathaway, Jessica Chastain,
and Michael Caine. The story revolves around a group of astronauts who travel through
a wormhole near Saturn in search of a new home for humanity as Earth faces environmental
collapse. The movie was widely appreciated for its emotional depth, scientific accuracy,
and Hans Zimmer's powerful soundtrack. It holds a rating of 8.6 on IMDb and is often
considered one of the greatest sci-fi films of the 21st century.

"""
# ---------------- CREATE PROMPT ----------------
prompt = prompt_template.invoke({
    "movie_paragraph": movie_paragraph
})
# ---------------- GET RESPONSE ----------------
response = model.invoke(prompt)

print("\n" + "=" * 60)
print("MOVIE INFORMATION")
print("=" * 60)

print(response.text)

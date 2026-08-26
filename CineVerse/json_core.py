from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

model = ChatMistralAI(
    model="mistral-small-latest"
)


class Movie(BaseModel):
    title: str = Field(description="Movie title")
    year: int = Field(description="Release year")
    director: str = Field(description="Movie director")
    genre: str = Field(description="Movie genre")
    rating: float | None = Field(description="IMDb rating")
    cast : List[str] = Field(description="Main cast")
    themes: Optional[List[str]] = Field(description="Key themes")

parser = PydanticOutputParser(pydantic_object=Movie)

prompt_template = ChatPromptTemplate.from_messages([
    ('system',"""
Extract the following information from the movie paragraph provided by the user
{format_instructions}
"""),
("human", "{paragraph}")
]
)

# ---------------- USER INPUT ----------------

para = input("Enter the movie paragraph: ")


# ---------------- CREATE PROMPT ----------------

final_prompt = prompt_template.invoke({
    "paragraph": para,
    'format_instructions': parser.get_format_instructions()
})


# ---------------- GET RESPONSE ----------------

response = model.invoke(final_prompt)
movie_data = parser.parse(response.text)

# ---------------- DISPLAY RESULT ----------------

print("\n" + "=" * 60)
print("MOVIE INFORMATION")
print("=" * 60)

print(response.text)
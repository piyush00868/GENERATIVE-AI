from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    "gemini-3-flash-preview",
    model_provider="google_genai"
)

response = model.invoke("What is Python?")
print(response.text)
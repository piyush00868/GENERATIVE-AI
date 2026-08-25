from dotenv import load_dotenv
from  langchain_mistralai import ChatMistralAI

load_dotenv()  # Load environment variables from .env file  

model = ChatMistralAI(model="mistral-small-latest", temperature=0.1)
response = model.invoke("What is python and use cases of it?")
print(response.text)
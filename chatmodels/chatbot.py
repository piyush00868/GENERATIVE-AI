from dotenv import load_dotenv
from  langchain_mistralai import ChatMistralAI

load_dotenv()  # Load environment variables from .env file  

model = ChatMistralAI(model="mistral-small-latest")
while True:
    print("-------------------------------- welcome to chat with Mistral AI --------------------------------")
    prompt = input("You: ")
    if prompt == "0":
        break
    response = model.invoke(prompt)
    print("Mistral AI:", response.text)
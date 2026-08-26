from dotenv import load_dotenv
from  langchain_mistralai import ChatMistralAI

load_dotenv()  # Load environment variables from .env file  

model = ChatMistralAI(model="mistral-small-latest")
#  add message history to the model
messages = []
print("-------------------------------- welcome to chat with Mistral AI --------------------------------")
while True:
    prompt = input("You: ")
    if prompt == "0":
        break
    messages.append({"role": "user", "content": prompt})
    response = model.invoke(messages)
    messages.append({"role": "assistant", "content": response.text})
    print("Mistral AI:", response.text)
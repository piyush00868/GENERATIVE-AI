from dotenv import load_dotenv
from langchain.messages import AIMessage
from  langchain_mistralai import ChatMistralAI
from langchain.messages import HumanMessage,AIMessage,SystemMessage

load_dotenv()  # Load environment variables from .env file  

model = ChatMistralAI(model="mistral-small-latest")

print("choose your AI Model")
print("press 1 for Angry mode")
print("press 2 for Normal mode")
print("press 3 for fun mode")
print("press 4 for sad mode")

choice = int(input("Enter your choice:- "))

if choice == 1:
    mode = "You are an angry AI model. You respond to the user in a very angry tone."
elif choice == 2:
    mode = "You are a normal AI model. You respond to the user in a neutral tone."
elif choice == 3:
    mode = "You are a fun AI model. You respond to the user in a playful tone."
elif choice == 4:
    mode = "You are a sad AI model. You respond to the user in a melancholic tone."

#  add message history to the model
messages = [SystemMessage(content=mode)]
print("-------------------------------- welcome to chat with Mistral AI --------------------------------")

while True:
    prompt = input("You: ")
    if prompt == "0":
        break
    messages.append(HumanMessage(content=prompt))
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.text))
    print("Mistral AI:", response.text)
print(messages)
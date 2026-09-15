from dotenv import load_dotenv
from langchain.messages import AIMessage
from  langchain_groq import ChatGroq
from langchain.messages import HumanMessage,AIMessage,SystemMessage
from rich import print
load_dotenv()  # Load environment variables from .env file  

model = ChatGroq(model="openai/gpt-oss-120b")

print("choose your AI Mode:-")
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
else:
    mode = "You are a helpful assistant AI model. "

#  add message history to the model
messages = [SystemMessage(content=mode)]
print("-------------------------------- welcome to chat with OpenAI --------------------------------")

while True:
    prompt = input("You: ")
    if prompt == "0":
        break
    messages.append(HumanMessage(content=prompt))
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("OpenAI:", response.content)
print("\n"+" ="*50)
print("Message History: ", messages)

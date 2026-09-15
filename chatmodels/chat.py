from dotenv import load_dotenv
from  langchain_groq import ChatGroq

load_dotenv()  # Load environment variables from .env file  

model = ChatGroq(model="openai/gpt-oss-120b")
response = model.invoke("What is python and use cases of it?" )
print(response.text)
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
load_dotenv()  # Load environment variables from .env file

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    )

model = ChatHuggingFace(llm=llm)

response = model.invoke("Why is python so popular and what are the use cases of it?")
print(response.text)
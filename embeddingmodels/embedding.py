from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


vector = model.encode("My name is Piyush.")

print(vector)

print(len(vector))

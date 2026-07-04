from pathlib import Path

from medical_rag import build_qa_chain


qa_chain = build_qa_chain()
user_query = input("Write Query Here: ").strip()

if not user_query:
    raise ValueError("Please enter a medical question.")

response = qa_chain.invoke({"query": user_query})

print("\nRESULT:", response.get("result", "No answer found."))
print("\nSOURCES:")
for document in response.get("source_documents", []):
    source = Path(document.metadata.get("source", "Unknown source")).name
    page = document.metadata.get("page", "N/A")
    print(f"- {source}, page {page}")

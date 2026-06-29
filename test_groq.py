from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

API_KEY = "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"

print("🚀 Starting test...")

try:
    llm = ChatGroq(
        groq_api_key=API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.7
    )
    print("✅ Client initialized")

    response = llm.invoke([HumanMessage(content="Hello Groq! Can you confirm you are working?")])
    print("🤖 Groq Response:", response.content)

except Exception as e:
    print("❌ Error:", e)

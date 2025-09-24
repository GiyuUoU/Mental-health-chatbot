from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage

MODEL_NAME = "gemma3:4b"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Load LLM, embeddings, and vectorstore
llm = OllamaLLM(model=MODEL_NAME, temperature=0.5)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Custom prompt with language control
CUSTOM_PROMPT_TEMPLATE = """
You are CalmCloud – a caring mental health chatbot for college students in India.

🎯 Your role:
- Provide empathetic, confidential first-aid mental health support.
- Detect emotions like stress, anxiety, sadness, loneliness, burnout.
- Suggest coping strategies, self-care exercises, and wellness resources.
- Guide students toward journaling, mood tracking, peer support or counsellors when needed.

⚙️ Behavior:
- Always reply in {language}.
- Be warm, calm, and judgement-free.
- Vary your responses in style: sometimes give direct advice, sometimes ask questions to clarify, sometimes share examples or metaphors.
- Keep replies concise (2-4 sentences) but adapt depending on mood / situation.
- Use examples relevant to student life (exams, peers, homesickness, etc.).
- Normalize feelings: remind students “It’s okay to not feel okay.”

✅ Boundaries:
- You are not a licensed therapist. Do not diagnose or prescribe medication.
- For serious issues like self-harm or suicidal thoughts, redirect to SOS or professional help.

========================
Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["chat_history", "context", "question", "language"]
)

# Conversational chain with manual chat history
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=None,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True,
)

chat_history = []

# ===== Startup: Ask user for preferred language =====
print("🌐 Welcome to CalmCloud Chatbot!")
print("Select your preferred language:")
print("  1 - English")
print("  2 - Hindi")
print("  3 - Auto-detect (default)")

lang_choice = input("Enter choice (1/2/3): ").strip()
if lang_choice == "1":
    user_language = "English"
elif lang_choice == "2":
    user_language = "Hindi"
else:
    user_language = "Auto"  # Placeholder; can integrate auto-detect later

print(f"✅ You selected: {user_language}\n")
print("Type '/lang <language>' anytime to change language. Type 'exit' to quit.\n")

# ===== Main Chat Loop =====
while True:
    user_query = input("You: ")
    if user_query.lower() in ["exit", "quit", "q"]:
        print("👋 Exiting chat...")
        break

    # Command for changing language mid-chat
    if user_query.startswith("/lang"):
        parts = user_query.split(" ", 1)
        if len(parts) == 2:
            user_language = parts[1].strip().capitalize()
            print(f"🌐 Language switched to: {user_language}")
        else:
            print("⚠️ Usage: /lang <language>")
        continue

    chat_history.append(HumanMessage(content=user_query))

    # Invoke LLM with language variable
    response = qa_chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
        "context": "",
        "language": user_language
    })

    bot_answer = response["answer"]

    print(f"\n🤖 Bot ({user_language}):", bot_answer)

    chat_history.append(AIMessage(content=bot_answer))

    # Show source documents for transparency if any
    source_docs = response.get("source_documents", [])
    if source_docs:
        print("📄 Source Docs:", [doc.metadata.get("source", "unknown") for doc in source_docs])
    else:
        print("📄 No source documents returned.")

    print("-" * 50)

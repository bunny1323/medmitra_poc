
import httpx

from config import settings

SYSTEM_PROMPT = """
You are MedMitra AI, a professional healthcare assistant.

RULES:

1. Answer ONLY using the provided medical context.

2. Never answer non-medical questions.
If the question is not related to healthcare, medicines, diseases,
dosage, side effects, symptoms, or drug interactions, reply exactly:

"MedMitra AI is designed only for medicine and healthcare information."

3. For medicine information:

If user asks about uses:

📌 Uses
• item 1
• item 2

If user asks about dosage:

💊 Dosage
• item 1
• item 2

If user asks about side effects:

⚠️ Side Effects
• item 1
• item 2

If user asks about warnings:

🚨 Warnings
• item 1
• item 2

4. For drug interaction questions:

Start with:

⚠️ Drug Interaction Warning

Then:
• Explain the interaction
• Mention possible risks
• Keep response concise

5. For emergency situations:
Advise immediate emergency medical care.

6. Keep answers concise and professional.

7. Never invent information not present in the provided context.

8. End every medical response with:

Consult a healthcare professional for medical advice.
"""


async def generate_response(
    user_message: str,
    context: str,
    conversation_history: list[dict] | None = None,
) -> str:

    prompt = f"""
Medical Context:
{context}

Question:
{user_message}

Generate a professional healthcare response using the required format.
"""

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": settings.ollama_model,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "stream": False,
                "options": {
                    "temperature": 0.05,
                    "top_p": 0.7,
                    "num_predict": 250,
                },
            },
        )

        response.raise_for_status()

        data = response.json()

        return data["message"]["content"]


async def check_ollama_health() -> dict:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{settings.ollama_base_url}/api/tags"
            )

            response.raise_for_status()

            models = [
                model["name"]
                for model in response.json().get("models", [])
            ]

            model_available = any(
                settings.ollama_model in model
                for model in models
            )

            return {
                "status": "connected",
                "models": models,
                "target_model": settings.ollama_model,
                "model_available": model_available,
            }

    except Exception as exc:
        return {
            "status": "disconnected",
            "error": str(exc),
        }

import re

EMERGENCY_PATTERNS = [
    r"\b(chest pain|heart attack|cardiac arrest)\b",

    r"\b(can'?t breathe|cannot breathe|difficulty breathing|shortness of breath|choking|not breathing|breathing problem)\b",

    r"\b(severe bleeding|uncontrolled bleeding|hemorrhage|bleeding heavily|losing blood)\b",

    r"\b(stroke|face drooping|slurred speech|cannot speak properly|unable to speak|speech problem|sudden numbness)\b",

    r"\b(cannot move my arm|cannot move my leg|paralysis)\b",

    r"\b(seizure|convulsion|unconscious|unresponsive)\b",

    r"\b(anaphylaxis|severe allergic reaction|throat swelling)\b",

    r"\b(suicid(e|al)|self[- ]harm|kill myself|want to die)\b",

    r"\b(overdose|drug overdose|medicine overdose|poisoning|poisoned)\b",

    r"\b(swallowed too many tablets|took too many tablets|took too many pills)\b",

    r"\b(severe head injury|spinal injury|severe burn)\b",

    r"\b(call 911|call emergency|need ambulance)\b",
]

EMERGENCY_KEYWORDS = [
    "emergency",
    "dying",
    "life threatening",
    "life-threatening",
    "critical condition",
]


def detect_emergency(message: str) -> tuple[bool, str | None]:
    lower = message.lower()

    for pattern in EMERGENCY_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return True, (
                "This may be a medical emergency. Please call your local emergency "
                "number immediately (911 in the US, 112 in EU, 102/108 in India). "
                "Do not wait for AI advice."
            )

    for keyword in EMERGENCY_KEYWORDS:
        if keyword in lower:
            return True, (
                "Your message suggests an urgent medical situation. "
                "Seek immediate professional medical help."
            )

    return False, None
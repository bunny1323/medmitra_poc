# MedMitra — Test Cases & Expected Behaviour
Version 0.1.0-poc

---

## How to read this document

- **Expected: RELEVANT** — retrieval should return results with relevance ≥ 0.55
- **Expected: LOW RELEVANCE** — no confident match; polite "not found" message shown
- **Expected: EMERGENCY** — keyword rule fires before vector search; 108 / crisis line shown
- **Expected: NO CRASH** — app must not throw an error

---

## 1. Normal Symptom Queries

| Query | Expected behaviour |
|---|---|
| `fever, cough, body pain and weakness` | RELEVANT — likely returns respiratory/flu-type diseases |
| `runny nose, sneezing and sore throat` | RELEVANT — likely returns common cold / allergic rhinitis |
| `severe headache and sensitivity to light` | RELEVANT — likely returns migraine / headache conditions |
| `wheezing and chest tightness` | RELEVANT — likely returns asthma / bronchospasm |
| `stomach pain and vomiting` | RELEVANT — likely returns gastritis / gastroenteritis |
| `anxiety, palpitations and shortness of breath` | RELEVANT — likely returns panic disorder |
| `joint pain swelling fingers` | RELEVANT — likely returns arthritis-type conditions |

---

## 2. Medicine Queries

> ⚠️ The medicine dataset is SYNTHETIC. Real drug names like paracetamol
> will be matched by category/indication similarity, not exact name.

| Query | Expected behaviour |
|---|---|
| `medicine for fever and pain` | RELEVANT — category: Antipyretic / Analgesic |
| `antibiotic for wound infection` | RELEVANT — category: Antibiotic, indication: Wound |
| `antifungal cream treatment` | RELEVANT — category: Antifungal |
| `medicine for depression` | RELEVANT — category: Antidepressant |
| `antiviral for virus` | RELEVANT — category: Antiviral |
| `medicine for diabetes` | RELEVANT — category: Antidiabetic |
| `antiseptic wound care` | RELEVANT — category: Antiseptic |

### Typo tolerance

| Query | Expected behaviour |
|---|---|
| `paracitamol` (typo) | LOW RELEVANCE or RELEVANT (semantic embeddings handle minor typos) |
| `cetrizine` (typo) | LOW RELEVANCE or LOW RELEVANCE (close to cetirizine — not in dataset) |

---

## 3. Emergency Tests

> These MUST trigger the emergency alert before any vector search.

| Query | Expected rule | Message shown |
|---|---|---|
| `I have severe chest pain and difficulty breathing` | cardiac + breathing | 108 emergency call |
| `patient is unconscious and not responding` | unconscious | recovery position + 108 |
| `severe bleeding after accident` | bleeding | apply pressure + 108 |
| `I think I am having a stroke` | stroke | FAST acronym + 108 |
| `patient is having a seizure` | seizure | clear area + 108 |
| `I have taken an overdose of medication` | overdose | Poison Control 1800-11-6117 |
| `I have suicidal thoughts` | mental_health | iCall 9152987821 |
| `throat swelling, severe allergic reaction` | anaphylaxis | EpiPen + 108 |

---

## 4. Edge Cases

| Query | Expected behaviour |
|---|---|
| *(empty input)* | Input box clears; search button does nothing |
| `how do I build a React website` | LOW RELEVANCE — medical search finds nothing useful |
| `@#$%^& !!!` | LOW RELEVANCE — no crash |
| `FEVER COUGH BODY PAIN` (uppercase) | RELEVANT — lowercased internally |
| Very long query (300+ chars) | Handled — truncated at 500 chars by input box |
| Search with Groq key missing | Shows context summary fallback — no crash |

---

## 5. Safety Rules (must pass every time)

- [ ] App never claims a **confirmed diagnosis**
- [ ] App never **prescribes dosage**
- [ ] App never **crashes** when GROQ_API_KEY is absent
- [ ] App shows **emergency message before** any retrieval for emergencies
- [ ] Low-relevance queries return **polite "not found"**, not hallucinated results
- [ ] JSON preview always shows correct **source type** (disease / medicine / emergency)
- [ ] **Disclaimer** is visible after every non-emergency response

---

## 6. Relevance Score Reference

```
≥ 0.65   → High relevance (green) — strong semantic match
0.55–0.64 → Acceptable (yellow) — Llama is called
< 0.55   → Low relevance (red) — "not found" returned, Llama NOT called
```

> Scores are cosine similarity values — they are NOT diagnosis probabilities.

# Prescription Upload Testing Results

| File Name | Medicines Extracted | Unreadable Flag | Error | Notes |
|----------|----------------------|-----------------|-------|-------|
| 1000_F_56617167...jpg | Betaloc, Dorzolanidum, Cimetidine, Oxprelol | false | null | parsing works but names noisy |
| prescription-4-l.jpg | Paracetamol, Metronidazole, Papain, Cermina | true | null | partially correct |
| OIP.webp | none | false | Failed to parse model response into JSON | unclear / malformed model output |
| OIP (1).webp | none | false | Failed to parse model response into JSON | unclear / malformed model output |
| prescription-placeholder.webp | Lisinopril, Amoxicillin, Ibuprofen | false | null | good sample result |
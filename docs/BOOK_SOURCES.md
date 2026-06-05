# Approved Medical Document Sources

The **MedMitra ML Search Service** relies on official, clinical medical books and guidelines for grounding information retrieval. Due to copyright restrictions and distribution compliance rules, **the PDF source files are not bundled with the source code repository.** Developers must manually download and place these files into the `data/books/` directory.

---

## 1. ICMR Standard Treatment Workflows — Volume I
- **Full Title**: ICMR Standard Treatment Workflows of India — Volume I
- **Issuing Authority**: ICMR and Department of Health Research, Government of India
- **Purpose**: Primary adult knowledge source. Serves queries categorized under `adult_core`.
- **Retrieval Priority**: 110
- **Coverage**: Respiratory, ENT, cardiology, neurology, nephrology, and other core adult conditions.
- **Exact Expected Filename**: `icmr_stw_volume_1.pdf`
- **Corpus Metadata Mapping**:
  - `age_group`: `adult`
  - `topic_group`: `adult_core`
  - `authority`: `ICMR`

---

## 2. ICMR Standard Treatment Workflows — Volume III
- **Full Title**: ICMR Standard Treatment Workflows of India — Volume III
- **Issuing Authority**: ICMR and Department of Health Research, Government of India
- **Purpose**: Secondary adult knowledge source. Serves queries categorized under `adult_extended`.
- **Retrieval Priority**: 100
- **Coverage**: Endocrinology, gastroenterology, dermatology, and additional adult conditions.
- **Exact Expected Filename**: `icmr_stw_volume_3.pdf`
- **Corpus Metadata Mapping**:
  - `age_group`: `adult`
  - `topic_group`: `adult_extended`
  - `authority`: `ICMR`

---

## 3. World Health Organization: IMCI Chart Booklet
- **Full Title**: WHO Integrated Management of Childhood Illness — Chart Booklet
- **Issuing Authority**: World Health Organization (WHO)
- **Purpose**: Primary child symptom and danger-sign source. Serves child queries.
- **Retrieval Priority**: 120
- **Exact Expected Filename**: `who_imci_chart_booklet.pdf`
- **Corpus Metadata Mapping**:
  - `age_group`: `child`
  - `topic_group`: `pediatric`
  - `authority`: `WHO`

---

## Setup Instructions

1. Retrieve the three PDF files from the legal issuing authority platforms.
2. Verify they match the exact filenames listed above.
3. Save them directly inside:
   `medmitra_ml_service/data/books/`
4. Run the validation checks script to confirm checksum integrity and existence:
   ```bash
   python -m scripts.validate_pdfs
   ```

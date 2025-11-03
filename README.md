
---

## Quantitative Modeling (Data Integration & Index Creation)

### Data Sources

| Dataset | Source | Key Indicators Used |
|----------|---------|--------------------|
| WGI (Worldwide Governance Indicators) | World Bank | Corruption, Rule of Law, Stability, Voice & Accountability, Government Effectiveness, Regulatory Quality |
| QoG (Quality of Government) | University of Gothenburg | Governance Readiness, E-Government Index, Women-Business-Law Index |
| ND-GAIN (Climate Readiness Index) | Notre Dame Global Adaptation Initiative | Overall Score, Readiness, Vulnerability |
| CCKP (Climate Change Knowledge Portal) | World Bank | Annual Precipitation Mean (1991–2020) |
| WJP (Rule of Law Index) | World Justice Project | Conceptual alignment only |

---

### Methodology Summary

1. **Data Cleaning & Standardization:** Cleaned and merged 5 global datasets.  
2. **Feature Scaling:** Standardized using z-scores; reversed vulnerability.  
3. **PCA:** Extracted underlying factors explaining 82% variance.  
4. **K-Means:** Clustered countries into 3 groups (Advanced, Transitional, Vulnerable).  
5. **Index Construction:** Created equal-weight and PCA-weighted indices (r = 0.99).  
6. **Visualization:** Global choropleth maps using Plotly.

---

## AI-Powered Legal Document Analysis (Gemini RAG)

### Purpose
Complement quantitative data with a contextual understanding of national climate laws using AI.

### Countries Included (6 representative regions)

| Country | Document Types |
|----------|----------------|
| Bangladesh | Climate Change Trust Act + NAP |
| Kenya | Climate Change Act + NAP |
| Fiji | Climate Change Act + NAP |
| Pakistan | Climate Change Act + NAP |
| Chile | Climate Framework Law + NAP |
| South Africa | Climate Change Bill + NAP |

**Why limited countries?**
- National climate laws are long, multilingual, and inconsistent in format.  
- Selecting one country per region ensured global coverage while keeping the dataset small for local computation.  
- This allowed controlled translation, accurate chunking, and cost-free Gemini API usage.

###  RAG Pipeline Workflow
1. Extract & translate policy PDFs.  
2. Chunk and embed texts with `all-MiniLM-L6-v2`.  
3. Store embeddings in ChromaDB.  
4. Use Gemini 2.5 Flash to answer questions using retrieved context.  
5. Display answers with transparent citations in Streamlit.

---

## Streamlit App Integration

| Tab | Function |
|-----|-----------|
| CLRI Map | Global readiness visualization |
| Policy Q&A | Gemini-powered retrieval and summarization |

---
## Results Summary

- PCA explained ~82% variance.  
- CLRI rankings are consistent across methods.  
- AI Q&A is accurate and context-based.  
- Integration works seamlessly across both tracks.

---

## Conclusion

The **Climate Legal Readiness Index (CLRI)** combines data science and AI to measure how well nations' legal and governance frameworks are prepared for climate action.  
It bridges quantitative rigor with qualitative legal interpretation, a scalable model for climate policy analytics.

---
"""

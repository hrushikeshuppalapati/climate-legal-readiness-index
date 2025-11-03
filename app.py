import streamlit as st
import pandas as pd
import plotly.express as px
import os
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import country_converter as coco

st.set_page_config(
    page_title="Climate Legal Readiness Index & Policy Q&A",
    page_icon="🌍",
    layout="wide",
)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data", "processed")
VECTOR_DIR = os.path.join(DATA_DIR, "vectorstore")
cc = coco.CountryConverter()
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=VECTOR_DIR)
collection = client.get_or_create_collection(name="climate_laws_nap")

api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    st.warning(" Gemini API key not found. Set GEMINI_API_KEY in Streamlit secrets or environment.")
else:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

st.markdown(
    """
    # 🌍 Climate Legal Readiness Index & Policy Q&A
    Integrating quantitative readiness assessment with AI-powered legal-document analysis.
    """
)

tab1, tab2 = st.tabs(["📊 CLRI Global Map (Track 1)", "💬 Policy Q&A (Track 2)"])

with tab1:
    st.subheader("Climate Legal Readiness Index (CLRI)")
    try:
        clri_path = os.path.join(DATA_DIR, "countries_clri_combined.csv")
        df = pd.read_csv(clri_path)

        if "iso_a3" not in df.columns:
            df["iso_a3"] = cc.convert(df["country"], to="ISO3")
            df.to_csv(clri_path, index=False)

        color_col = (
            "CLRI_PCA_z"
            if "CLRI_PCA_z" in df.columns
            else "CLRI" if "CLRI" in df.columns else df.columns[1]
        )

        fig = px.choropleth(
            df,
            locations="iso_a3",
            color=color_col,
            hover_name="country",
            color_continuous_scale="Viridis",
            title=f"Global Climate Legal Readiness Index ({color_col})",
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Track 1 Error: {e}")
        st.warning("CLRI data could not be loaded.")


with tab2:
    st.subheader("Ask Questions About Climate Laws & Policies (Gemini RAG)")


    try:
        metas = collection.get(include=["metadatas"])["metadatas"]
        country_options = ["General / Analogy"] + sorted(list({m["country"] for m in metas}))
    except Exception:
        country_options = ["General / Analogy"]

    country_choice = st.selectbox("Select Country Focus:", country_options)
    user_query = st.text_input("Enter your question:")

    if st.button("Get Answer") and user_query.strip():
        with st.spinner("Retrieving context from vector database …"):
            try:
                q_emb = embedder.encode([user_query])[0]

                if country_choice != "General / Analogy":
                    res = collection.query(
                        query_embeddings=[q_emb],
                        n_results=5,
                        where={"country": country_choice},
                        include=["documents", "metadatas"],
                    )
                else:
                    res = collection.query(
                        query_embeddings=[q_emb],
                        n_results=5,
                        include=["documents", "metadatas"],
                    )

                docs = res["documents"][0]
                metas = res["metadatas"][0]

                context = ""
                for i, (d, m) in enumerate(zip(docs, metas), start=1):
                    context += f"[Source {i}: {m['country']} ({m['doc_type']})]\n{d}\n\n"

                system_prompt = """
You are an expert global-climate-policy analyst.
Use ONLY the provided context from national climate laws and NAPs
to answer the user’s question concisely and factually.
Cite sources as [Source #].  If the question refers to a non-indexed country,
respond by analogy from similar regional examples.
"""
                full_prompt = f"{system_prompt}\n\nQuestion:\n{user_query}\n\nContext:\n{context}"

                if not api_key:
                    st.error("Gemini API key missing – cannot generate answer.")
                else:
                    with st.spinner("…"):
                        response = model.generate_content(full_prompt)
                        answer = response.text

                    st.markdown("### Answer")
                    st.write(answer)

                with st.expander("View Retrieved Context (Top 5)"):
                    for i, (d, m) in enumerate(zip(docs, metas), start=1):
                        st.markdown(f"**Source {i}:** {m['country']} – {m['doc_type']}")
                        st.caption(f"File: {m['source_file']}")
                        st.write(d[:600] + "…")
                        st.divider()

            except Exception as e:
                st.error(f"Error retrieving context or querying Gemini: {e}")

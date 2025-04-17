import streamlit as st
from utils.file_loader import load_text_from_file
from utils.scoring import generate_scores_and_feedback
import pandas as pd



#top name 
st.set_page_config(page_title="Synopsis Checker", layout="centered")
#titlename 
st.title("Synopsis Checker")
#markdown
st.markdown("Upload an **article** and its **synopsis** to evaluate how well it summarizes the content.")

#mode seletor
mode= st.selectbox("Choose LLM Mode:",  ["Local", "ChatGPT"], index=0)

#upload article and syonpsis
article = st.file_uploader("Upload the main article (.txt or .pdf)", type=["txt", "pdf"])
synopsis = st.file_uploader("Upload the synopsis (.txt or .pdf)",type=["txt", "pdf"])                    

#butuon if action is done
if st.button("Analyze"):

    if not article :
        st.error("Please upload the article file!")
    elif not synopsis:
        st.error("Please upload the syonpsis file!")
    else:
        text_from_article = load_text_from_file(article)
        text_from_synopsis = load_text_from_file(synopsis)

        with st.spinner("Analyzing...!!"):
            scores, feedback = generate_scores_and_feedback(text_from_article, text_from_synopsis, mode=mode)

        #make table format 
        score_table = {
            "Properties": [],
            "Value": []
        }

        #max scores to cole like __/25
        score_max = {
            "Content Coverage": 25,
            "Clarity": 25,
            "Coherence": 25,
            "Factual Consistency": 25
        }
        rows=[] #make list to avoid default numbers in UI
        for key, value in scores.items():
            rows.append({
                "Properties": key,
                "Value": f"{value}/{score_max.get(key, 25)}"
            })

        #make it into dataframe
        df = pd.DataFrame(rows)

        #print table without index
        st.subheader("Evaluation Score")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("Feedback")
        for key, value in feedback.items():
            st.markdown(f"**{key}:** {value}")


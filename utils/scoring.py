from pathlib import Path
import re
from utils.model_loader import local_llm
from utils.embedding_utils import get_semantic_summary
from utils.anonymize import anonymize_text
from utils.openai_api import call_chatgpt

prompt_path = Path("prompts/prompt.txt")

#read prompt
def get_prompt_Template():
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()
#correct the prompt in correct format
def format_prompt(article_summary: str, synopsis_summary: str) -> str:
    template = get_prompt_Template()

    #strip both summary of blank spaces
    cleaned_article = article_summary.strip()
    cleaned_synopsis = synopsis_summary.strip()

    #remove placeholder [[Article]] and [[synopsis]] and make final prompt
    prompt = template.replace("[[ARTICLE]]", cleaned_article)
    prompt = prompt.replace("[[SYNOPSIS]]", cleaned_synopsis)
    return prompt

def get_score_feedback(llm_output):
    scores = {}
    feedback = {}

    #take scores like: - Content Coverage: 40/50
    score_pattern = re.findall(r"- (.*?): (\d+)/\d+", llm_output)

    #take feedback after the "Feedback:" section
    feedback_section = llm_output.split("Feedback:")[-1]
    feedback_pattern = re.findall(r"- (.*?): (.*?)(?=\n- |\Z)", feedback_section, re.DOTALL)

    #align the scores 
    for name, value in score_pattern:
        scores[name.strip()] = int(value.strip())

    #make the feeback together
    for name, comment in feedback_pattern:
        feedback[name.strip()] = comment.strip().replace("\n", " ")

    return scores, feedback

def generate_scores_and_feedback(article, synopsis, mode="Local"):
    #get semantic summaries from embeddings
    article_summary = get_semantic_summary(article)
    synopsis_summary = get_semantic_summary(synopsis)
    
    #if chatght is being used anonymize the text from synopsis n article
    if mode == "ChatGPT":
        article_summary = anonymize_text(article_summary)
        synopsis_summary = anonymize_text(synopsis_summary)
        print("RAW article_summary prompt:\n", article_summary,"\n\n")
        print("RAW synopsis_summary prompt:\n", synopsis_summary,"\n\n")

    #make prompt for llama model
    prompt = format_prompt(article_summary, synopsis_summary)
    #check prompt being sent in correctt format
    #print("\n\nPROMPT BEING SENT TO LLM:\n", prompt, "\n\n")

    #decide local or chatgpt
    if mode == "Local":
        llm = local_llm()
        #invoke local model 
        output = llm.invoke(prompt)
    elif mode == "ChatGPT":
        output = call_chatgpt(prompt)
    else:
        raise NotImplementedError("Only local mode is implemented for now.")

    
    #check raw output
    #print("RAW MODEL OUTPUT:\n", output)

    #put output into correct format to display using get_score_feeback
    scores, feedback = get_score_feedback(output)

    return scores, feedback
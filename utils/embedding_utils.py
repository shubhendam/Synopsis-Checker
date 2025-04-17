import spacy
import re
from sklearn.metrics.pairwise import cosine_similarity
from utils.embedding_loader import local_embedding_model

#load spaCy NLP model once can be downloaed in venv 
nlp = spacy.load("en_core_web_sm")

def split_into_sentences(text):
    doc = nlp(text)
    sentences = []

    #make each para to good proper english sentence
    for sentence in doc.sents:
        cleaned = sentence.text.strip()
        if cleaned:
            sentences.append(cleaned)
    return sentences

#dynamically decide how many ranks of sentence is good for our summary
def determine_top_k(num_sentences):
    if num_sentences <= 4:
        return 3
    elif num_sentences <= 8:
        return 6
    elif num_sentences <= 15:
        return 13
    else:
        return 17

#make ure factual data is always captured so boost that sentenceif it has any
def contains_factual_data(sentence):
    factual_data = [
        r'\b\d{4}\b',                              #years like 2025
        r'\b\d+(\.\d+)?\s?(%)\b',                  #percentages
        r'[$₹€£]\s?\d+',                           #currency
        r'\b\d+(\.\d+)?\s?(km|kg|tons|million|billion|crore)\b',  #overall units ( can be added more)
        r'\b\d+(\.\d+)?\b'                         #numbers any numbers
    ]
    for fact in factual_data:
        match = re.search(fact, sentence, re.IGNORECASE)
        if match:
            return True
    return False

#get summary of single para graph
def summarize_paragraph(text):
    sentences = split_into_sentences(text)
    
    top_k = determine_top_k(len(sentences)) #tok_k as in how many sentece to take based on para size
    embedder = local_embedding_model() #load embedding model

    #sentence embedding all sentences 
    sentence_embeddings = embedder.embed_documents(sentences)

    #take avg embedding to figure out what would be the best points of topic in the sentences overall
    avg_embedding = []
    for col in zip(*sentence_embeddings): #zip will gve column wise grp for calculating avg
        avg_val = sum(col) / len(col)
        avg_embedding.append(avg_val)

    #use cosine similarity to compair each sentence with full paragrappg 
    similarities = cosine_similarity([avg_embedding], sentence_embeddings)[0]

    #add extra points to boost factually correct sentences 
    for i in range(len(sentences)):
        if contains_factual_data(sentences[i]):
            similarities[i] += 0.2

    #rant with higher = more important
    ranked_indices = list(range(len(similarities)))
    ranked_indices.sort(key=lambda i: similarities[i], reverse=True)

    # Pick top-k most relevant sentences and keep their original order
    top_indices = ranked_indices[:top_k]
    top_indices.sort()

    summary_sentences = []
    for i in top_indices:
        summary_sentences.append(sentences[i])

    return " ".join(summary_sentences) #make list to sentense and sent it 

def get_semantic_summary(text,):
    paragraphs = text.split("\n\n")
    summary_parts = []

    for para in paragraphs:
        para = para.strip()
        if para:
            summary = summarize_paragraph(para)
            if summary:
                summary_parts.append(summary)

    return " ".join(summary_parts)






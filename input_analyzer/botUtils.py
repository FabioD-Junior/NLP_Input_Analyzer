## Bot.utils by Fabio Duarte Junior
## Imports
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
import warnings
import re
import time

warnings.filterwarnings("ignore")

nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')
stop_words = stopwords.words('english')

KB_CSV_PATH = "qa2.csv"
DEBUG_MODE  = True
knowledge_base = pd.DataFrame()
vec_kbase=""

profits = {
    "tesla": {
        2019: 1500000,
        2020: 1700000,
        2021: 1700000,
        2022: 1766600,
        2023: 1234000,        
        2024: 22300000
    },
    "apple": {
        2019: 2500000,
        2020: 3700000,
        2021: 4700000,
        2022: 10766600,
        2023: 7234000,        
        2024: 8230000
    },
    "scotia bank": {
        2019: 2500000,
        2020: 3700000,
        2021: 4700000,
        2022: 10766600,
        2023: 7234000,        
        2024: 8230000
    },    
    "ibm": {
        2019: 2500000,
        2020: 3700000,
        2021: 4700000,
        2022: 10766600,
        2023: 7234000,        
        2024: 8230000
    }    
}

def load_database(numbers_to_text=True):
    global knowledge_base
    global vec_kbase
    
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    knowledge_base = pd.read_csv(KB_CSV_PATH)
    if numbers_to_text:
        knowledge_base['message'] = knowledge_base['message'].apply(transcribe_numbers)

    knowledge_base['processed_texts'] = knowledge_base['message'].apply(preprocess_text)
        
    vec_kbase = knowledge_base['processed_texts'].tolist()
                                          
def get_sentiment(message):
    import spacy
    from spacytextblob.spacytextblob import SpacyTextBlob

    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')
    doc = nlp(message)

    return doc._.polarity

def get_year(message):
    import re
    year = re.search(r'\b(19\d{2}|20\d{2})\b', message)
    if year:
        return year.group(0)
    else:
        return None

def get_entities(message):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(message)
    org = None
    entity_list = []
    entity_dict = dict()
    
    for entity in doc.ents:
        entity_list.append(entity.text + " : " + entity.label_)
        entity_dict[entity.text] = entity.label_
        if entity.label_=="ORG" :
            org = preprocess_text(entity.text)
    if len(entity_list) > 0:
        return "<br>".join(entity_list), org, entity_dict
    else:
        return "No entity found for this message <br>", org, entity_dict

def preprocess_text(text):
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    lemmas_without_stopwords = [lemma for lemma in lemmas if lemma.lower() not in stop_words and lemma.isalpha()]

    return ' '.join(lemmas_without_stopwords)

def get_C_distance(message, numbers_to_text=True):
    
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    if numbers_to_text:
        message = preprocess_text(message)
                                         
    user_text_processed = preprocess_text(message)
 
    vectorizer = TfidfVectorizer()
    bow_matrix = vectorizer.fit_transform(vec_kbase + [user_text_processed])

    cosine_distances = cosine_similarity(bow_matrix[-1], bow_matrix[:-1])
   
    max_distance_index = cosine_distances.argmax()
     
    knowledge_base['distances'] = cosine_distances.flatten().tolist()

    return knowledge_base['message'].iloc[max_distance_index], knowledge_base['response'].iloc[max_distance_index], knowledge_base['distances'].iloc[max_distance_index], knowledge_base['intent'].iloc[max_distance_index]


def get_L_distance(message, numbers_to_text=True):
    import pandas as pd
    from Levenshtein import distance as levenshtein_distance
    
    if numbers_to_text:
        message = preprocess_text(message)

    user_text_processed = preprocess_text(message)
     
    distances = knowledge_base['processed_texts'].apply(lambda x: levenshtein_distance(user_text_processed, x))
    
    min_distance_index = distances.idxmin()
    
    return knowledge_base['message'].iloc[min_distance_index], knowledge_base['response'].iloc[min_distance_index], distances.iloc[min_distance_index], knowledge_base['intent'].iloc[min_distance_index]

def number_to_words(n):
    units = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    if 0 <= n < 10:
        return units[n]
    elif 10 <= n < 20:
        return teens[n-10]
    elif 20 <= n < 100:
        if n % 10 == 0:
            return tens[n//10]
        else:
            return tens[n//10] + "-" + units[n%10]
    elif 100 <= n < 1000:
        if n % 100 == 0:
            return units[n//100] + " hundred"
        else:
            return units[n//100] + " hundred and " + number_to_words(n % 100)
    elif 1000 <= n < 10000:
        if n % 1000 == 0:
            return units[n//1000] + " thousand"
        else:
            return units[n//1000] + " thousand " + number_to_words(n % 1000)
    else:
        return str(n)

def transcribe_numbers(message):
    import re
    def replace_number(match):
        number = int(match.group(0))
        return number_to_words(number)

    return re.sub(r'\b\d+\b', replace_number, message)

def get_generic_response(value):
    import random

    generic_responses = [
        "The answer you're looking for is `<value>`.",
        "That would be `<value>`.",
        "Definitely, it's `<value>`.",
        "Without a doubt, `<value>` is what you're seeking.",
        "I can confirm that it's `<value>`.",
        "Based on what you said, it would be `<value>`.",
        "From what I understand, `<value>` is the answer.",
        "After a quick check, I can tell you it's `<value>`.",
        "Undoubtedly, it's `<value>`.",
        "Certainly, we can say it's `<value>`.",
        "It appears to be `<value>`.",
        "The solution appears to be `<value>`.",
        "Your answer seems to be `<value>`.",
        "Looks like `<value>` is the key.",
        "Evidently, `<value>` is the correct response.",
        "Clearly, the response is `<value>`.",
        "It's safe to say that `<value>` is the answer.",
        "Yes, the outcome is `<value>`.",
        "Affirmatively, it's `<value>`.",
        "Indeed, `<value>` is the right answer.",
        "Correct, that would be `<value>`.",
        "Precisely, it's `<value>`.",
        "Absolutely, the answer is `<value>`.",
        "Yes, it's confirmed to be `<value>`.",
        "Surely, the response you need is `<value>`.",
        "It's clear that the answer is `<value>`.",
        "Yes, without any doubt, it's `<value>`.",
        "Certainly, that's `<value>`.",
        "Unquestionably, it's `<value>`.",
        "Inarguably, the answer is `<value>`.",
        "Indeed, it's `<value>`.",
        "Without hesitation, it's `<value>`.",
        "Yes, it turns out to be `<value>`.",
        "For sure, it's `<value>`.",
        "Absolutely, it's `<value>`.",
        "The straightforward answer is `<value>`.",
        "Conclusively, it's `<value>`.",
        "Yes, the precise answer is `<value>`.",
        "It's unmistakably `<value>`.",
        "By all accounts, it's `<value>`.",
        "It's confirmed that the answer is `<value>`.",
        "Yes, it's definitively `<value>`.",
        "Clearly, without question, it's `<value>`.",
        "The evidence points to `<value>`.",
        "Undeniably, the answer is `<value>`.",
        "It's universally agreed that it's `<value>`.",
        "Without any contradiction, it's `<value>`.",
        "It's universally acknowledged as `<value>`.",
        "Beyond any doubt, it's `<value>`.",
        "It's universally accepted that `<value>` is the answer."
    ]
    follow_up_phrases = [
        "Is there anything else I can assist you with?",
        "Do you have any other questions?",
        "Can I help you with anything else?",
        "What else can I do for you today?",
        "Is there anything more you'd like to know?",
        "Would you like further assistance with anything?",
        "How else may I be of service to you?",
        "Any other inquiries or concerns I can address?",
        "What more can I help you with?",
        "Is there another area where you need assistance?",
        "Would you need help with anything else?",
        "Do you have any additional questions?",
        "Anything else on your mind?",
        "How can I further assist you?",
        "Is there more information you require?",
        "Are there any other topics you'd like to explore?",
        "Do you need clarification on anything else?",
        "Can I provide assistance with another matter?",
        "Would you like to ask about something else?",
        "Is there any other way I can be of help?",
        "Any more questions or concerns?",
        "How else can I support you?",
        "Do you have further inquiries?",
        "Would you like more information on another subject?",
        "Is there something else you're curious about?",
        "Need more assistance with anything?",
        "What else can I clarify for you?",
        "Can I offer help in any other areas?",
        "Do you wish to continue discussing any other topics?",
        "Is there any other assistance you require?",
        "Would you like to explore any other issues?",
        "Anything else you'd like assistance with?",
        "Are there more questions I can help with?",
        "How else may I assist you today?",
        "Do you have any other areas of interest you need help with?",
        "Would you like to inquire about something else?",
        "Is there another question I can answer for you?",
        "Any other matters you need help with?",
        "How can I further support your needs?",
        "Do you wish to ask anything else?",
        "Is there more help you need from me?",
        "Would you like further explanation on any topic?",
        "Is there another subject you need assistance with?",
        "Can I help clarify anything else for you?",
        "Would you like to discuss another matter?",
        "Any other assistance required?",
        "How else can I contribute to your understanding?",
        "Do you have any more queries or needs?",
        "Is there anything else on your mind that I can help with?",
        "Would you like any other information or assistance?"
    ]

    chosen_generic_response = random.choice(generic_responses)
    chosen_follow_up = random.choice(follow_up_phrases)
    return chosen_generic_response.replace("<value>", value) + "<br>" + chosen_follow_up

def func_profit(company, year):
    profit = None
    
    if company == None:
        return "In order to check the profit, please inform company and year"
    
    if year == None:
        return "In order to check the profit, please inform company and year"
    
    if int(year) > 0:
        profit = profits.get(company.lower(), {}).get(int(year)), None
                                                      
    if profit[0] == None:
        return "No profits found for this data, please inform company and year"                                              
    
    return "The profit for company : " + company + " year : " + year + " is :" + str(profit[0])

def func_time():
    from datetime import datetime
    current_time = datetime.now()
    print("time:", current_time)
    print(current_time.strftime("%H:%M:%S"))
    return str(current_time.strftime("%H:%M:%S") ) 

def toggle_debug():
    global DEBUG_MODE
    if DEBUG_MODE:
        DEBUG_MODE = False
    else:
        DEBUG_MODE = True
        
def generic_entities(message,entity_list):
    print(entity_list)
    words = message.split()
    for i, word in enumerate(words):
        if word in entity_list:
            words[i] = word.replace(word, str(entity_list[word]))
    return ' '.join(words)

def func_revenue(company, year,generic=False):
    profit = None
    
    if company == None:
        return "In order to check the revenue, please inform company and year"
    
    if year == None:
        return "In order to check the revenue, please inform company and year" 
    
    if int(year) > 0:
        profit = profits.get(company.lower(), {}).get(int(year)), None
    
    
    if profit[0] == None:
        return "No revenue found for this data, please inform company and year"                                              
    if generic:
        return get_generic_response(str(profit[0]))
    else:    
        return "The revenue for company  <b>" + company + " </b>, year <b> " + year + "</b> is :" + str(profit[0])

    
    
def message_analysis(message, debug_mode=True):
    debug_mode=DEBUG_MODE
    import time
    
    
    print("Step 0 ",time.time()) 
    step_begin = time.time()
    start = time.time()
    match, response, distance, intent = get_C_distance(message) 
    end  =  time.time()
    if debug_mode:
        print("Step 1 ", time.time())
        response_text = "<hr><h4>..:: Message Analysis ::..</h4> <hr>"     
        response_text = response_text + ":::☑︎<strong>Sentiment indicator: </strong>" + str(get_sentiment(message)) + "<br>"
        
        print("Step 2 ", time.time())
        response_text = response_text + ":::☑︎<i>Year :</i>" + str(get_year(message)) + "<br>"
        
        print("Step 3 ", time.time())
        response_text = response_text + ":::☑︎<b> Entities:</b> <br>"
        entities,org, dict_entities  = get_entities(message) 
        response_text = response_text + entities + "<hr>"
        
        
        print("Step 4 ", time.time())                                      
        response_text = response_text + ":::☑︎<b> Knowledge Base </b>"           + "<br>"
        
        response_text = response_text + "|➱<b> Cossine Similarity </b><small>" +str(end-start) +  "</small><br>"
        response_text = response_text + "☞ <u>Best match        : </u>" + match         + "<br>"
        response_text = response_text + "☞ <u>Intent            : </u>" + intent         + "<br>"
        response_text = response_text + "☞ <u>Potential response: </u>" + str(response) + "<br>"
        response_text = response_text + "☞ <u>Cossine Similarity: </u>" + str(distance) + "<br>"
        
        if len(str(response)) < 15:
            friendly_response = get_generic_response(str(response))
        else:
            friendly_response = response
        response_text = response_text + "☞ <u>Sugested response :</u>" + friendly_response + "<br><hr>"
        print(intent.lower())
        print(response)
        if intent.lower() =="function":
            if response=="func_profit":
                response_text = response_text + "☞ <u>Function response :</u>" + func_profit(org, (get_year(message))) + "<br><hr>"   
            if response=="func_time":
                response_text = response_text + "☞ <u>Function response :</u>" + get_generic_response(str(func_time()))   + "<br><hr>" 
            if response=="toggle_debug":
                toggle_debug()
                response_text = "Debug Mode toggled successfully"
        
        print("Step 5 ", time.time())
        start = time.time()
        l_match, l_response, l_distance, l_intent = get_L_distance(message) 
        end  =  time.time()
        
        response_text = response_text + "|➱<b> Levenshtein distance </b><small>" +str(end-start) +  "</small><br>"
        response_text = response_text + "☞ <u>Best match          : </u>" + l_match         + "<br>"
        response_text = response_text + "☞ <u>Intent              : </u>" + l_intent         + "<br>"
        response_text = response_text + "☞ <u>Potential response  : </u>" + str(l_response) + "<br>"
        response_text = response_text + "☞ <u>Levenshtein Distance: </u>" + str(l_distance) + "<br>" 

        if len(str(l_response)) < 15:
            friendly_response = get_generic_response(str(l_response))
        else:
            friendly_response = l_response
        response_text = response_text + "☞ <u>Sugested response :</u>" + friendly_response   + "<br><hr>"     
        
        print(l_intent.lower())
        
        if l_intent.lower() =="function":
            print(l_intent.lower())
            if l_response=="func_profit":
                response_text = response_text + "☞ <u>Function response :</u>" + func_profit(org, (get_year(message))) + "<br><hr>"
            
            if l_response=="func_time":
                response_text = response_text + "☞ <u>Function response :</u>" + get_generic_response(str(func_time()))   + "<br><hr>"     
        
        print("Step 6")
        generic_msg = generic_entities(message,dict_entities)
        
        match, response, distance, intent = get_C_distance(generic_msg) 
        
        l_match, l_response, l_distance, l_intent = get_L_distance(generic_msg) 
        
        response_text = response_text + "|➱<b> Generic entities replacement</b><br>"
        response_text = response_text + "☞ <u>New message         : </u>" + str(generic_msg)         + "<br>"
        response_text = response_text + "☞ <u>Cosin match: </u>" + match        + "<br>"
        response_text = response_text + "☞ <u>Levenshtein match: </u>" + str(l_match) + "<br>" 
        
        if intent.lower() =="function":
            response_text = response_text + "<hr> |➱ Cosin Function response <br>"            
            if response=="func_profit":
                response_text = response_text + "☞ <u>Function response :</u>" + func_profit(org, (get_year(message))) + "<br>"
            
            if response=="func_time":
                response_text = response_text + "☞ <u>Function response :</u>" + get_generic_response(str(func_time()))   + "<br>" 

            if response=="func_revenue":
                response_text = response_text + "✎" + func_revenue(org, get_year(message),generic=True) + "<br><hr>"

                
        if l_intent.lower() =="function":
            response_text = response_text + "|➱Levenshtein Function response <br>" 
           
            if l_response=="func_profit":
                response_text = response_text + "☞ <u>Function response :</u>" + func_profit(org, (get_year(message))) + "<br>"
            
            if l_response=="func_time":
                response_text = response_text + "☞ <u>Function response :</u>" + get_generic_response(str(func_time()))   + "<br>"                 
           
            if l_response=="func_revenue":
                response_text = response_text + "✎" + func_revenue(org, get_year(message),generic=True) + "<br><hr>"    
        
        return response_text
    else:
        return response 

import os
import time
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import BertTokenizer
import random
import os
import nltk
import collections
import string
import json, requests
import re

import json
import requests
from urllib.request import urlopen
import string
import pandas as pd

import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
import string 


def top_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
    """
    # batch support!
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
        logits = torch.where(logits < min_values, 
                             torch.ones_like(logits, dtype=logits.dtype) * -float('Inf'), 
                             logits)
    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        sorted_logits = sorted_logits.masked_fill_(sorted_indices_to_remove, filter_value)
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)
    
    return logits

def nGramBlock(sent, n):
    duplication = False
    tokens = sent
    ngram_list = []
    for i in range(len(tokens)-n + 1):
        ngram_list.append(tuple(tokens[i: i+n]))
    ngram_set = set(ngram_list)
    if len(ngram_set) != len(ngram_list):
        duplication = True
    return duplication


def preprocess(text):
	new_text = text.lower()
	punctuations = set(string.punctuation)
	returned_text = ""
	for idx in range(len(new_text)):
		chars = new_text[idx]
		if chars in punctuations:
			if idx-1 >= 0 and idx+1 < len(new_text):
				if new_text[idx-1].isalpha() and new_text[idx+1].isalpha():
					returned_text += new_text[idx]
		else:
			
			returned_text += new_text[idx]
	return returned_text

genre_dict = {'humor': 'comedy', 'science': 'sci-fi', 'animation': 'animation', 'animations': 'animation', 'animated': 'animation', 'comedy': 'comedy', 'comedies': 'comedy', 'funny': 'comedy', 'crime': 'crime', 'crimes': 'crime', 'thriller': 'thriller', 'thrillers': 'thriller', 'mystery': 'mystery', 'mysteries': 'mystery', 'musical': 'musical', 'biography': 'biography', 'biographies': 'biography', 'history': 'history', 'histories': 'history', 'romance': 'romance', 'romantic': 'romance', 'romantic': 'romance', 'sport': 'sport', 'western': 'western', 'documentary': 'documentary', 'documentaries': 'documentary', 'news': 'news', 'horror': 'horror', 'horrors': 'horror', 'scify': 'sci-fi', 'sci': 'sci-fi', 'scifi': 'sci-fi', 'sci fy': 'sci-fi', 'sci-fi': 'sci-fi', 'science fiction': 'sci-fi', 'super hero': 'action', 'superhero': 'action', 'adventure': 'adventure', 'drama':'drama', 'dramas': 'drama', 'family': 'family', 'war':'war', 'fantasy':'fantasy', 'action':'action'}
def label_genre(input_text, genres):
	new_text = []
	the_list = input_text.split(" ")
	idx_list = []
	for idx, token in enumerate(the_list):
		the_token = preprocess(token)
		if the_token in genre_dict:
			if the_token == "family" or the_token == "war" or the_token == "funny" or the_token =="mystery" or the_token =="animated" or the_token =="romantic":
				if idx+1 < len(the_list):
					next_token = the_list[idx+1]
					if "movie" in next_token:
						real_genre = genre_dict[preprocess(token)]
						if real_genre not in genres:
							genres[real_genre] = len(genres)
						genre_idx = genres[real_genre]
						new_text.append("[MOVIE_GENRE_"+str(genre_idx)+"]")
						idx_list.append(real_genre)
			elif the_token == "science":
				if idx+1 < len(the_list):
					next_token = the_list[idx+1]
					if "fiction" in next_token:
						real_genre = "sci-fi"
						if real_genre not in genres:
							genres[real_genre] = len(genres)
						genre_idx = genres[real_genre]
						new_text.append("[MOVIE_GENRE_"+str(genre_idx)+"]")
						idx_list.append(real_genre)
			elif the_token == "sci":
				if idx+1 < len(the_list):
					next_token = the_list[idx+1]
					if "fi" in next_token or "fy" in next_token:
						real_genre = "sci-fi"
						if real_genre not in genres:
							genres[real_genre] = len(genres)
						genre_idx = genres[real_genre]
						new_text.append("[MOVIE_GENRE_"+str(genre_idx)+"]")
						idx_list.append(real_genre)
			elif the_token == "humor":
				if idx+1 < len(the_list):
					next_token = the_list[idx+1]
					if "genre" in next_token:
						real_genre = "comedy"
						if real_genre not in genres:
							genres[real_genre] = len(genres)
						genre_idx = genres[real_genre]
						new_text.append("[MOVIE_GENRE_"+str(genre_idx)+"]")
						idx_list.append(real_genre)
			else:						
				real_genre = genre_dict[preprocess(token)]
				if real_genre not in genres:
					genres[real_genre] = len(genres)
				genre_idx = genres[real_genre]
				idx_list.append(real_genre)
				new_text.append("[MOVIE_GENRE_"+str(genre_idx)+"]")
		else:
			 new_text.append(token)
	returned_text = " ".join(new_text)
	if idx_list == []:
		returned_text = input_text
	return returned_text, genres, idx_list

def add_SEP(text, idx_list, case="genre"):
    if "[SEP]" not in text:
        text = text + " [SEP] "
    text += case + ": " + ", ".join(idx_list)
    return text+";"

def remove_duplicate_movie_plots(text):
    text = text.replace("by [MOVIE_P_DIRECTOR", "with [MOVIE_P_ACTOR")
    text = text.replace("P_DIRECTOR", "P_ACTOR")
    if debug_print:
        print("Before remove duplication: " + text)
    split_by_movie_plot = text.split("[MOVIE_PLOT]")
    if len(split_by_movie_plot) > 2:
        #remove the second plot
        mv_plot_counter = 0
        temp_list = []
        first = split_by_movie_plot[0].strip() + " [MOVIE_PLOT] "
        second = split_by_movie_plot[1].strip()
        #process second
        remove_last = second.split("<")
        second = "<".join(remove_last[:len(remove_last)-1])
        return (first + second).strip()
    else:
        return text

def convert_back(text, the_dict, proposed_result, case="GENRE", template_movie=None, template_genre="action"):

    if debug_print:
        print("orig: " + text)
    if "<credibility>" in text:
        the_text_list = text.split("<credibility>")
        text_result = []
        for idx, element in enumerate(the_text_list):
            if idx == 0:
                if element != "":
                    text_result.append(element)
            else:
                other_text = element.split("<")
                with_credibility = other_text[0]
                
                rest_of_text = "<".join(other_text[1:])
                tokenized_cred = with_credibility.split(" ")

                if len(tokenized_cred) < 15:
                    text_result.append(with_credibility)
                if rest_of_text != "":
                    rest_of_text = "<"+rest_of_text
                    text_result.append(rest_of_text)
                    
                if debug_print:
                    print("rest of text: " + rest_of_text)
                
        if text_result != []:
            text = ""
            for component in text_result:
                component = component.strip()
                if component[0] != "<":
                    component = "<credibility>"+component
                else:
                    component += "The plot is like this: [MOVIE_PLOT]" #updated here
                text += component + " "
            text = text.strip()
        else:
            text = "The plot is like this: [MOVIE_PLOT]"

        if debug_print:
            print("processed credibility: " + text)
    orig = text.split("[SEP]")
    text = orig[0].strip()
    
#     text = re.sub(r'\<[[a-z]*[_]*[[a-z]*\>', ' ', text).strip()
    text = re.sub(r'\<[[a-z]*[_]*[[a-z]*\>', '[TEMP_SPLIT]', text).strip()
    temp_sentences = text.split("[TEMP_SPLIT]")
    the_sentences = []
    for temp_sent in temp_sentences:
        tokenized_temp = temp_sent.strip().split(" ")
        last_token = tokenized_temp[len(tokenized_temp)-1].strip()

        if "[MOVIE_" in last_token and last_token[len(last_token)-1] == "]":
            if "did" in tokenized_temp[0].lower() or "do" in tokenized_temp[0].lower() or "have" in tokenized_temp[0].lower() or "who" in tokenized_temp[0].lower() or "what" in tokenized_temp[0].lower() or "how" in tokenized_temp[0].lower():
                tokenized_temp[len(tokenized_temp)-1] += " ?"
            else:
                if "[MOVIE_PLOT]" not in tokenized_temp[len(tokenized_temp) -1]:
                    tokenized_temp[len(tokenized_temp)-1] += " ."
        joined_tokens = " ".join(tokenized_temp)
        the_sentences.append(joined_tokens)
    
    text = " ".join(the_sentences)
    if debug_print:
        print("TEXT: " + text)
    
    new_text = []
    tokenized_text = text.split(" ")
    placeholder_id_to_text = {y:x for x,y in the_dict.items()}
    if debug_print:
        print(placeholder_id_to_text)
    i = 0
    mentioned_movie = ""
    has_the_word_and = False
    taken_idx = {}
    for token in tokenized_text:
        if "[MOVIE_"+case+"_" in token or (case == "TITLE" and "TITLE" in token):
            the_tokens = token.split("[MOVIE_")
            if debug_print:
                print("the tokens: " + str(the_tokens))
            for more_token in the_tokens:
                if case+"_" in more_token:
                    ending = more_token.split("]")
                    for ttoken in ending:
                        if  case+"_" in ttoken:
                            index_old = ttoken.split("_")[1]

                            new_idx = ""
                            for chars in index_old:
                                if chars != "]":
                                    new_idx += chars
                                else:
                                    break
                            new_idx = int(new_idx)

                            if new_idx in placeholder_id_to_text:
                               
                                if has_the_word_and and new_idx +1 in placeholder_id_to_text:
                                    new_idx = new_idx + 1
                                real_word = placeholder_id_to_text[new_idx]
                                
                                
                                taken_idx[new_idx] = True
                            else:
                                if i < len(proposed_result):
                                    real_word = proposed_result[i]
                                    i += 1
                                else:
                                    if case=="GENRE":
                                        real_word = template_genre
                                    elif case == "TITLE":
                                        if proposed_result == []:
                                            real_word = template_movie
                                            mentioned_movie = real_word
                                        else:
                                            real_word = proposed_result[i]
                                            mentioned_movie = real_word
                                            i += 1
                                if real_word not in the_dict:
                                    the_dict[real_word] = len(the_dict)
#                                 i += 1
                            if real_word != "":
                                if case == "TITLE":
                                    
                                    split_real_word = real_word.strip().split(" ")
                                    no_year = split_real_word[:len(split_real_word)-1]
                                    if debug_print:
                                        print("test new idx: no year" + (" ".join(no_year)).strip())
                                    new_text.append((" ".join(no_year)).strip())
                                    mentioned_movie = real_word
                                else:
                                    if real_word == "family" or real_word == "war":
                                        real_word += " movies"
                                    new_text.append(real_word.strip())
                                    mentioned_movie = real_word.strip() #it's actually genre here
                                
                        else:
                            if ttoken != "":
                                new_text.append(ttoken.strip())
                                if ttoken == "and":
                                    has_the_word_and =True
                else:
                    if more_token != "":
                        new_text.append(more_token.strip())
        else:
            new_text.append(token.strip())
            
            
    result_text = " ".join(new_text)
    if mentioned_movie != "":
        if case == "TITLE":
            movie_no_year_split = mentioned_movie.split(" ")
            movie_no_year = " ".join(movie_no_year_split[:len(movie_no_year_split)-1])
        else:
            movie_no_year = mentioned_movie

        fix_period_or_question_mark = result_text.split(movie_no_year)
        temp_list = []
        for text in fix_period_or_question_mark:
            clean_text = text.strip()
            if len(clean_text) >= 1 and clean_text[0] not in set(string.punctuation):
                clean_text = " "+clean_text
            temp_list.append(clean_text)
        result_text = (" " +movie_no_year).join(temp_list)

    return result_text, the_dict, mentioned_movie

def force_rec(text, the_dict, proposed_result, template_movie=None):
    if debug_print:
        print("FORC REC orig: " + text)
    text = re.sub(r'\<[[a-z]*[_]*[[a-z]*\>', ' ', text).strip()
    orig = text.split("[SEP]")
    text = orig[0].strip()
    new_text = []
    tokenized_text = text.split(" ")
    placeholder_id_to_text = {y:x for x,y in the_dict.items()}
    if debug_print:
        print(placeholder_id_to_text)
    i = 0
    template_genre = "action"
    mentioned_movie = ""
    for token in tokenized_text:
        if "TITLE_" in token:
            the_tokens = token.split("[MOVIE_")
            if debug_print:
                print("the tokens: " + str(the_tokens))
            for more_token in the_tokens:
                if "TITLE_" in more_token:
                    ending = more_token.split("]")
                    for ttoken in ending:
                        if "TITLE_" in ttoken:
                            real_word = template_movie
                            if i < len(proposed_result):
                                if proposed_result[i] not in the_dict:
                                    if debug_print:
                                        print("Force rec: " + proposed_result[i])
                                    real_word = proposed_result[i]
                                    i += 1
                                else:
                                    i +=1
                            else:
                                real_word = template_movie
                               
                            mentioned_movie = real_word
                            if real_word not in the_dict:
                                the_dict[real_word] = len(the_dict)

                            if real_word != "":
                                split_real_word = real_word.split(" ")
                                no_year = " ".join(split_real_word[:len(split_real_word)-1])
                                new_text.append(no_year.strip())
                            else:
                                real_word = template_movie
                                new_text.append(real_word.strip())
                        else:
                            if ttoken != "":
                                new_text.append(ttoken.strip())
                else:
                    if more_token != "":
                        new_text.append(more_token.strip())
        else:
            new_text.append(token.strip())
    
    result_text = " ".join(new_text)
    if mentioned_movie != "":
        
        movie_no_year_split = mentioned_movie.split(" ")
        movie_no_year = " ".join(movie_no_year_split[:len(movie_no_year_split)-1])

        fix_period_or_question_mark = result_text.split(movie_no_year)
        temp_list = []
        for text in fix_period_or_question_mark:
            clean_text = text.strip()
            if len(clean_text) >= 1 and clean_text[0] not in set(string.punctuation):
                clean_text = " "+clean_text
            temp_list.append(clean_text)

        result_text = (" " +movie_no_year).join(temp_list)
        
    return result_text.strip(), the_dict, mentioned_movie


MOVIE_URL= "MOVIE_NER_SERVICE_URL"  # test
def fetch_ner(last_bot_response, user_utterance, url=MOVIE_URL):
    data = {
        'last_bot_response': last_bot_response,
        'user_utterance': user_utterance
    }
    headers = {
        'Content-Type': 'application/json',
    }
    data = json.dumps(data)
    resp = requests.post(url, headers=headers, data=data, timeout=20)
    return resp.json()  


def min_edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1,s2 = s2,s1
    distances = range(len(s1) + 1)
    for index2,char2 in enumerate(s2):
        new_distances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1],
                                             distances[index1+1],
                                             new_distances[-1])))
        distances = new_distances

    return distances[-1]


def movie_placeholder(text, raw):
	tokens = text.split()
	
	indexer = 0
	new_text = []
	b_movie_counter = 0
	while indexer < len(tokens):
		token = tokens[indexer]
		tag = raw[indexer]
		if tag == 'O':
			new_text.append(token)
		else:
			if 'B-movies' in tag:
				new_text.append("[MOVIE_TITLE]")
				b_movie_counter += 1
			elif "-movies" not in tag:				
				new_text.append(token)
		indexer += 1
	text_with_placeholder = " ".join(new_text)
	return text_with_placeholder, b_movie_counter

def process_more_than_one_movie(movies, text, raw):
	tokens = text.split()
	temp = []
	movie_taken = {}
	counter = 0
	for movie in movies:		
		movie_taken[counter] = (movie, False)
		counter += 1

	counter = 0
	movie_list = []
	if debug_print:
		print("RAW: " + str(raw))
	b_eval = 0
	b_movie = 0
	for token, tag in zip(tokens, raw):
		if 'B-movies' in tag:
			temp.append(token)
			b_movie += 1
		elif 'I-movies' in tag:
			temp.append(token)
			if debug_print:
				print(temp)
		else:
			#evaluate temp
			new_temp = preprocess(" ".join(temp))
			if debug_print:
				print("ELSE: " + str(b_movie) + " " + str(b_eval) + " " + str(b_movie - 1))
			if b_eval == b_movie - 1:
				while counter < len(movies):
					if not movie_taken[counter][1]:
						title = movie_taken[counter][0]['title']
						year = movie_taken[counter][0]['year']

						score = min_edit_distance(new_temp, preprocess(title))
						if score < 3:
							movie_list.append(title + " ({})".format(year))
							temp = []
							b_eval += 1
							break
					counter += 1

	if b_eval == b_movie - 1:
		new_temp = preprocess(" ".join(temp))
		while counter < len(movies):
			if not movie_taken[counter][1]:
				title = movie_taken[counter][0]['title']
				year = movie_taken[counter][0]['year']

				score = min_edit_distance(new_temp, preprocess(title))
				if score < 3:
					movie_list.append(title + " ({})".format(year))
					temp = []
					b_eval += 1
					break
			counter += 1
	return movie_list

def get_multiple_movies(real_text, movie_list):
    result = []
    dup_movies = {}
    for movie in movie_list:
        title = movie['title']
        if debug_print:
            print("title: " + str(title))
        year = movie['year']
        movie_type = movie['type']

        if preprocess(title) in preprocess(real_text) and preprocess(title) not in dup_movies:
            result.append(title + " ({})".format(year))
            dup_movies[preprocess(title)] = 1
    return result

title_to_tmdb = {}
def get_movies(ner_result, raw_text, movie_counter):
	movie_list = ner_result['movie']
	raw = ner_result['raw'][0]
	text = preprocess(raw_text)
	if movie_list == []:
		return ""
	else:
		movies = []
		if movie_counter == 1:
			title = movie_list[0]['title']
			year = movie_list[0]['year']
			tmdb = movie_list[0]['tmdbId']
			tokenized_title = preprocess(title).split(" ")
			counter = 0
			movie_name = []
			for token in tokenized_title:
				if token not in stop_words:
					if token in text.split(" "):
						counter += 1

			tokenized_text = text.split(" ")
			for token, tag in zip(tokenized_text, raw):
				if "-movies" in tag:
					movie_name.append(token)

			total_len = len(tokenized_title)
			normalized = counter*100/total_len

			if total_len > 2:
				if normalized < 60:					
					return ""
			processed_movie_name = 	" ".join(movie_name)

			if total_len <= 2:
				if normalized < 50:
					if total_len == 2:
						new_title = title.replace(" ", "").lower()
						
						if new_title in preprocess(processed_movie_name):
							result_title = title + " ({})".format(year)
							title_to_tmdb[result_title] = tmdb
							return result_title
						else:
							return ""
					else:
						return ""
			title_to_tmdb[title + " ({})".format(year)] = tmdb
			return title + " ({})".format(year)
		elif movie_counter > 1:
			result = get_multiple_movies(text, movie_list)
			str_result = "; ".join(result)
			return str_result
		else:
			return ""

def process_movie_title(text_with_placeholder, movies, mentioned):
	text_split = text_with_placeholder.split(" ")
	movie_in_text = movies.split(";")
	taken_movie = []
	for movie in movie_in_text:
		if movie not in taken_movie:
			taken_movie.append(movie.strip())
	if debug_print:
		print(taken_movie)
	taken_movie.reverse()
	if debug_print:
		print("new: " + str(taken_movie))
		print("------------")
	result = []
	related_movie = ""
	for token in text_split:
		if token == "[MOVIE_TITLE]":
			if len(taken_movie) > 0:
				related_movie = taken_movie.pop()
			if related_movie not in mentioned:
				mv_idx = len(mentioned)
				mentioned[related_movie] = mv_idx

			the_idx = mentioned[related_movie]

			token = "[MOVIE_TITLE_{}]".format(the_idx)

		result.append(token)

	str_result = " ".join(result)
	return str_result, mentioned

stop_words = ["of", "in", "the", "to", "is", "a", "on", "into", "with"]
def create_movie_slot(rec_prev_utt, seeker_utt, movie_mentioned):
    rec = preprocess(re.sub(r'\<[[a-z]*[_]*[[a-z]*\>', ' ', rec_prev_utt).strip())
    if debug_print:
        print(rec)
    seek = preprocess(seeker_utt)
    ner_result = fetch_ner(rec, seek)
    if debug_print:
        print(ner_result)
    text = preprocess(seek)
    try:
        raw = ner_result['raw'][0]
    except:
        return seeker_utt, "", movie_mentioned

    text_with_placeholder, movie_counter = movie_placeholder(text, raw)
    movie_result = get_movies(ner_result, text, movie_counter)
    if debug_print:
        print("movie result: " + str(movie_result))
        print(text_with_placeholder)
        
    if movie_result == "":
        text_with_placeholder = seeker_utt
        mentioned = movie_mentioned
    else:
        if debug_print:
            print("debug here")
        text_with_placeholder, mentioned = process_movie_title(text_with_placeholder, movie_result, movie_mentioned)
        
    return text_with_placeholder, movie_result, mentioned

TMDB_KEY = "TMDB_KEY"
def load_movie_name(input_file):
	names = pd.read_csv(input_file, sep="\t")
	name_list = names['Name'].values

	return name_list

person_db = {}
name_list = load_movie_name("TSV_MOVIE_DATA_PATH")

def search_person(person_name):
	processed_person_name = person_name.replace(" ", "%20")
	the_url = "apiURL".format(TMDB_KEY, processed_person_name)
	person_info_json = urlopen(the_url).read().decode('utf8')
	person_info = json.loads(person_info_json)
	return person_info

def person_is_there(person_info):
	return person_info["total_results"] > 0

def find_name(new_text, name_db, name_dict, director_dict, people_dict):
	temp_text = new_text.split("[SEP]")
	new_text = temp_text[0]
	ending = ""
	if len(temp_text) > 1 and temp_text[1] != "":
		ending = "[SEP]" + temp_text[1]
	new_name_dict = name_dict
	new_director_dict = director_dict
	new_people_dict = people_dict
	punctuations = set(string.punctuation)
	returned_text = ""
	tokenized_text = new_text.encode('ascii', 'ignore').decode('ascii').split(" ")
	the_text = []
	people_names = []
	info_list = []
	index = 0
	people_indexer = 1

	while index < len(tokenized_text):
		token = tokenized_text[index]
		condition1 = len(token) > 1 and token[0].isupper() and index+1 < len(tokenized_text)
		condition2 = len(token) > 1 and (token[0] == "(" or token[0] == '"') and index+1 < len(tokenized_text) and token[1].isupper() and index+1 < len(tokenized_text) and token[len(token)-1] not in punctuations
		
		if condition1:
			name_token = token
			before_punct = ""

		the_punct = ""
		if condition2:
			name_token = token[1:]						
			before_punct = token[0]	

		processed = False
		after_punct = ""
		next_names = []
		if condition1 or condition2:

			
			next_idx = index+1
			while next_idx < len(tokenized_text):
				next_token = tokenized_text[next_idx]
				

				if len(next_token) > 1 and (next_token[0].isupper() or next_token == "de") and len(next_token) > 1:
					if name_token in name_db:
						
						#check if next token has punctuations?
						next_name = ""
						# print(next_token)
						end = len(next_token)
						if end >= 3 and next_token[:3] == "Jr.":
							init = 3
							next_name = "Jr."
						else:
							init = 0
						time_to_quit = False
						
						for pos in range(init, end):
							char = next_token[pos]
							if char not in punctuations:
								if not time_to_quit:
									next_name += char
							else:
								if (char == "'" or char == "-") and pos != len(next_token) -1 and next_token[pos+1] not in punctuations:
										next_name += char
								else:
									after_punct += char
									time_to_quit = True
									

						next_names.append(next_name)
						if time_to_quit:
							break
				else:
					break
				next_idx += 1

		if len(next_names) >= 1:
			
			the_names = " ".join(next_names)
			full_name = name_token + " " + the_names
			next_names = []
			result = search_person(full_name)
			
			if "Star Wars" not in full_name and "Captain America" not in full_name and "James Bond" not in full_name:
				if person_is_there(result):
					full_name_from_db = result["results"][0]["name"]
					edit_score = min_edit_distance(full_name, full_name_from_db)
					if edit_score <= 2:
						

						people_names.append(full_name)
						info = result["results"][0]
						person_id = info["id"]
						person_job = info["known_for_department"]
						person_top_movies = info["known_for"]
						if person_id not in person_db:
							person_db[person_id] = {"job": person_job, "top_movies": person_top_movies, "name": full_name}
						info_list.append(person_id)
						if person_job == "Acting":
							job_type = "ACTOR"
							if full_name in new_name_dict:
								name_idx = new_name_dict[full_name]
							else:
								name_idx = len(new_name_dict)
								new_name_dict[full_name] = name_idx

						elif person_job == "Directing":
							job_type = "DIRECTOR"
							if full_name in new_director_dict:
								name_idx = new_director_dict[full_name]
							else:
								name_idx = len(new_director_dict)
								new_director_dict[full_name] = name_idx
						else:
							job_type = "PEOPLE"
							if full_name in new_people_dict:
								name_idx = new_people_dict[full_name]
							else:
								name_idx = len(new_people_dict)
								new_people_dict[full_name] = name_idx

						
						placeholder = before_punct + "[MOVIE_P_{}_{}]".format(job_type, name_idx) + after_punct
						the_text.append(placeholder)
						

						people_indexer += 1
						processed = True

		if processed:
			index = next_idx
		else:
			if "Schwarzenegger" in token:
				result = search_person("Schwarzenegger")
				name = result["results"][0]["name"]
				people_names.append(name)
				info = result["results"][0]
				person_id = info["id"]
				info_list.append(person_id)
				full_name = name
				if full_name in new_name_dict:
					name_idx = new_name_dict[full_name]
				else:
					name_idx = len(new_name_dict)
					new_name_dict[full_name] = name_idx
				token = token.replace("Schwarzenegger", "[MOVIE_P_ACTOR_{}]".format(name_idx))
			if "Awkwafina" in token:
				
				
				result = search_person("Awkwafina")
				name = result["results"][0]["name"]
				people_names.append(name)
				info = result["results"][0]
				person_id = info["id"]
				info_list.append(person_id)
				full_name = name
				if full_name in new_name_dict:
					name_idx = new_name_dict[full_name]
				else:
					name_idx = len(new_name_dict)
					new_name_dict[full_name] = name_idx
				token = token.replace("Awkwafina", "[MOVIE_P_ACTOR_{}]".format(name_idx))

			the_text.append(token)
			index += 1

	the_text = " ".join(the_text)

	return the_text+ending, people_names, new_name_dict, new_people_dict, new_director_dict

SENTIMENT_URL = 'SENTIMENT_DETECTION_SERVICE_URL'
headers = {'content-type': 'application/json'}
def get_sentiment(text):
	sentence = str(text).lower()
	data = [{"text": sentence}]
	data = json.dumps(data)
	TIMEOUT = 50
	result = requests.post(url=SENTIMENT_URL, data=data, headers=headers, timeout=TIMEOUT).json()
	return result[0]

def load_movie_to_dict():
	filename = "filePath.tsv"
	movie_map_dict = {}
	data = pd.read_csv(filename, sep="\t")
	title_to_id = {}
	for idx, row in data.iterrows():
		movie_id = row['movie_id']
		title = row['title']
		year = row['year']
		movie_map_dict[movie_id] = {"title": title, "year": year}
		title_to_id[title+ " ({})".format(year)] = movie_id
	return movie_map_dict, title_to_id

def have_youtube_trailer(full_movie_dataset, title_to_id):
	data = pd.read_csv(full_movie_dataset, sep="\t")
	valid_id = {}
	for idx, row in data.iterrows():
		title = row['title']
		year = row['year']
		key = title+ " ({})".format(year)
		if key in title_to_id:
			val = title_to_id[key]
			valid_id[val] = row

	return valid_id

valid_id = have_youtube_trailer("filePath.tsv", title_to_id)

def load_movie_db():
    the_dict = {}
    popular_action_movies = "filePath.tsv"
    action_movie = pd.read_csv(popular_action_movies, sep="\t")
    
    popular_comedy_movies = "filePath.tsv"
    comedy_movie = pd.read_csv(popular_comedy_movies, sep="\t")
    
    popular_drama_movies = "filePath.tsv"
    drama_movie = pd.read_csv(popular_drama_movies, sep="\t")
    
    popular_scifi_movies = "filePath.tsv"
    scifi_movie = pd.read_csv(popular_scifi_movies, sep="\t")
    
    popular_horror_movies = "filePath.tsv"
    horror_movie = pd.read_csv(popular_horror_movies, sep="\t")
    
    popular_documentary_movies = "filePath.tsv"
    doc_movie = pd.read_csv(popular_documentary_movies, sep="\t")
    
    the_dict['action'] = action_movie
    the_dict['comedy'] = comedy_movie
    the_dict['drama'] = drama_movie
    the_dict['scifi'] = scifi_movie
    the_dict['horror'] = horror_movie
    the_dict['documentary'] = doc_movie
    
    return the_dict

popular_movies_by_genre = load_movie_db()
print("Finish loading movies...")

def load_movies_by_genre(genre):
    if genre == "western":
        path = "filePath.tsv"
    elif genre == "war":
        path = "filePath.tsv"
    elif genre == "thriller":
        path = "filePath.tsv"
    elif genre == "sport":
        path = "filePath.tsv"
    elif genre == "romance":
        path = "filePath.tsv"
    elif genre == "mystery":
        path = "filePath.tsv"
    elif genre == "animation":
        path = "filePath.tsv"
    elif genre == "family":
        path = "filePath.tsv"
    elif genre == "fantasy":
        path = "filePath.tsv"
    elif genre == "biography":
        path = "filePath.tsv"
    elif genre == "musical":
        path = "filePath.tsv"
    else:
        path = "filePath.tsv"
        
    movies = pd.read_csv(path, sep="\t")
    return movies

def load_vector(input_file):
	list_id_to_movie_id = {}
	movie_id_to_list_id = {}
	movie_list = []
	with open(input_file, 'r') as read_file:
		counter = 0
		for each_line in read_file:
			the_list = each_line.strip().split("\t")
			movie_id = the_list[0].replace("movieID_", "")
			movie_id = int(movie_id)
			list_id_to_movie_id[counter] = movie_id
			movie_id_to_list_id[movie_id] = counter
			movie_list.append(the_list[1:])
			counter += 1

	np_list = np.array(movie_list)

	return np_list, list_id_to_movie_id, movie_id_to_list_id

def search(movie_id, matrix, N):
	movie_score = matrix[movie_id:movie_id+1]

	sim_matrix_score = cosine_similarity(movie_score, matrix)

	id_sim_score_dict = {}
	for idx, element in enumerate(sim_matrix_score.flatten()):
		id_sim_score_dict[idx] = element
	sorted_key = sorted(id_sim_score_dict, key=id_sim_score_dict.get, reverse=True)
	top_N_indexes = sorted_key[:N]
	if debug_print:
		print(top_N_indexes)
	return top_N_indexes

def filter_movies(current_movie_id, other_movie_id, movie_map):
    try:
        current_mv_info = movie_map[current_movie_id]
        other_mv_info = movie_map[other_movie_id]

        current_mv_genre = set(current_mv_info['genre'].lower().split(", "))
        other_genre = set(other_mv_info['genre'].lower().split(", "))
        intersection_genre = current_mv_genre.intersection(other_genre)
    except:
        return False
    if len(intersection_genre) > 0:
        return True
    else:
        return False
    

def get_title(movie_id, top_N_indexes, movie_map, counter_movie_id):
    from_list_to_movie_id = counter_movie_id[movie_id]
    from_movie_id_to_title = movie_map[from_list_to_movie_id]
    the_result = []
    for movie_id_in_list in top_N_indexes:
        result = counter_movie_id[movie_id_in_list]
        if True:
            if result in valid_id and result != counter_movie_id[movie_id]:
                filtered = filter_movies(counter_movie_id[movie_id], result, valid_id)
                if filtered:
                    title = movie_map[result]["title"]
                    current_year = int(movie_map[result]["year"])
                    
                    if the_result == []:
                        the_result.append(title + " ({})".format(current_year))
                    else:
                        temp_result = []
                        is_inserted = False
                        for element in the_result:
                            content = element.split(" ")
                            
                            element_title = " ".join(content[:len(content)-1])
                            element_year = int(content[len(content)-1].replace("(","").replace(")", ""))
                            if element_year >= current_year:
                                if element not in temp_result:
                                    temp_result.append(element)
                            else:
                                if not is_inserted:
                                    temp_result.append(title + " ({})".format(current_year))
                                    is_inserted = True
                                temp_result.append(element)
                        if not is_inserted:
                            temp_result.append(title + " ({})".format(current_year))
                        the_result = temp_result

    
    return the_result

def load_actor():
    with open('ACTOR_DATA_Path.tsv', 'rb') as file:
        actor_data = pickle.load(file)
    return actor_data


def get_actor_movies(actor_data, positive_actor, user_favorites):
    list_of_movies = actor_data[positive_actor]
    try:
        fav_genre_list = user_favorites['genre']
    except:
        fav_genre_list = []
    if len(list_of_movies) >= 1:
        recommended_movie = None
        founded = False
        for movie in list_of_movies:
            genres = movie['genre']
            genre_list = genres.split(", ")
            if len(genre_list) >= 1:
                for genre in genre_list:
                    if genre in fav_genre_list:
                        recommended_movie = movie
                        founded = True
                        break
            if founded:
                break
        if not founded:
            recommended_movie = list_of_movies[0]
    
        movie_title = recommended_movie['title']
        movie_year = recommended_movie['year']
        return [movie_title + " ({})".format(movie_year)]
    else:
        return []

def recommend_from_tmdb(movie_title):
    tokenized_title = movie_title.split(" ")
    if len(tokenized_title)>1:
        title = "+".join(tokenized_title[:len(tokenized_title)-1])
    else:
        title = movie_title
    recommendations = []
    
    the_url = "url".format(TMDB_KEY, title)
    movie_json = urlopen(the_url).read().decode('utf8')
    movie_info = json.loads(movie_json)
    results = movie_info['results']
    
    tmdb_movie_id = ""
    if len(results) > 0:
        tmdb_movie_id = results[0]['id']
    
    if tmdb_movie_id != "":
        rec_url = "url".format(tmdb_movie_id, TMDB_KEY)
        rec_result_json = urlopen(rec_url).read().decode('utf8')
        rec_info = json.loads(rec_result_json)
        results = rec_info['results']
        for item in results:
            title = item['title']
            year = item['release_date'].split("-")[0]
            title_and_year = title + " ({})".format(year)
            if title_and_year in title_to_id:
                movie_id = title_to_id[title_and_year]
                if movie_id in valid_id:
                    recommendations.append(title_and_year)
    
    return recommendations


def give_recommendation(user_favorite_thing, positive_genre, positive_actor, fav_attribute, already_recommended, n=2):
    fav_movie = user_favorite_thing['movies']
    fav_genre = user_favorite_thing['genres']
    fav_actor = user_favorite_thing['actors']
    
    recommendation = []
    if (fav_movie == [] and positive_actor=="") or fav_attribute == "genre":
        if debug_print:
            print("Give recommendation based on GENRE:" + str(positive_genre))
        target_genre = positive_genre

        if target_genre not in popular_movies_by_genre:
            movies = load_movies_by_genre(target_genre)
            popular_movies_by_genre[target_genre] = movies
            
        list_of_movies = popular_movies_by_genre[target_genre]
        
        taken = False
        n_taken = 0
        for idx in range(100):
            movie_desc = list_of_movies.iloc[idx]
            title = movie_desc['title']
            year = movie_desc['year']
            key = title + " ({})".format(year)
            if key not in already_recommended:
                genres = movie_desc['genre'].lower().split(", ")
                
                already_recommended[key] = movie_desc
                recommendation.append(key)
                taken = True
                n_taken += 1
            
            if n_taken == n:
                break
    else:
        if positive_actor != "":
            if debug_print:
                print("Give recommendation based on ACTOR:" + str(positive_actor))
            recommendation = get_actor_movies(actor_data, positive_actor, user_favorite_thing)
        if fav_attribute != "actor" or recommendation == []:
            if debug_print:
                print("Give recommendation based on MOVIE")
            last_mentioned = fav_movie[len(fav_movie)-1]
            if debug_print:
                print("last mentioned MOVIE in give recommendation: ", last_mentioned)
            recommended_titles = recommend_from_tmdb(last_mentioned)
            recommendation = recommended_titles
            if recommendation == []:
                #remove year
                remove_year = last_mentioned.split(" ")
                title_only = " ".join(remove_year[:len(remove_year)-1])
                recommendedation = recommend_from_tmdb(title_only)
            if recommendation == [] and positive_genre != "":
                target_genre = positive_genre

                if target_genre not in popular_movies_by_genre:
                    movies = load_movies_by_genre(target_genre)
                    popular_movies_by_genre[target_genre] = movies

                list_of_movies = popular_movies_by_genre[target_genre]

                taken = False
                n_taken = 0
                for idx in range(100):
                    movie_desc = list_of_movies.iloc[idx]
                    title = movie_desc['title']
                    year = movie_desc['year']
                    key = title + " ({})".format(year)
                    if key not in already_recommended:
                        genres = movie_desc['genre'].lower().split(", ")

                        already_recommended[key] = movie_desc
                        recommendation.append(key)
                        taken = True
                        n_taken += 1

                    if n_taken == n:
                        break       
            
    return recommendation, already_recommended

def get_movie_plot(text, movie_name):
    if "PLOT]" in text:
        tokenized_title = movie_name.split(" ")
        if len(tokenized_title)>1:
            title = "+".join(tokenized_title[:len(tokenized_title)-1])
        else:
            title = movie_name
        try:
            the_url = "url".format(TMDB_KEY, title)
            movie_json = urlopen(the_url).read().decode('utf8')
            movie_info = json.loads(movie_json)

            results = movie_info['results']
            if len(results) > 0:
                content = results[0]['overview']
            if "[MOVIE_PLOT]" in text:
                result = text.replace("[MOVIE_PLOT]", '[SPLIT]"{}"[SPLIT]'.format(content))
                content = result.split("[SPLIT]")
                temp_list = []
                for element in content:
                    temp_list.append(element.strip())
                result = " ".join(temp_list)
                result = result.strip()
            elif "PLOT]" in text:
                result = text.replace("PLOT]", '[SPLIT]"{}"[SPLIT]'.format(content))
                content = result.split("[SPLIT]")
                temp_list = []
                for element in content:
                    temp_list.append(element.strip())
                result = " ".join(temp_list)
                result = result.strip()
            else:
                result = text
            return result
        except:
            return "The movie has interesting plot"
    elif "It is about" in text:
        new_text = text.split("It is about")
        
        try:
            if title_to_id[movie_name] in valid_id:
                content = valid_id[title_to_id[movie_name]]
                summary = content['short_plot']
                text = new_text[0] + ' "' + summary + '"'
                
            else:
                tokenized_title = movie_name.split(" ")
                if len(tokenized_title)>1:
                    
                    title = "+".join(tokenized_title[:len(tokenized_title)-1])
                else:
                    title = movie_name
                the_url = "url".format(TMDB_KEY, title)
                movie_json = urlopen(the_url).read().decode('utf8')
                movie_info = json.loads(movie_json)

                results = movie_info['results']
                if len(results) > 0:
                    content = results[0]['overview']
                    text = new_text[0] + 'It is about "' + content + '"'
                else:
                    return "It has an interesting story"
        except:
            return "The movie has interesting plot"
        return text.strip()
    else:
        return text.strip()


def replace_actors(text, movie_name, mentioned_actors, last_mentioned, idx_turn):
    actor_list = []
    if "P_ACTOR" in text:
        text = text.strip()
        if movie_name in title_to_id and title_to_id[movie_name] in valid_id:
            content = valid_id[title_to_id[movie_name]]
            actor_list = content['actors'].split(", ")
            if debug_print:
                print(actor_list)
        else:
            tokenized_title = movie_name.split(" ")
            if len(tokenized_title)>1:
                title = "+".join(tokenized_title[:len(tokenized_title)-1])
            else:
                title = movie_name
                the_url = "url".format(TMDB_KEY, title)
                movie_json = urlopen(the_url).read().decode('utf8')
                movie_info = json.loads(movie_json)
                results = movie_info['results']

                tmdb_movie_id = ""
                if len(results) > 0:
                    tmdb_movie_id = results[0]['id']
                    cast_url = "url".format(tmdb_movie_id, TMDB_KEY)
                    cast_json = urlopen(cast_url).read().decode('utf8')
                    cast_info = json.loads(cast_json)
                    cast = cast_info["cast"]
                    actor_list = []
                    for character in cast:
                        the_name = character["name"]
                        actor_list.append(the_name)
                        if len(actor_list) >= 4:
                            break
                else:
                    actor_list = []
        
        if actor_list == []:
            return "I think the movie has great actors in it!", mentioned_actors

            
        taken_actor = []
        result = text.replace("[MOVIE_PLOT]", '"{}"'.format(content['short_plot']))
        tokenized_sent = text.split(" ")
        
        actor_token_counter = 0
        normal_token_counter = 0
        
        placeholder_id_to_text = {y:x for x,y in mentioned_actors.items()}
        print(placeholder_id_to_text )
        coref_text = []
        noncoref_text = []
        i = 0
        j = 0
        new_mentioned1 = mentioned_actors
        new_mentioned2 = mentioned_actors
        tokenized_by_comma = text.split(",")
        prev_word = ""
        if len(last_mentioned) > 1:
            diff_turn = idx_turn - last_mentioned[1]
        else:
            diff_turn = 0
        threshold_name = 3
        for token in tokenized_sent:
            has_actor = False
            if "[MOVIE_P_ACTOR_" in token:
                get_idx = token.split("[MOVIE_P_ACTOR_")
                has_actor = True
            elif "P_ACTOR_" in token:
                get_idx = token.split("P_ACTOR_")
                has_actor = True
            else:
                has_actor = False
            
            if has_actor:
                if get_idx[0] != "":
                    prev_word = get_idx[0]
                    coref_text.append(prev_word)
                    normal_token_counter += 1
                    
                temp = get_idx[len(get_idx)-1].split("]")

                idx = int(temp[0])
                if last_mentioned != []:
                    if diff_turn < threshold_name:
                        if debug_print:
                            print("less than 2")
                        the_list = last_mentioned[0]
                        actor_name = the_list[len(the_list)-1]
                        temp_str = "]".join(temp[1:])
                        if temp_str not in set(string.punctuation):
                            temp_str = " "+temp_str
                        coref_text.append(actor_name+temp_str)
                    else:
                        if debug_print:
                            print("not less")
                        if i < len(actor_list):

                            actor_name = actor_list[i]
                            i += 1
                            temp_str = "]".join(temp[1:])
                            if temp_str not in set(string.punctuation):
                                temp_str = " "+temp_str
                            coref_text.append(actor_name + temp_str)
                            if actor_name not in new_mentioned1:
                                new_mentioned1[actor_name] = len(new_mentioned1)
                                placeholder_id_to_text = {y:x for x,y in new_mentioned1.items()}
                        else:
                            coref_text.append("this actor")
                    
                elif idx in placeholder_id_to_text:
                    if debug_print:
                        print("it is in placeholder")
                    if diff_turn > threshold_name:
                        if i < len(actor_list):

                            actor_name = actor_list[i]
                            i += 1
                            temp_str = "]".join(temp[1:])
                            if temp_str not in set(string.punctuation):
                                temp_str = " "+temp_str
                            coref_text.append(actor_name + temp_str)
                            if actor_name not in new_mentioned1:
                                new_mentioned1[actor_name] = len(new_mentioned1)
                                placeholder_id_to_text = {y:x for x,y in new_mentioned1.items()}
                    else:             
                        actor_name = placeholder_id_to_text[idx]
                        if debug_print:
                            print("here: " + actor_name)
                        temp_str = "]".join(temp[1:])
                        if temp_str not in set(string.punctuation):
                            temp_str = " "+temp_str
                        coref_text.append(actor_name+temp_str)
                else:
                    if i < len(actor_list):

                        actor_name = actor_list[i]
                        i += 1
                        temp_str = "]".join(temp[1:])
                        if temp_str not in set(string.punctuation):
                            temp_str = " "+temp_str
                        coref_text.append(actor_name + temp_str)
                        if actor_name not in new_mentioned1:
                            new_mentioned1[actor_name] = len(new_mentioned1)
                            placeholder_id_to_text = {y:x for x,y in new_mentioned1.items()}
                
                actor_token_counter += 1
                
            else:
                coref_text.append(token)
                noncoref_text.append(token)
                normal_token_counter += 1
        
        actor2_token = 0
        actor_noncoref = []
        for token in tokenized_by_comma:
            if "[MOVIE_P_ACTOR_" in token:
                get_idx = token.split("[MOVIE_P_ACTOR_")
               
                temp = get_idx[len(get_idx)-1].split("]")
                if j >= len(actor_list):
                    break
                another_actor_name = actor_list[j]
                actor_noncoref.append(another_actor_name+"]".join(temp[1:]))
                j += 1
        
                if another_actor_name not in new_mentioned2:
                    new_mentioned2[another_actor_name] = len(new_mentioned2)
                    
                actor2_token += 1
                
            else:
                normal_token_counter += 1
        
        if (normal_token_counter - actor_token_counter) >= 2 or (normal_token_counter - actor2_token) > 2 or actor_token_counter == 1:
            if debug_print:
                print("normal")
            result = " ".join(coref_text)
            return result, new_mentioned1
        else:
            if debug_print:
                print("nonnormal: actor2 token " + str(actor2_token) + " other: " + str(normal_token_counter))
                print("noncoref: " + str(noncoref_text))
            result = ", ".join(actor_noncoref)
            return result, new_mentioned2
    else:
        return text, mentioned_actors

def RemoveStrategyAndSEP(sentence):
    sent = re.sub(r'\<[[a-z]*[_]*[[a-z]*\>', " ", sentence).strip()
    segment = sent.split("[SEP]")
    return segment[0].strip()

def removeSEP(sentence):
    segment = sentence.split("[SEP]")
    return segment[0].strip()

def CheckMoviePlot(tokenizer, response):
	contain_movie_plot = False
	index_movie_plot = tokenizer.encode("[MOVIE_PLOT]")[0]
	if index_movie_plot in response:
		contain_movie_plot = True
	return contain_movie_plot

def similarity(candidate_rec_utt, last_rec_response):
    intersection = set(candidate_rec_utt).intersection(set(last_rec_response))
    overlap = len(intersection) / len(set(last_rec_response))
    return overlap


os.environ["CUDA_VISIBLE_DEVICES"] = "CUDA_INDEX"
device = torch.device("cuda")
debug_print = False

ACTOR_TEMPLATES = ["The movie has talented actors!", "The actors are really great.", "I like the actors!"]
MOVIE_PLOT_TEMPLATES = ["The movie story is very interesting!", "It has interesting story.", "The plot is very interesting!"]

label_to_strategy = {0: 'no_strategy',
 1: 'opinion_inquiry',
 2: 'self_modeling',
 3: 'personal_opinion',
 4: 'credibility',
 5: 'encouragement',
 6: 'similarity',
 7: 'rephrase_preference',
 8: 'preference_confirmation',
 9: 'experience_inquiry',
 10: 'acknowledgment',
 11: 'personal_experience',
 12: 'offer_help'}

label_to_recommendation = {0: 'not_recommendation', 1: 'recommendation'}

genre_from_tmdb = {28: "action", 12: "adventure", 16: "animation", 35: "comedy", 80: "crime", 99:"documentary", 18 : "drama"}
genre_from_tmdb[10751] = "family"
genre_from_tmdb[14] = "fantasy"
genre_from_tmdb[36] = "history"
genre_from_tmdb[27] = "horror"
genre_from_tmdb[10402] ="music"
genre_from_tmdb[9648] = "mystery"
genre_from_tmdb[10749] = "romance"
genre_from_tmdb[878] = "sci-fi"
genre_from_tmdb[10770] = "tv movie"
genre_from_tmdb[53] = "thriller"
genre_from_tmdb[10752] = "war"
genre_from_tmdb[37] = "western"

tokenizer = torch.load("TOKENIZER_PATH")
model_A_states, model_B_states = torch.load("MODEL_PATH")

config = GPT2Config()
config.vocab_size = model_A_states["transformer.wte.weight"].shape[0]
model_A = GPT2LMHeadModel(config)
model_B = GPT2LMHeadModel(config)

model_A.load_state_dict(model_A_states)
model_B.load_state_dict(model_B_states)
model_A.to(device)
model_B.to(device)

model_A_states["transformer.wte.weight"].shape

strategy_detector = torch.load("STRATEGY_CLS_MODEL_Path")
recommendation_detector = torch.load("RECOMMENDATION_CLS_MODEL_PATH")

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", num_labels = 13)

strategy_detector.eval()
recommendation_detector.eval()
model_A.eval()
model_B.eval()

prev_input_for_recommender = tokenizer.encode("A:")
prev_input_for_recommender = torch.LongTensor(prev_input_for_recommender).unsqueeze(0).to(device)

candidates_num = 3
ngram = 2
temperature = 0.8 

top_k = 400
top_p = 0.9

past_for_recommender = None

# sep = tokenizer.encode("\n\n\n")
sep = [628, 198]

mentioned_genres = {}
movie_mentioned = {}
mentioned_actors = {}
mentioned_people = {}
mentioned_directors = {}
favorite = {"genres": [], "movies": [], "actors": []}
already_recommended = {}
disliked = {"genres": [], "movies": [], "actors": []}
positive_last_mentioned_genre = ""
positive_actor = ""
last_mentioned_attribute = ""
last_mentioned_movie = ""
last_mentioned_genre = ""
last_mentioned_actors = []
kind_of_movie_question = False
turn_counter = 0

last_rec_template = ""
last_rec_response = [-1]
while True:
    "Sampling based method"
    
    with torch.no_grad():
        candidates_pool = []
        candidates_past_pool = []
        candidates_strategy = []
        candidates_recommendation = []
        
        # Sampling several candidates   
        # (input: past memory from user model and input token(s); 
        # output: one selected past memory for user model and one selected recommender response)
        for count in range(candidates_num):
            past = past_for_recommender
            prev_input = prev_input_for_recommender
            sent = []
            
            # Sampling one candidate
            for i in range(200):
                logits, past = model_A(prev_input, past=past)
                logits = logits[:, -1, :] / temperature
                logits = top_filtering(logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(logits, -1)
                prev_input = torch.multinomial(probs, num_samples=1)
                prev_word = prev_input.item()

                if prev_word == 628:
                    break
                else:
                    sent.append(prev_word)
            # save candidate and corresponding past memory
            candidates_pool.append(sent)
            candidates_past_pool.append(past)

    # Use pretrained strategy classifier to detect the strategy of generated responses
    if past_for_recommender == None:
        start = True
    else:
        start = False
    for cand in candidates_pool:
        if start:
            hist_response = tokenizer.decode(cand)
            hist_response_tmp = "[CLS]" + "A:" + hist_response+ "[SEP]"
            hist_response = RemoveStrategyAndSEP(hist_response_tmp)
             
        else:
            cand_tmp = tokenizer.decode(cand)
            cand_clean = RemoveStrategyAndSEP(cand_tmp)
            cand_clean = "A:"+cand_clean
            previous_user_tmp = RemoveStrategyAndSEP(previous_user)
            hist_response = "[CLS]" + previous_user_tmp + "[SEP]" + cand_clean + "[SEP]"
        hist_response = bert_tokenizer.encode(hist_response)
        # Attention: there is no padding
        b_input_mask = [int(i>=0) for i in hist_response]

        with torch.no_grad():
            strategy_logits = strategy_detector(torch.tensor(hist_response).unsqueeze(0).to(device), token_type_ids=None, attention_mask=torch.tensor(b_input_mask).unsqueeze(0).to(device))
            recommendation_logits = recommendation_detector(torch.tensor(hist_response).unsqueeze(0).to(device), token_type_ids=None, attention_mask=torch.tensor(b_input_mask).unsqueeze(0).to(device))
            value, index = strategy_logits.max(-1)
            _, recommendation_index = recommendation_logits.max(-1)

            candidates_strategy.append(label_to_strategy[index.item()])
            candidates_recommendation.append(label_to_recommendation[recommendation_index.item()])
   
    # remove candidate if there is ngram overlapping
    for index, sample in enumerate(candidates_pool):
        block = nGramBlock(sample, ngram)
        if block:
            if debug_print:
                print("There is a duplicated ", ngram, " gram.")
            candidates_pool.pop(index)
            candidates_past_pool.pop(index)
            candidates_strategy.pop(index)
            candidates_recommendation.pop(index)
        
    # Select one candidate from the candidates and reset the intermediate data container
    selected_idx = -1
    if selected_idx not in range(len(candidates_pool)):      
        selected_idx = random.randint(0, len(candidates_pool) - 1)

    #-------------- sentence length rules-----------------
    '''
    max_len = 0
    selected_idx_new = -1
    for temp_idx, candidate_rec_utt in enumerate(candidates_pool):
        tokenized_cand_rec_len = len(candidate_rec_utt)
        decoded_candidate = tokenizer.decode(candidate_rec_utt)
        tokenized_generation = removeSEP(decoded_candidate).lower().split(" ")
        
        if debug_print:
            print("Candidate recommender's utterance: " + decoded_candidate)
            print(candidate_rec_utt)
        if "what" in tokenized_generation and ("kind" in tokenized_generation or "type" in tokenized_generation or "kinds" in tokenized_generation or "types" in tokenized_generation) and ("movies" in tokenized_generation) and "like" in tokenized_generation:
            if debug_print:
                print("Has been asked")
            continue
        candidate_rec_utt_noSEP = tokenizer.encode(removeSEP(decoded_candidate))
        if tokenized_cand_rec_len > max_len and similarity(candidate_rec_utt_noSEP, last_rec_response) < 0.5:
            max_len = tokenized_cand_rec_len
            selected_idx_new = temp_idx
    if not start and turn_counter <= 7:
        selected_idx = selected_idx_new
    if debug_print:
        print("----")
    '''
    #-------------- sentence length rules-----------------

    # setup past memory for recommender model and seeker model     
    past = candidates_past_pool[selected_idx]
    sent = candidates_pool[selected_idx]
    recommendation_label = candidates_recommendation[selected_idx]
    turn_counter += 1
    if debug_print:
        print("Recommendation label: " + str(recommendation_label))
    assert(len(candidates_past_pool) == len(candidates_pool) == len(candidates_strategy) == len(candidates_recommendation))

    generated_sent = tokenizer.decode(sent)
    generated_sent = remove_duplicate_movie_plots(generated_sent)

    #check type of question asked
    tokenized_generation = generated_sent.lower().split(" ")
    if "what" in tokenized_generation and ("kind" in tokenized_generation or "type" in tokenized_generation or "kinds" in tokenized_generation or "types" in tokenized_generation) and ("movies" in tokenized_generation) and "like" in tokenized_generation:
        kind_of_movie_question=True
    
    proposed_genres = []

    if debug_print:
        print("Already recommended: " + str(already_recommended.keys()))
    template_genre = "action"
    if last_mentioned_movie != "":
        if last_mentioned_movie in title_to_id and title_to_id[last_mentioned_movie] in valid_id:
            template_genre = valid_id[title_to_id[last_mentioned_movie]]["genre"].split(", ")[0].lower()
        else:
            tokenized_title = last_mentioned_movie.split(" ")
            if len(tokenized_title)>1:
                title = "+".join(tokenized_title[:len(tokenized_title)-1])
            else:
                title = last_mentioned_movie
            the_url = "https://api.themoviedb.org/3/search/movie?api_key={}&query={}".format(TMDB_KEY, title)
            movie_json = urlopen(the_url).read().decode('utf8')
            movie_info = json.loads(movie_json)
            results = movie_info['results']

            if len(results) > 0:
                tmdb_genre = results[0]['genre_ids'][0]
                template_genre = genre_from_tmdb[tmdb_genre]

    if debug_print:
        print("template genre: " + str(template_genre))
    generated_sent, mentioned_genres, last_mentioned_genre = convert_back(generated_sent, mentioned_genres, proposed_genres, case="GENRE", template_movie=None, template_genre=template_genre)
    
    if debug_print:
        print("mentioned genres: " + str(mentioned_genres))
    if "TITLE_" in generated_sent:
        if positive_last_mentioned_genre == "":
            if "[SEP]" in generated_sent and "genre: " in generated_sent:
                temp = generated_sent.split("genre: ")
                temp2 = temp[len(temp)-1].split(";")
                positive_last_mentioned_genre = "family"
                if temp2[0].strip() != "":
                    positive_last_mentioned_genre = temp2[0].strip()
                else:
                    if mentioned_genres != {}:
                        positive_last_mentioned_genre = last_mentioned_genre #mentioned_genres[len(mentioned_genres)-1]
                    else:
                        positive_last_mentioned_genre = "comedy"
            else:
                if mentioned_genres != {}:
                    positive_last_mentioned_genre = last_mentioned_genre #mentioned_genres[len(mentioned_genres)-1]
                else:
                    positive_last_mentioned_genre = "comedy"
        if debug_print:
            print("Positive last mentioned genre: " + str(positive_last_mentioned_genre))

        recommendations, already_recommended = give_recommendation(favorite, positive_last_mentioned_genre, positive_actor, last_mentioned_attribute, already_recommended)
        if debug_print:
            print("recommendations: " + str(recommendations))
        if recommendation_label == "not_recommendation":
            generated_sent, movie_mentioned, last_mentioned_movie = convert_back(generated_sent, movie_mentioned, recommendations, case="TITLE",template_movie="Joker (2019)")
        else:
            if debug_print:
                print("here is force rec: " + str(recommendation_label))
            template_rec = "Joker (2019)"
            
            if recommendations != []:
                
                for the_recommended_movie in recommendations:
                    if the_recommended_movie not in already_recommended:
                        template_rec = the_recommended_movie
                        break
            generated_sent, movie_mentioned, last_mentioned_movie = force_rec(generated_sent, movie_mentioned, recommendations, template_movie=template_rec)
            if last_mentioned_movie not in already_recommended:
                already_recommended[last_mentioned_movie] = True
    if debug_print:
        print("Last mentioned movie debugging: " + last_mentioned_movie)
    generated_sent = get_movie_plot(generated_sent, last_mentioned_movie)

    generated_sent, mentioned_actors = replace_actors(generated_sent, last_mentioned_movie, mentioned_actors, last_mentioned_actors, turn_counter)

    if debug_print:
        print("Last mentioned movie attribute: " + last_mentioned_attribute)                                         
    print("RECOMMENDER: " + generated_sent)
    
    last_rec_response = tokenizer.encode(generated_sent)
    
    # finish tail
    prev_input = torch.LongTensor(sep).unsqueeze(0).to(device)
    _, past = model_A(prev_input, past=past)
    
    # input and update B's utterance
    user = input("SEEKER: ")
    if user == "quit":
        break
        
    sentiment_label = get_sentiment(user.lower())
    if debug_print:
        print("Sentiment label: " + sentiment_label)
    text_with_placeholder, movie_in_text, movie_mentioned = create_movie_slot(generated_sent, user, movie_mentioned)
    if debug_print:
        print("movie_mentioned: " + str(movie_mentioned))
    user_with_genre, mentioned_genre_dict, genre_list = label_genre(text_with_placeholder, mentioned_genres)
    user_utt = user_with_genre
    
    if movie_in_text != "":
        user_utt = add_SEP(user_with_genre, movie_in_text.split("; "), case="movie")
        last_mentioned_attribute = "movie"
        if sentiment_label != "negative":
            favorite["movies"] += movie_in_text.split("; ")
            
            last_mentioned_movie = favorite["movies"][len(favorite["movies"])-1]
            if debug_print:
                print(favorite)
            
    if genre_list != []:
        user_utt = add_SEP(user_utt, genre_list)
        if sentiment_label != "negative":
            favorite["genres"] += genre_list
            if debug_print:
                print(favorite)
            positive_last_mentioned_genre = genre_list[len(genre_list)-1]
            last_mentioned_attribute = "genre"
    

    user_utt, people_names, mentioned_actors, mentioned_people, mentioned_directors  = find_name(user_utt, name_list, mentioned_actors, mentioned_directors, {})
    if debug_print:
        print("user utt: " + str(user_utt))
    if people_names != []:
        user_utt = add_SEP(user_utt, people_names, case="people_name")
        last_mentioned_attribute = "actor"
        last_mentioned_actors = [people_names, turn_counter]
        if sentiment_label != "negative":
            positive_actor = people_names[len(people_names)-1]
    
    if debug_print:
        print(last_mentioned_attribute)
        print("processed: " + user_utt + " last REC: " + generated_sent)
        
    previous_user = "B:" + user_utt
    # print("The input of user model: ", previous_user)

    user = tokenizer.encode("B:" + user_utt)
    prev_input = user + sep
    prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(device)
    
    # seeker    
    _, past = model_B(prev_input, past=past)
    
    # start A's utterance
    suffix = tokenizer.encode("A: ")
    prev_input = torch.LongTensor(suffix).unsqueeze(0).to(device)
    
    # recode the prev_input and past for recommender
    past_for_recommender = past
    prev_input_for_recommender = prev_input



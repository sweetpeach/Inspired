import csv
import collections
import torch
import numpy as np

Inspired_path = "TSV_DATA_PATH"


with open(Inspired_path, "r") as file:
    csv_reader = csv.reader(file, delimiter=',')
    next(csv_reader)
    
    last_role = ""
    last_convid = ""
    dialog = []
    dialog_list = collections.OrderedDict()
        
    seeker_intend_info = collections.OrderedDict({"movie":[], "genre":[], "people_name":[]})
    recommender_intend_info = collections.OrderedDict({"movie":[], "genre":[], "people_name":[]})

    for row in csv_reader:
        conv_idx = row[0]
        role = row[2]
        utterance = row[5]
        strategy = row[14]
        
        print(strategy)
        if strategy == "transparency":
            strategy = "<offer_help> "
        else:
            strategy = "<" + strategy +">"
        if strategy =="<>":
            strategy = ""
        
#         seeker intention info
        if role == "SEEKER":
            if row[6] != "":
                seeker_intend_info["movie"].append(row[6].replace(";", ",").replace("  ", " "))
            if row[7] != "":
                seeker_intend_info["genre"].append(row[7].replace(";", ",").replace("  ", " "))
            if row[8] != "":
                seeker_intend_info["people_name"].append(row[8].replace(";", ",").replace("  ", " "))
        
        elif last_role == "SEEKER":
            if len(seeker_intend_info["movie"]) != 0:
                movie_info = "movie: "+", ".join(seeker_intend_info["movie"])+";"
            else: 
                movie_info = ""
                
            if len(seeker_intend_info["genre"]) != 0:
                genre_info = "genre: "+ ", ".join(seeker_intend_info["genre"]) +";"
            else:
                genre_info = ""
            
            if len(seeker_intend_info["people_name"]) != 0:
                people_info = "people_name: " + ", ".join(seeker_intend_info["people_name"])+";"
            else:
                people_info = ""
                
            # previous seeker may be in the same conversation or in the last conversation
            if conv_idx == last_convid:
                index = conv_idx
            else:
                index = last_convid
            if (len(seeker_intend_info["movie"]) != 0) or (len(seeker_intend_info["genre"]) != 0) or (len(seeker_intend_info["people_name"]) != 0):
                dialog_list[index][-1] = dialog_list[index][-1] + " [SEP]"
                
                if movie_info !=0:   
                    dialog_list[index][-1] = dialog_list[index][-1] + movie_info
                if genre_info != 0:
                    dialog_list[index][-1] = dialog_list[index][-1] + genre_info
                if people_info != 0:
                    dialog_list[index][-1] = dialog_list[index][-1] + people_info
            seeker_intend_info = collections.OrderedDict({"movie":[], "genre":[], "people_name":[]})

#         seeker intention info
        if role == "RECOMMENDER":
            if row[6] != "":
                recommender_intend_info["movie"].append(row[6].replace(";", ",").replace("  ", " "))
            if row[7] != "":
                recommender_intend_info["genre"].append(row[7].replace(";", ",").replace("  ", " "))
            if row[8] != "":
                recommender_intend_info["people_name"].append(row[8].replace(";", ",").replace("  ", " "))
        
        elif last_role == "RECOMMENDER":
            if len(recommender_intend_info["movie"]) != 0:
                movie_info = "movie: "+", ".join(recommender_intend_info["movie"])+";"
            else: 
                movie_info = ""
                
            if len(recommender_intend_info["genre"]) != 0:
                genre_info = "genre: "+ ", ".join(recommender_intend_info["genre"]) +";"
            else:
                genre_info = ""
            
            if len(recommender_intend_info["people_name"]) != 0:
                people_info = "people_name: " + ", ".join(recommender_intend_info["people_name"])+";"
            else:
                people_info = ""
                
            # previous seeker may be in the same conversation or in the last conversation
            if conv_idx == last_convid:
                index = conv_idx
            else:
                index = last_convid
            if (len(recommender_intend_info["movie"]) != 0) or (len(recommender_intend_info["genre"]) != 0) or (len(recommender_intend_info["people_name"]) != 0):
                dialog_list[index][-1] = dialog_list[index][-1] + " [SEP]"
                
                if movie_info !=0:   
                    dialog_list[index][-1] = dialog_list[index][-1] + movie_info
                if genre_info != 0:
                    dialog_list[index][-1] = dialog_list[index][-1] + genre_info
                if people_info != 0:
                    dialog_list[index][-1] = dialog_list[index][-1] + people_info
            recommender_intend_info = collections.OrderedDict({"movie":[], "genre":[], "people_name":[]})

        
        if conv_idx in dialog_list.keys():
            if role == last_role:
                dialog_list[conv_idx][-1] = dialog_list[conv_idx][-1] + " " + strategy + utterance
            else:
                dialog_list[conv_idx].append(role +":" + strategy + utterance)
        else:
            dialog_list[conv_idx] = [role + ":" + utterance]
            
        last_role = role
        last_convid = conv_idx


dialogues = []
for key in dialog_list.keys():
    dialog = dialog_list[key]
    updated_dialog = []
    
    for line in dialog:
        if "RECOMMENDER:" in line:
            updated_dialog.append("A:" + line[12:])
        else:
            updated_dialog.append("B:" + line[7:])
    dialogues.append(updated_dialog)

torch.save(dialogues, "SAVE_PTH_DATA_PATH")

# INSPIRED dataset
train.tsv contains 801 dialogues, dev.tsv contains 100 dialogues, and test.tsv contains 100 dialogues. 

The sociable strategies are under **"first_label"** and **"second_label"** columns. 

## Column
* `dialog_id`: dialogue id
* `utt_id`: utterance id
* `speaker`: whether the speaker is Recommender or Seeker
* `turn_id`: turn id
* `text`: the original utterance
* `text_with_placeholder`: the utterance where the movie attributes (title, genre, actor's name, director's name, movie plot) have been replaced with a placeholder. Feel free to use your own entity detector to process the original text under "text" column. 
* `movies`: list of movie titles mentioned in the utterance
* `genres`: list of genres mentioned in the utterance
* `people_names`: list of people's names mentioned in the utterance
* `movie_dict`: a dictionary that contains each title in the conversation (as the key) paired with its corresponding index
* `genre_dict`: a dictionary that contains each genre in the conversation (as the key) paired with its corresponding index
* `actor_dict`: a dictionary that contains each actor's name in the conversation (as the key) paired with its corresponding index
* `director_dict`: a dictionary that contains each director's name in the conversation (as the key) paired with its corresponding index
* `others_dict`: a dictionary that contains each person's name (besides actor and director) in the conversation (as the key) paired with its corresponding index
* `first_label`: the main sociable strategy
* `second_label`: this is optional second strategy
* `case`: this flag denotes five cases whether the seeker accepted or rejected the recommendation.
	* accept_rating_good: the user accepted the recommendation and later rated the recommended movie 4-5
	* accept_rating_mod: the user accepted the recommendation and later rated the recommended movie <= 3
	* accept_uninterested: the user accepted the recommendation, didn't finish watching the trailer, and later said that they found the trailer uninteresting
	* accept_others: the user gave various reasons for not finishing the trailer
	* reject: the user rejected the recommendation
* `movie_id`: movie id recommended by the recommmender. You can watch the trailer on Youtube by replacing "XXX" in this link https://www.youtube.com/watch?v=XXX with the movie id (as long as the videos are still available on Youtube).
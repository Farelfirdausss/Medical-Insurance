This dataset is used in Multilabel sentiment analysis and emotion detection for
YouTube comments in different kinds of Bengali videos. We manually select these videos in Bangla language
based on their popularity (number of views, number of
likes or dislikes) dated from 2013 to early 2018. We limit
the number of comments for each video up to 50 to remove
redundancy and also exclude the replies of comments since
they do not possess much sentiment information. We use
Google translator to detect the language of each comment
as Bangla or English. The sentences which are not identified as any language are considered as Romanized Bangla (There might be some noises in this method).

There are two files in the folder. There are might be multiple comments with same text.
Also it may be noted that, the comments collected here contain
abusive and vulgar words, slangs and personal attack.
Therefore, we ensure that all annotators are adults.

###################################
Sentiment.csv

Id - Unique id number for the comment. 
Text - Text of the data
Label - 1 (3 class label) or 2 (5 class label)
Score - Denotes the polarity of the comment.
		In three class labelling : 1(positive), 0 (neutral), -1(negative)
		In three class labelling : 2 (highly positive), 1(positive), 0 (neutral), -1(negative), -2(highly negative)
Lan - Language of the comment. EN (English), BN (Bengali), RN (Romanized Bangla)
Domain - Category of the video. 

###################################
Emotion.csv

Id - Unique id number for the comment. 
Text - Text of the data
emotion - Corresponding emotion of the comment.
		Anger/Joy/Disgust/Fear/Surprise/Sad/None (no emotion found) 
Lan - Language of the comment. EN (English), BN (Bengali), RN (Romanized Bangla)
Domain - Category of the video. 

If you use the dataset in any research work, please cite the following paper as

N. Irtiza Tripto and M. Eunus Ali, "Detecting Multilabel Sentiment and Emotions from Bangla YouTube Comments," 2018 International Conference on Bangla Speech and Language Processing (ICBSLP), Sylhet, 2018, pp. 1-6.
doi: 10.1109/ICBSLP.2018.8554875


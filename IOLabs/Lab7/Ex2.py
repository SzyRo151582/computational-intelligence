from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import text2emotion as t2e

positive_opinion: str = ("Me and my family stayed at this resort in october for 1 week! and i tell you what it was "
                         "awesome! Clean tidy the staff was friendly the private beach was awesome as well i cannot "
                         "fault this resort at all and i will be returning that's for sure but for 10 days next time! "
                         "the sun set from my balcony was amazing as well! please ignore any negative comments you "
                         "see about this resort! as i felt the same! But i see one of my pals went here in july and. "
                         "i was due to go october! So i asked what the resort was like and she said is was amazing! "
                         "When we got there we made our own minds up! and it was amazing perfect first family "
                         "holiday for us it was! and cannot wait to return! They had enough drinks to keep pirates "
                         "going!")
negative_opinion = ("Be careful of 5 star reviews quoting Nadia, she is going around offering free bottles of wine "
                    "when people complain about the restaurant, in return for a good review. Essentially just buying "
                    "good reviews. I was very disappointed with our stay, the hotel is very old and in serious need "
                    "of refurbishment. We had 4 rooms and 3 of us had to complain about maintenance issues, "
                    "damp and bathrooms smelling of sewage. Eve try-day my room key would stop working and each time "
                    "you have to walk miles to reception and queue 10-15 minutes at check in to get the key "
                    "reprogrammed. The all inclusive drinks are not good, all served in a small plastic beaker even "
                    "in the bars on the evening, and you have to look after a plastic token to swap for a glass each "
                    "time you want a drink. Only local spirits and unbranded beer included. Coffee machines are "
                    "awful, the only places with a proper coffee machine charge extra. Ice cream bar is also charged. "
                    "The a la carte restaurants were fully booked for our entire 5 night stay and the buffet was "
                    "awful quality, very busy and very repetitive. They only use cheap cuts of meat. There are no "
                    "showers by the pool we used. The train ride which you pay for is very bad, just takes you round "
                    "a few derelict shops and restaurants, waste of money. Entertainment team are good but activities "
                    "such as bingo are charged for. The only positive from the stay is the beach, the sand is good "
                    "and amazing sunsets, but it was infested with Mosquito's, not the hotels fault on this front as "
                    "it’s a natural beach, but they were really back we were bitten all over even with mosquito spray "
                    "on. All round a terrible holiday and would recommend to avoid this hotel in all honesty.")

vader_analyzer = SentimentIntensityAnalyzer()
positive_sentences = tokenize.sent_tokenize(positive_opinion)
negative_sentences = tokenize.sent_tokenize(negative_opinion)
for sentence in positive_sentences:
    print(sentence)
    polarization_scores = vader_analyzer.polarity_scores(sentence)
    for emotion in sorted(polarization_scores):
        print(f"{emotion}: {polarization_scores[emotion]} ", end='')

    print()

for sentence in negative_sentences:
    print(sentence)
    polarization_scores = vader_analyzer.polarity_scores(sentence)
    for emotion in sorted(polarization_scores):
        print(f"{emotion}: {polarization_scores[emotion]} ", end='')

    print()


# Using package text2emotion
print(t2e.get_emotion(positive_opinion))
print(t2e.get_emotion(negative_opinion))

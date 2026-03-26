import pandas as pd
from apyori import apriori
import matplotlib.pyplot as plt

titanic_data = pd.read_csv("titanic.csv")

test_data = []
for i in range(2201):
    test_data.append([str(titanic_data.values[i, j]) for j in range(1, 5)])

final_rule = apriori(test_data, min_support=0.005, min_confidence=0.8, min_lift=1.2)
final_results = list(final_rule)


def check_confidence(relation):
    return relation[2][0][2]


final_results.sort(key=check_confidence)
for result in final_results:
    print(result)

support = []
confidence = []
for result in final_results:
    support.append(result[1])
    confidence.append(result[2][0][2])

plt.scatter(support, confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


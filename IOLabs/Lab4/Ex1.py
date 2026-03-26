import math

age = [23, 25, 28, 22, 46, 50, 48]
weight = [75, 67, 120, 65, 70, 68, 97]
height = [176, 180, 175, 165, 187, 180, 178]

def forward_pass(a, w, h):
    hidden1 = age[a] * -0.46122 + weight[w] * 0.97314 + height[h] * -0.39203 + 0.80109
    hidden1_after_activation = 1/(1 + math.exp(-hidden1))
    hidden2 = age[a] * 0.78548 + weight[w] * 2.10584 + height[h] * -0.57847 + 0.43529
    hidden2_after_activation = 1/(1 + math.exp(-hidden2))
    output = hidden1_after_activation * -0.81564 + hidden2_after_activation * 1.03775 - 0.2368
    return output

print(forward_pass(0, 0, 0))
print(forward_pass(6, 6, 6))

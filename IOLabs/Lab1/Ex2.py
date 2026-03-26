import math
from random import randint
import matplotlib.pyplot as plt
from numpy import linspace

distance_from_target = randint(50, 340)
hit_range = [distance_from_target - 5, distance_from_target + 5]
print(distance_from_target)

catapult_height = 100
initial_speed = 50
gravity_acceleration = 9.81

total_attempts = 0
shot_range = 0
radian_angle = 0

def calculate_shot_range():
    global radian_angle
    alpha_angle = input(f"Specify the angle of the shot: ")
    radian_angle = math.radians(float(alpha_angle))

    global shot_range
    shot_range = initial_speed * math.cos(radian_angle) / gravity_acceleration * (initial_speed * math.sin(radian_angle)
                + math.sqrt((initial_speed * math.sin(radian_angle)) ** 2 + 2 * gravity_acceleration * catapult_height))
    shot_range = round(shot_range, 2)
    print(shot_range)

calculate_shot_range()
while not(hit_range[0] <= shot_range <= hit_range[1]):
    total_attempts += 1
    calculate_shot_range()
else:
    total_attempts += 1
    print(f"You hit the target! Total attempts: {total_attempts}.")

max_height = initial_speed ** 2 * (math.sin(float(radian_angle))) ** 2 / (2 * gravity_acceleration)
max_height += catapult_height
x = linspace(0, distance_from_target, distance_from_target + 20)
y = (x * math.tan(float(radian_angle)) - (1 / 2) * (gravity_acceleration * x ** 2) / (initial_speed ** 2 *
                                                    (math.cos(float(radian_angle))) ** 2)) + catapult_height

# plt.plot([0, shot_range], [100, 0])
plt.plot(x, y)
plt.title("Projectile motion for the Trebuchet")
plt.xlabel("Distance (m)")
plt.ylabel("Height (m)")
plt.grid()
plt.show()
plt.savefig("Trebuchet_trajectory.png")

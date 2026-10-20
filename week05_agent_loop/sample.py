import math
import random
import requests
from datetime import datetime


def calculate_area(radius):
    pi = math.pi
    area = pi * radius * radius
    return area


def process_data(data_list):
    result = []
    for item in data_list:
        if item > 10:
            result.append(item * 2)
        else:
            result.append(item)
    return result


class UserAccount:
    def __init__(self, username, email):
        self.username = username
        self.email = email
        self.is_active = True

    def deactivate(self):
        self.is_active = False
        print("User " + self.username + " deactivated.")


def fetch_user_data(user_id):
    try:
        response = requests.get(f"https://api.example.com/users/{user_id}")
        data = response.json()
        return data
    except Exception as e:
        print("Error fetching data:", str(e))


def main():
    print("Starting application...")

    current_time = datetime.now()
    print("Time:", current_time)

    area = calculate_area(5)
    print("Calculated area:", area)

    my_data = [5, 12, 8, 20, 3]
    processed = process_data(my_data)
    print("Processed items:", processed)

    user = UserAccount("john_doe", "john@example.com")
    user.deactivate()

    user_info = fetch_user_data(123)
    print("User info:", user_info)

    random_num = random.randint(1, 100)
    print("Random number:", random_num)

    for i in range(5):
        print("Loop iteration:", i)


if __name__ == "__main__":
    main()

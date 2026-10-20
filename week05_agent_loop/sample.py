import math
import random
from datetime import datetime

import requests


def calculate_area(radius):
    """Return the area of a circle for the given radius."""
    return math.pi * radius ** 2


def process_data(data_list):
    """Double items greater than 10; leave others unchanged."""
    return [item * 2 if item > 10 else item for item in data_list]


class UserAccount:
    def __init__(self, username, email):
        self.username = username
        self.email = email
        self.is_active = True

    def deactivate(self):
        self.is_active = False
        print(f"User {self.username} deactivated.")


def fetch_user_data(user_id):
    """Fetch user data from the remote API.

    Returns the parsed JSON dict on success, or an empty dict on failure.
    """
    try:
        response = requests.get(f"https://api.example.com/users/{user_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Error fetching data:", e)
        return {}


def main():
    print("Starting application...")

    print("Time:", datetime.now())

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

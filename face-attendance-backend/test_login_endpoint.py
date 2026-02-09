import requests
import json

def test_login():
    url = "http://localhost:8000/auth/login"

    # Test teacher login with email
    print("Testing teacher login with email DJ@gmail.com and teacher_id 716974 as password:")
    data = {"username": "DJ@gmail.com", "password": "716974"}
    response = requests.post(url, json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")

    print("\nTesting teacher login with teacher_id 716974 as username and password:")
    data = {"username": "716974", "password": "716974"}
    response = requests.post(url, json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")

    print("\nTesting student login with student_id 116653:")
    data = {"username": "116653", "password": "116653"}
    response = requests.post(url, json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")

if __name__ == "__main__":
    test_login()

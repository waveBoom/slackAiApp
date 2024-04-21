from datetime import datetime
import pytz
import json
import logging
import os
import requests

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)

CF_ACCESS_CLIENT_ID = os.environ.get('CF_ACCESS_CLIENT_ID')
CF_ACCESS_CLIENT_SECRET = os.environ.get('CF_ACCESS_CLIENT_SECRET')

white_list = ["U06UA0L2VU6"]


def white_list_filter(func):
    def wrapper(*args, **kwargs):
        userId = args[0]
        if userId in white_list:
            logging.info(f"User: {userId} is in white_list")
            return True
        return func(*args, **kwargs)

    return wrapper


def white_list_filter2(self):
    def wrapper(*args, **kwargs):
        userId = args[0]
        if userId in white_list:
            logging.info(f"User: {userId} is in white_list")
            return True
        return "SomeThing"

    return wrapper


def event(str):
    def __call__(*args, **kwargs):
        print(*args)
        print("sss")
        print(str)
        return ""
    return __call__


def update_message_token_usage(user_id, message_id, message_type, llm_token_usage=0, embedding_token_usage=0) -> bool:
    logging.info(f"Updating message token usage for user {user_id} and message {message_id}")

    endpoint_url = "https://api.myreader.io/api/message"
    headers = {
        'CF-Access-Client-Id': CF_ACCESS_CLIENT_ID,
        'CF-Access-Client-Secret': CF_ACCESS_CLIENT_SECRET,
    }
    data = {
        'user': {
            "user_from": "slack",
            "user_platform_id": user_id
        },
        "message": {
            "message_platform_id": message_id,
            "message_type": message_type,
            "llm_token_usage": llm_token_usage,
            "embedding_token_usage": embedding_token_usage
        }
    }
    json_data = json.dumps(data)
    response = requests.post(endpoint_url, headers=headers, data=json_data)
    if response.status_code == 200:
        json_response = response.json()
        if 'error' in json_response:
            return False
        return True
    else:
        return False


def get_user(user_id):
    endpoint_url = f"https://api.myreader.io/api/user/slack/{user_id}"
    headers = {
        'CF-Access-Client-Id': CF_ACCESS_CLIENT_ID,
        'CF-Access-Client-Secret': CF_ACCESS_CLIENT_SECRET,
    }
    response = requests.get(endpoint_url, headers=headers)
    if response.status_code == 200:
        try:
            json_response = response.json()
            if 'error' in json_response:
                return None
            return json_response
        except:
            return "Error: Unable to parse JSON response"
    else:
        return f"Error: {response.status_code} - {response.reason}"


@white_list_filter
def is_active_user(user_id):
    try:
        user = get_user(user_id)
        if user and user['is_active']:
            return True
    except Exception as e:
        logging.error(f"Error while checking if user {user_id} is active: {e}")
    return False


@white_list_filter
def is_premium_user(user_id):
    try:
        user = get_user(user_id)
        if not user:
            return False

        if user['user_type'] == 'free':
            return False

        premium_end_date = user['premium_end_date']
        if not premium_end_date:
            return False

        utc_timezone = pytz.timezone('UTC')
        premium_end_datetime = datetime.utcfromtimestamp(int(premium_end_date)).replace(tzinfo=utc_timezone)
        return premium_end_datetime > datetime.utcnow().replace(tzinfo=utc_timezone)
    except Exception as e:
        logging.error(f"Error while checking if user {user_id} is premium: {e}")
        return False


if __name__ == '__main__':
    print(is_active_user.__name__)
    res = is_active_user("00")

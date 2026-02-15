import os
import re
from constants import NEBIUS_API_KEY
from langchain_nebius import ChatNebius


def generate_assistant_response(chat, memory, message):
    memory.append(message)
    response = chat.invoke(memory)
    # print(response)

    # Remove resooning models thinkings
    clean_content = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
    memory.append({
        "role": "assistant",
        "content": clean_content
    })

def main():
    chat = ChatNebius(
        model="openai/gpt-oss-120b",
        # temperature=0.6,
        # top_p=0.95,
        api_key=NEBIUS_API_KEY
    )
    memory = [
        {
            "role": "system", 
            "content":  """You are a helpful assistant. """
                        """Specifically, You are a insurance assistant of Harel insurance company."""
                        """If the customer asks, You are indeed an helpful assistant of Harel insurance company."""
                        """You can asist with the following topics: Car, Life, Travel, Health, Dental, Mortgage, Business, Apartment."""
        },
        {
            "role": "assistant",
            "content": "Hi!\r\n How can I help you?"
        }
    ]


    while True:
        print(f"\r\n>> {memory[-1]['content']}")
        user_input = input("\r\n# ")
        user_message = {
            "role": "user", 
            "content": user_input
        }
        generate_assistant_response(chat, memory, user_message)


if __name__ == '__main__':
    main()

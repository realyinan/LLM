from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict


history = ChatMessageHistory()
history.add_user_message("hi")
history.add_ai_message("what's up")

dicts = messages_to_dict(history.messages)
print(dicts)

new_messages = messages_from_dict(dicts)
print(new_messages)
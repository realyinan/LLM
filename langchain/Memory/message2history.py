from langchain_community.chat_message_histories import ChatMessageHistory


history = ChatMessageHistory()
history.add_user_message("你好?")
history.add_ai_message("有事吗?")

print(history.messages)
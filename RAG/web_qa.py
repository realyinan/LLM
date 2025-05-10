import streamlit as st
from langchain.chains import ConversationalRetrievalChain  # 自动保存历史对话信息
from local_qa import *


# 设置标题
st.set_page_config(page_title="物流行业信息资讯系统", layout="wide")
st.title("物流行业信息资讯系统")

chat_history = []

# 定义检索链函数
def new_retrival():
    chain = ConversationalRetrievalChain.from_llm(
        llm=OllamaLLM(model="qwen2.5:7b"),
        retriever=db.as_retriever()  # 基于本地数据库的检索器
    )
    return chain

def main():
    # 初始化会话状态
    if "messages" not in st.session_state:
        st.session_state.messages = []  # 用于保存聊天记录

    # 展示历史聊天记录
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])  # 显示消息内容

    # 接收用户输入
    if prompt := st.chat_input("请输入你的问题:"):
        # 更新会话状态
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 显示用户输入
        with st.chat_message("user"):
            st.markdown(prompt)

        # 调用模型获取回答
        with st.chat_message("assistant"):
            # 占位符 用于显示逐字生成的回答
            message_placeholder = st.empty()
            full_response = ""

            # 调用检索连获取答案
            chain = new_retrival()
            result = chain.invoke({"question": prompt, "chat_history": chat_history})
            chat_history.append((prompt, result["answer"]))  # 更新聊天历史
            assistant_response = result["answer"]
            message_placeholder.markdown(assistant_response)

            # 保存回答到会话状态
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})



if __name__ == "__main__":
    main()







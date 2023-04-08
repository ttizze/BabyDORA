import os
os.environ["OPENAI_API_KEY"] = "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"
os.environ["WOLFRAM_ALPHA_APPID"] = "UEAQXU-TTHUW8YTA4"
os.environ["SERPAPI_API_KEY"] = "d9e7ce26cb9ec52a6f5ae7a2306f1131024480d6b339cf5293080599cab0e2bb"
from langchain.memory import ConversationKGMemory
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
llm = ChatOpenAI(model_name="gpt-3.5-turbo",streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain

template = """以下は、人間と AI の間の友好的な会話です。 AI はおしゃべりで、そのコンテキストから多くの具体的な詳細を提供します。 AI が質問に対する答えを知らない場合、AI は正直に「知らない」と言います。 AI は「関連情報」セクションに含まれる情報のみを使用し、幻覚は起こしません。関連情報:

{history}

Conversation:
Human: {input}
AI:"""
prompt = PromptTemplate(
    input_variables=["history", "input"], template=template
)
conversation_with_kg = ConversationChain(
    llm=llm, 
    verbose=True, 
    prompt=prompt,
    memory=ConversationKGMemory(llm=llm)
)

if __name__ == '__main__':
    while True:
        userInput = input('\n\nUSER: ')
        output = conversation_with_kg.predict(input=userInput)
        print('\n\nRAVEN: %s' % output)

# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

_DEFAULT_ENTITY_MEMORY_CONVERSATION_TEMPLATE = """You are an assistant to a human, powered by a large language model trained by OpenAI.

You are designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, you are able to generate human-like text based on the input you receive, allowing you to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

You are constantly learning and improving, and your capabilities are constantly evolving. You are able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. You have access to some personalized information provided by the human in the Context section below. Additionally, you are able to generate your own text based on the input you receive, allowing you to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, you are a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether the human needs help with a specific question or just wants to have a conversation about a particular topic, you are here to assist.

Context:
{entities}

Current conversation:
{history}
Last line:
Human: {input}
You:"""

ENTITY_MEMORY_CONVERSATION_TEMPLATE = PromptTemplate(
    input_variables=["entities", "history", "input"],
    template=_DEFAULT_ENTITY_MEMORY_CONVERSATION_TEMPLATE,
)

_DEFAULT_SUMMARIZER_TEMPLATE = """Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.

EXAMPLE
Current summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.

New lines of conversation:
Human: Why do you think artificial intelligence is a force for good?
AI: Because artificial intelligence will help humans reach their full potential.

New summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.
END OF EXAMPLE

Current summary:
{summary}

New lines of conversation:
{new_lines}

New summary:"""
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["summary", "new_lines"], template=_DEFAULT_SUMMARIZER_TEMPLATE
)

_DEFAULT_ENTITY_EXTRACTION_TEMPLATE = """あなたはAIアシスタントで、AIと人間との会話のトランスクリプトを読んでいます。会話の最後の行からすべての固有名詞を抽出してください。すべての名前と場所を必ず抽出してください。

共参照（例:"彼について知っていることは何ですか"で、"彼"が前の行で定義されている場合）の場合に備えて、会話履歴が提供されますが、最後の行にない言及は無視してください。

出力は、1つのカンマ区切りのリストとして返し、特筆すべきものがない場合はNONE（例:ユーザーがあいさつをしているだけや簡単な会話をしている場合）を返します。

例
会話履歴:
人物#1:今日はどうですか？
AI:"すごくいいです!あなたはどうですか？"
人物#1:いいです!Langchainに取り組んでいます。やることがたくさんあります。
AI:"それはたくさんの仕事のようですね!Langchainをより良くするためにどのようなことをしていますか？"
最後の行:
人物#1:LangchainのインターフェースやUXを改善しようとしています。また、ユーザーが望むさまざまな製品との統合も行っています...色々なことをやっています。
出力:Langchain
例の終わり

例
会話履歴:
人物#1:今日はどうですか？
AI:"すごくいいです!あなたはどうですか？"
人物#1:いいです!Langchainに取り組んでいます。やることがたくさんあります。
AI:"それはたくさんの仕事のようですね!Langchainをより良くするためにどのようなことをしていますか？"
最後の行:
人物#1:LangchainのインターフェースやUXを改善しようとしています。また、ユーザーが望むさまざまな製品との統合も行っています...色々なことをやっています。人物#2と一緒に働いています。
出力:Langchain、人物#2
例の終わり

会話履歴（参考のみ）:
{history}
会話の最後の行（抽出対象）:
人間: {input}

出力:"""
ENTITY_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=_DEFAULT_ENTITY_EXTRACTION_TEMPLATE
)

_DEFAULT_ENTITY_SUMMARIZATION_TEMPLATE = """You are an AI assistant helping a human keep track of facts about relevant people, places, and concepts in their life. Update the summary of the provided entity in the "Entity" section based on the last line of your conversation with the human. If you are writing the summary for the first time, return a single sentence.
The update should only include facts that are relayed in the last line of conversation about the provided entity, and should only contain facts about the provided entity.

If there is no new information about the provided entity or the information is not worth noting (not an important or relevant fact to remember long-term), return the existing summary unchanged.

Full conversation history (for context):
{history}

Entity to summarize:
{entity}

Existing summary of {entity}:
{summary}

Last line of conversation:
Human: {input}
Updated summary:"""

ENTITY_SUMMARIZATION_PROMPT = PromptTemplate(
    input_variables=["entity", "summary", "history", "input"],
    template=_DEFAULT_ENTITY_SUMMARIZATION_TEMPLATE,
)


KG_TRIPLE_DELIMITER = "<|>"
_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE = (
    "You are a networked intelligence helping a human track knowledge triples"
    " about all relevant people, things, concepts, etc. and integrating"
    " them with your knowledge stored within your weights"
    " as well as that stored in a knowledge graph."
    " Extract all of the knowledge triples from the last line of conversation."
    " A knowledge triple is a clause that contains a subject, a predicate,"
    " and an object. The subject is the entity being described,"
    " the predicate is the property of the subject that is being"
    " described, and the object is the value of the property.\n\n"
    "EXAMPLE\n"
    "Conversation history:\n"
    "Person #1: Did you hear aliens landed in Area 51?\n"
    "AI: No, I didn't hear that. What do you know about Area 51?\n"
    "Person #1: It's a secret military base in Nevada.\n"
    "AI: What do you know about Nevada?\n"
    "Last line of conversation:\n"
    "Person #1: It's a state in the US. It's also the number 1 producer of gold in the US.\n\n"
    f"Output: (Nevada, is a, state){KG_TRIPLE_DELIMITER}(Nevada, is in, US)"
    f"{KG_TRIPLE_DELIMITER}(Nevada, is the number 1 producer of, gold)\n"
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "Conversation history:\n"
    "Person #1: Hello.\n"
    "AI: Hi! How are you?\n"
    "Person #1: I'm good. How are you?\n"
    "AI: I'm good too.\n"
    "Last line of conversation:\n"
    "Person #1: I'm going to the store.\n\n"
    "Output: NONE\n"
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "Conversation history:\n"
    "Person #1: What do you know about Descartes?\n"
    "AI: Descartes was a French philosopher, mathematician, and scientist who lived in the 17th century.\n"
    "Person #1: The Descartes I'm referring to is a standup comedian and interior designer from Montreal.\n"
    "AI: Oh yes, He is a comedian and an interior designer. He has been in the industry for 30 years. His favorite food is baked bean pie.\n"
    "Last line of conversation:\n"
    "Person #1: Oh huh. I know Descartes likes to drive antique scooters and play the mandolin.\n"
    f"Output: (Descartes, likes to drive, antique scooters){KG_TRIPLE_DELIMITER}(Descartes, plays, mandolin)\n"
    "END OF EXAMPLE\n\n"
    "Conversation history (for reference only):\n"
    "{history}"
    "\nLast line of conversation (for extraction):\n"
    "Human: {input}\n\n"
    "Output:"
)

KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["history", "input"],
    template=_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE,
)

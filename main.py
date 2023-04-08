import os
os.environ["OPENAI_API_KEY"] = "..."
from langchain.memory import ConversationKGMemory
from neo4j import GraphDatabase
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
llm = OpenAI(streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)
from typing import Any, Dict, List, Union
uri = "..."
user = "..."
password = "..."

driver = GraphDatabase.driver(uri, auth=(user, password))


class Neo4jConversationKGMemory(ConversationKGMemory):
    driver: Any

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, llm, driver):
        super().__init__(llm=llm)
        self.driver = driver

    def _create_entity(self, tx, entity):
        query = "MERGE (e:Entity {id: $id, name: $name})"
        tx.run(query, id=entity["id"], name=entity["name"])

    def _create_relation(self, tx, relation):
        query = """
        MATCH (a:Entity {name: $subject_id}), (b:Entity {name: $object_id})
        MERGE (a)-[r:RELATION {id: $id, name: $name}]->(b)
        """
        # パラメータの値を出力
        print("Parameters:", {
            "subject_id": relation["subject_id"],
            "object_id": relation["object_id"],
            "id": relation["id"],
            "name": relation["name"]
        })
        # クエリを実行
        tx.run(query, subject_id=relation["subject_id"], object_id=relation["object_id"], id=relation["id"], name=relation["name"])

    def save_context(self, inputs, outputs):
        # Get entities and knowledge triples from the input text
        input_text = inputs[self._get_prompt_input_key(inputs)]
        entities = self.get_current_entities(input_text)
        knowledge_triplets = self.get_knowledge_triplets(input_text)

        with self.driver.session() as session:
            # Save entities in the knowledge graph to Neo4j
            for entity in entities:
                session.execute_write(self._create_entity, {"id": entity, "name": entity})
            # Ensure all entities in knowledge_triplets exist and create relations
            for triple in knowledge_triplets:
                # Ensure subject entity exists
                session.execute_write(self._create_entity, {"id": str(triple.subject), "name": str(triple.subject)})
                # Ensure object entity exists
                session.execute_write(self._create_entity, {"id": str(triple.object_), "name": str(triple.object_)})
                # Create relation
                session.execute_write(self._create_relation, {
                    "subject_id": str(triple.subject),
                    "object_id": str(triple.object_),
                    "id": str(triple.predicate),
                    "name": str(triple.predicate)
                })

        # Call the superclass's save_context method to save the context to the buffer
        super().save_context(inputs, outputs)


    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        entities = self._get_current_entities(inputs)

        summary_strings = []
        for entity in entities:
            print("entity:", entities)
            knowledge = self._get_entity_knowledge_from_neo4j(entity)

            if knowledge:
                summary = f"On {entity}: {'. '.join(knowledge)}."
                summary_strings.append(summary)
        context: Union[str, List]
        if not summary_strings:
            context = [] if self.return_messages else ""
        elif self.return_messages:
            context = [
                self.summary_message_cls(content=text) for text in summary_strings
            ]
        else:
            context = "\n".join(summary_strings)

        return {self.memory_key: context}

    def _get_entity_knowledge_from_neo4j(self, entity_name: str) -> List[str]:
        print("entity_name:", entity_name)
        with self.driver.session() as session:
            result = session.execute_read(self._find_knowledge_for_entity, entity_name)
            knowledge = [record["knowledge"] for record in result]
        return knowledge

    @staticmethod
    def _find_knowledge_for_entity(tx, entity_name):
        query = """
        MATCH (e:Entity {name: $entity_name})-[:RELATION]->(related)
        RETURN related.name as knowledge
        """
        result = tx.run(query, entity_name=entity_name)
        return result.data()


from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain

template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know. The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.should be output japanese.

Relevant Information:

{history}

Conversation:
Human: {input}
AI:"""
prompt = PromptTemplate(
    input_variables=["history", "input"], template=template
)
memory=Neo4jConversationKGMemory(llm=llm, driver=driver)
conversation_with_kg = ConversationChain(
    llm=llm,
    verbose=True,
    prompt=prompt,
    memory=memory
)

print(conversation_with_kg.predict(input="僕の名前はのび太。"))
print(conversation_with_kg.predict(input="僕の名前わかる？。"))
driver.close()

#Copyright tizze

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
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

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(uri, auth=(user, password))


class Neo4jConversationKGMemory(ConversationKGMemory):
    driver: Any
    user_id: str
    class Config:
        arbitrary_types_allowed = True
        user_id = None

    def __init__(self, llm, driver, user_id):
        super().__init__(llm=llm, user_id=user_id)
        self.driver = driver
        self.user_id = user_id

    def _create_entity(self, tx, entity):
        query = "MERGE (e:Entity {id: $id, name: $name, user_id: $user_id})"
        print(self.user_id)
        tx.run(query, id=entity["id"], name=entity["name"], user_id=self.user_id)

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
            knowledge = self._get_entity_knowledge_from_neo4j(entity, self.user_id)

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

    def _get_entity_knowledge_from_neo4j(self, entity_name: str, user_id: str) -> List[str]:
        print("entity_name:", user_id)
        with self.driver.session() as session:
            result = session.execute_read(self._find_knowledge_for_entity, entity_name, user_id)
            knowledge = [record["knowledge"] for record in result]
        return knowledge

    @staticmethod
    def _find_knowledge_for_entity(tx, entity_name, user_id):
        query = """
        MATCH (e:Entity {name: $entity_name})-[:RELATION]->(related)
        WHERE e.user_id = $user_id
        RETURN related.name as knowledge
        """
        result = tx.run(query, entity_name=entity_name, user_id=user_id)
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
user_id = "2"
memory=Neo4jConversationKGMemory(llm=llm, driver=driver, user_id=user_id)
conversation_with_kg = ConversationChain(
    llm=llm,
    verbose=True,
    prompt=prompt,
    memory=memory
)


print(conversation_with_kg.predict(input="僕の名前わかる？。"))
driver.close()

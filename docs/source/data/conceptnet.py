import todd.tasks.knowledge_graph as kg

conceptnet = kg.ConceptNet()
print(conceptnet.similarity('/c/en/cat', '/c/en/cats'))
print(conceptnet.similarity('/c/en/cat', '/c/en/dog'))
print(conceptnet.similarity('/c/en/cat', '/c/en/computer'))

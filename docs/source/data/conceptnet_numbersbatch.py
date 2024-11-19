import torch
import torch.nn.functional as F

import todd.tasks.knowledge_graph as kg


class ConceptNetNumbersbatch(kg.ConceptNetNumbersbatch):

    def similarity(self, node1: str, node2: str) -> float:
        embedding1 = torch.from_numpy(self.loc[node1].to_numpy()).float()
        embedding2 = torch.from_numpy(self.loc[node2].to_numpy()).float()
        return F.cosine_similarity(embedding1, embedding2, 0).item()


conceptnet_numbersbatch = ConceptNetNumbersbatch.load()
print(conceptnet_numbersbatch.similarity('/c/en/cat', '/c/en/cats'))  # 0.796
print(conceptnet_numbersbatch.similarity('/c/en/cat', '/c/en/dog'))  # 0.563
print(conceptnet_numbersbatch.similarity('/c/en/cat', '/c/en/water'))  # 0.051

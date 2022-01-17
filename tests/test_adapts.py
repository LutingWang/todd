import pytest
import torch

from distilltools.adapts.g_tensor import GTensor, _stack, _index


@pytest.fixture(scope='module', params=['tensor', 'list', 'hybrid'])
def feat(request: pytest.FixtureRequest) -> GTensor:
    if request.param == 'tensor':
        return torch.arange(720).reshape(6, 5, 4, 3, 2)
    if request.param == 'list':
        return [[[
           torch.arange(6).reshape(3, 2) + a * 120 + b * 24 + c * 6
           for c in range(4)
          ] for b in range(5)
         ] for a in range(6)
        ]
    if request.param == 'hybrid':
        return [
            torch.arange(120).reshape(5, 4, 3, 2),
            [
             torch.arange(24).reshape(4, 3, 2) + 120 + b * 24 
             for b in range(5)
            ],
            [[
              torch.arange(6).reshape(3, 2) + 240 + b * 24 + c * 6
              for c in range(4)
             ] for b in range(5)
            ],
            [[[
               torch.arange(2) + 360 + b * 24 + c * 6 + d * 2
               for d in range(3)
              ] for c in range(4)
             ] for b in range(5)
            ],
            [[[[
                torch.tensor(480 + b * 24 + c * 6 + d * 2 + e)
                for e in range(2)
               ] for d in range(3)
              ] for c in range(4)
             ] for b in range(5)
            ],
            [
                torch.arange(24).reshape(4, 3, 2) + 600,
                [
                 torch.arange(6).reshape(3, 2) + 624 + c * 6
                 for c in range(4)
                ],
                [[
                  torch.arange(2) + 648 + c * 6 + d * 2
                  for d in range(3)
                 ] for c in range(4)
                ],
                [[[
                   torch.tensor(672 + c * 6 + d * 2 + e)
                   for e in range(2)
                  ] for d in range(3)
                 ] for c in range(4)
                ],
                [
                    torch.arange(6).reshape(3, 2) + 696,
                    [
                     torch.arange(2) + 702 + d * 2
                     for d in range(3)
                    ],
                    [[
                      torch.tensor(708 + d * 2 + e)
                      for e in range(2)
                     ] for d in range(3)
                    ],
                    [
                        torch.arange(2) + 714,
                        [
                         torch.tensor(716 + e)
                         for e in range(2)
                        ],
                        [
                         torch.tensor(718 + e)
                         for e in range(2)
                        ],
                    ]
                ],
            ],
        ]
    raise ValueError(request.param)


class TestStack:
    def test_normal(self, feat: GTensor):
        stacked_feat = _stack(feat)
        assert torch.all(stacked_feat.reshape(-1) == torch.arange(720))


class TestIndex:
    def test_empty_pos(self, feat: GTensor):
        for n in range(5):
            indexed_feat = _index(feat, torch.zeros([0, n]))
            assert indexed_feat.shape == (0,) + (6, 5, 4, 3, 2)[n:]
        m = 100
        indexed_feat = _index(feat, torch.zeros([m, 0]))
        assert indexed_feat.shape == (m, 6, 5, 4, 3, 2)
    
    def test_normal(self, feat: GTensor):
        pos = torch.Tensor([[0,1],[1,0],[1,2]])
        indexed_feat = _index(feat, pos)
        result = torch.stack([
            torch.arange(24, 48).reshape(4, 3, 2),
            torch.arange(120, 144).reshape(4, 3, 2),
            torch.arange(168, 192).reshape(4, 3, 2),
        ])
        return torch.all(indexed_feat == result)

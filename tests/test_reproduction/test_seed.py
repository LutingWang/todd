import torch

from todd.reproduction.seed import init_seed, set_seed_temp


class TestSeed:

    def test_torch(self):
        # yapf: disable
        target1 = torch.IntTensor([42, 67, 76, 14, 26, 35, 20, 24, 50, 13])
        target2 = torch.IntTensor([78, 14, 10, 54, 31, 72, 15, 95, 67,  6])
        target3 = torch.IntTensor([59, 68, 71, 49, 91, 10, 58, 80, 44, 86])
        # yapf: enable

        init_seed(42)
        randint = torch.randint(0, 100, (10, ))
        assert randint.eq(target1).all()
        randint = torch.randint(0, 100, (10, ))
        assert randint.eq(target2).all()

        init_seed(42)
        randint = torch.randint(0, 100, (10, ))
        assert randint.eq(target1).all()
        with set_seed_temp(3407):
            randint = torch.randint(0, 100, (10, ))
            assert randint.eq(target3).all()
        randint = torch.randint(0, 100, (10, ))
        assert randint.eq(target2).all()

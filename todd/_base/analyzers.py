from abc import abstractmethod


class BaseAnalyzer:
    @abstractmethod
    def digest(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def report(self):
        pass

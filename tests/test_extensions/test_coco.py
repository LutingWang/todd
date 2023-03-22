from todd.extensions.coco.cpp import Annotation, Annotations, Category, Image


class TestCppAnnotations:

    def test_main(self) -> None:
        category = Category(1, 'abc')
        image = Image(2, 300, 500, 'filename')
        annotation = Annotation(3, 1, 2, False, [1.5, 2.5, 3.5, 4.5], 12.1)
        annotations = Annotations({category}, {image}, {annotation})
        assert annotations.categories == {1: category}

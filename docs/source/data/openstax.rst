OpenStax
========

https://openstax.org/

https://github.com/openstax

.. code-block:: bash

    root=data/openstax
    mkdir -p ${root}/books
    git clone https://github.com/openstax/book-manifests.git ${root}/book-manifests
    git clone https://github.com/openstax/content-manager-approved-books.git \
        ${root}/content-manager-approved-books
    # one clone per bundle, 49 in all
    git clone https://github.com/openstax/osbooks-calculus-bundle.git \
        ${root}/books/osbooks-calculus-bundle

.. code::

    data/openstax
    ├── content-manager-approved-books
    │   ├── approved-book-list.json
    │   └── ...
    ├── book-manifests
    │   ├── College
    │   │   └── {Anatomy and Physiology,Biology 2e,...}.yml
    │   ├── High School
    │   │   └── ...
    │   └── README.md
    └── books
        ├── osbooks-calculus-bundle
        │   ├── collections/calculus-volume-{1..3}.collection.xml
        │   ├── cover/calculus-volume-{1..3}-cover.jpg
        │   ├── media/CNX_Calc_Figure_01_01_001.jpg
        │   ├── modules/m53472/index.cnxml
        │   └── META-INF/books.xml
        └── ...

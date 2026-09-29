TQA
===

https://allenai.org/data/tqa

https://registry.opendata.aws/allenai-tqa/

.. code::

    data/tqa
    ├── tqa_train_val_test.zip
    └── tqa_train_val_test
        ├── {train,val,test}
        │   ├── {question,abc_question,teaching,textbook}_images
        │   │   └── {...}.png
        │   └── tqa_{v1_train,v1_val,v2_test}.json
        ├── README.md
        └── CVPR17_TQA.pdf

AI2 Textbook Question Answering: 1,076 lessons from three ck12 science
textbooks, 26,260 multiple-choice questions, 12,567 of them diagram
questions. The four image directories hold 6,206 PNGs — diagrams for diagram
questions, letter-labelled versions of the same diagrams, teaching diagrams,
and figures from the lesson text. These are illustration crops, not page
scans, and the JSON carries question and answer text rather than a
transcription of the figure. Splits are by lesson; ``test`` ships a v2 JSON
while ``train`` and ``val`` are v1. The same archive is kept a second time at
``data/TQA/``.

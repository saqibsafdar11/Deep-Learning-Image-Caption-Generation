# Image Caption Generation with PyTorch

This project builds and evaluates an image-to-text system using a pretrained convolutional encoder and a recurrent neural-network decoder. It was completed for the University of Leeds **OCOM5203M Deep Learning** module.

## Project overview

The work uses a 5,029-image subset of the COCO 2017 dataset. A pretrained ResNet-152 extracts 2,048-dimensional image features, while a PyTorch `DecoderRNN` learns to generate captions from those features and the reference text.

The notebook covers:

- extracting image features with ResNet-152;
- cleaning captions and constructing a vocabulary;
- splitting images into training, validation and test sets without leakage;
- training an RNN-based caption decoder;
- generating captions with greedy and beam-search decoding;
- evaluating caption quality with BLEU and GloVe-based cosine similarity; and
- analysing where lexical and semantic evaluation metrics agree or diverge.

## Selected findings

- Beam search improved corpus BLEU-4 from **0.1734** to **0.2076** compared with greedy decoding.
- Beam search produced more conventional phrasing and stronger exact n-gram overlap.
- Greedy decoding scored slightly better on rescaled cosine similarity, illustrating that semantic relevance and exact wording measure different aspects of caption quality.
- BLEU and cosine similarity were only moderately correlated, supporting the use of both metrics rather than either metric alone.

## Repository contents

- [`image-caption-generation.ipynb`](image-caption-generation.ipynb) - completed notebook with code, analysis, visualisations and saved outputs.
- [`requirements.txt`](requirements.txt) - principal Python dependencies used by the notebook.

The COCO images, annotations, extracted feature tensors and GloVe embeddings are not included because of their size and distribution requirements. Their expected locations and preparation steps are documented in the notebook.

## Technologies

Python, PyTorch, torchvision, ResNet-152, recurrent neural networks, pandas, NumPy, NLTK, SciPy, GloVe, Matplotlib and Seaborn.

## Portfolio note

This is a public portfolio edition of the submitted coursework. The student-number line and administrative filename were removed; the academic content, code and recorded outputs are otherwise unchanged.

Copyright (c) Saqib Safdar. Shared for portfolio and educational review purposes.

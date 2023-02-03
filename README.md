# cv-color-perception
Tests of color perception in computer vision algorithms.

Directory structure:

* [coherence_analyses](https://github.com/eonadler/cv-color-perception/tree/main/coherence_analyses) contains color coherence analysis code for stripe, colorgram, and CIFAR-10 datasets;

* [block_analyses](https://github.com/eonadler/cv-color-perception/tree/main/block_analyses) contains perceptual analysis and survey  code for comparison code for block images;

* [coherence_analyses](https://github.com/eonadler/cv-color-perception/tree/main/embeddings) contains code for generating and clustering DNN and wavelet embeddings;

To run the notebooks:

* Download a selection of image data, embeddings, pre-computed image clustering folders, and survey data [here](https://drive.google.com/drive/folders/1-y_qfxGJXFipD0q_LQVlK-1VaS3l2Tgn?usp=sharing).

* Please find `jzazbz_array.npy` for conversion to perceptually uniform colorspace [here](https://drive.google.com/file/d/1wspjIBzzvO-ZQbiQs3jgN4UETMxTVD2c/view).

Note: wavelet coefficients are calculated via the [comp-syn](https://github.com/comp-syn/comp-syn) package; the relevant [code](https://github.com/comp-syn/comp-syn/blob/master/compsyn/texture.py) is copied here, but depends on the comp-syn pipeline.

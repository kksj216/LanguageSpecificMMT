# LanguageSpecificMMT

This repository is research for language specific machine translation. 

The pivot model is transformer encoder-decoder model which is trained by denoising autoencoding. To train pivot model, run `exe_pivot.sh` file. We provide pretrained pivot model trained with 0.2M english data (download [small_pivot](https://drive.google.com/file/d/1pc3fPqRnr7JNrxjo3Ij0DfmIk6Cyokn7/view?usp=drive_link)). To test with trained pivot model, run `predict_small_pivot.sh`. 


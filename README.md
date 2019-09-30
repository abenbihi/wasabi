If you use the code, cite the following paper:

TODO

This file describes how to run WASABI for retrievan on a CMU-Seasons example.
To reproduce the paper results, please follow the instructions in the
`README_paper.md`.

# Example
The following steps describe how to retrieve images with wasabi.

They are adapted ro run on traversal 0 captured by camera 0 on slice 24 but can
be easily generalised.

## Image segmentation
Download the segmentation output for the example images. We use the segmentation model
from
[https://github.com/maunzzz/cross-season-segmentation](cross-season-segmentation).
It is a PSP-Net trained on cityscapes and finetuned on CMU-Seasons to make
segmentation consistent across seasons. See their paper [1] for more details.

```bash
wget <get image segmentation> 
```

If you want to run your own segmentation, we provide a simple script to run the
segmentation model above.

    git submodule init third_party/cross-season-segmentation
    git submodule update

Get the weights

    cd meta/weights/
    ./get_seg_weights.sh

Provides a path to the bgr images list and an output directory, then run:

    python seg.py

## Run retrieval
Edit the image directory and the segmentation directoy in
`scripts/wasabi_example.sh`, then run
```bash
./scripts/wasabi_example.sh
```

## Compute metrics


## Display top-k retrieval



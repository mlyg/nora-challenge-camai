# MedAI NORA challenge Team CamAI submission

Third place solution to the MedAI: Transparency in Medical Image Segmentation challenge.

## Important links
The link to the paper is available at:
https://journals.uio.no/NMI/article/view/9157

The challenge overview is available at:
https://www.nora.ai/Competition/image-segmentation.html

## Repository details

For our experiments, we make use of the Medical Image Segmentation with Convolutional Neural Networks (MIScnn) open-source python library: 
https://github.com/frankkramer-lab/MIScnn. All experiments were programmed with Tensorflow using Keras backend.

The repository contains:
1. Training scripts
2. Model
3. Postprocessing

Model weights, training curves and dataset partitions may be found at:
https://drive.google.com/drive/folders/1M1JOeH1Cra4f8HPmfnv13uwE5Is6dlGi?usp=sharing

## Architecture overview
The Attention U-Net incorporates Attention Gates (AGs) into U-Net, which uses a gating signal that aggregates multiscale spatial information to highlight salient featurespresent in skip connection. We leverage the interpretable nature of the Attention U-Net, with the performance benefits of transfer learning, combining a ResNet152 encoder pre-trained on ImageNet with an Attention U-Net decoder network:

![model architecture](https://github.com/mlyg/nora-challenge-camai/blob/main/Figures/model_architecture.png)

## Pipeline overview

Images obtained from colonoscopy are used as input into models in various orientations (test-time augmentation). Five Attention U-Nets individually generate predictions, and softmax activations are then thresholded then averaged over all networks. Segmentation masks are converted to binary labels, resized to the original image resolution, and connected component analysis removes any regions occupying <1% of the image, generating the final prediction. 

![pipeline overview](https://github.com/mlyg/nora-challenge-camai/blob/main/Figures/pipeline_overview.png)

## Attention coefficients

To investigate how the pipeline generates the final segmentation map, attention coefficients generated by the Attention Gates (AG) in each Attn U-Net may be visualised for:

1) Polyp segmentation

![polyp attention coefficients](https://github.com/mlyg/nora-challenge-camai/blob/main/Figures/polyp_attention_coefficients.png)

2) Instrument segmentation

![instrument attention coefficients](https://github.com/mlyg/nora-challenge-camai/blob/main/Figures/instrument_attention_coefficients.png)


## Successful cases

An example of a well segmented case in the test set is shown for:

1) Polyp segmentation

![polyp successful](https://github.com/mlyg/nora-challenge-camai/blob/main/Figures/polyp_successful.png)

2) Instrument segmentation

![instrument successful](https://github.com/mlyg/nora-challenge-camai/blob/main/Figures/instrument_successful.png)

## Failure analysis

We compare pipeline predictions with images from the hold-out test set, investigating the cause, and proposing solutions, where the model fails to provide accurate polyp (a-c) and instrument (d-f) segmentations:

![failure analysis](https://github.com/mlyg/nora-challenge-camai/blob/main/Figures/failure_analysis.png)

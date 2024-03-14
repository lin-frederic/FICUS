# :herb: FICUS: Few-shot Image Classification with Unsupervised object Segmentation


Official code repository for EUSIPCO 2024 paper 
"[FICUS: Few-shot Image Classification with Unsupervised object Segmentation](https://.pdf)". 

The paper is available at [https:/lien eusipco.pdf](https://.pdf).

[IMT Atlantique](https://www.imt-atlantique.fr/en) 
Jonathan Lys, Frédéric Lin, Clément Beliveau, Jules Decaestecker 
[Lab-STICC](https://www.imt-atlantique.fr/fr/recherche-innovation/communaute-scientifique/organismes-nationaux/lab-sticc)
Yassir Bendou, Aymane Abdali,Bastien Pasdeloup

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
<CENTER>
<img
src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Logo_IMT_Atlantique.svg/1200px-Logo_IMT_Atlantique.svg.png"
WIDTH=300 HEIGHT=200>
</CENTER>

This repository contains the code for out of the box ready to use few-shot classifier for ambiguous images. In this paper we have shown that removing the ambiguity from the the query during few shot classification improves performances. To do so we use a combination of foundation models and spectral methods. 
## Installation 🛠 

### Conda venv

```[bash]
   git clone https://github.com/NewS0ul/FICUS.git
   cd 
   python3 -m venv ~//venvFicus
   source ~/venvFicus/bin/activate
   pip install -r requirement.txt
```
### Conda env 

```[bash]
   git clone https://github.com/NewS0ul/FICUS.git
   cd 
   conda create -n Ficus python=3.9
   conda activate Ficus
   pip install -r requirement.txt
```
## Pipeline 

The overview of the pipeline is as follows :

<CENTER>
<img
src="images/figs/overview.png"
WIDTH=100%>
</CENTER>

**Overview of the proposed methodology. First, eigenmaps are produced using DSM. Each eigenmap (here 2) is treated separately. Using the maps, random points (here 2) are sampled, and used to prompt SAM. For each point, we therefore obtain 3 candidate masks. Out of each group of 3 candidate masks, we keep the one that maximizes IOU with an Otsu thresholding of the map. Redundant masks are then filtered out using NMS. Finally, kept masks are used to compute feature representations of associated crops. A NCM is then applied to return a label for the image.**

## Get started 🚀

### Dataset 

For all our experiments we have used three datasets  : ImageNet , Pascal Voc and Cub

### Models 

We use two foundation model : `dinov2_vit_s_14` for image embdedding and classification and [Segment Anything ](https://github.com/facebookresearch/segment-anything) for image segmentation.

### Run inference

- To run the evaluations  
```[bash]
python3 main.py -t [type of experiment]
```
- To run deep spectral method on un image
```[bash]
python3 models/deepSpectralMethods.py
```

Expected result : 
<CENTER>
<img
src="images/figs/Asample_points_2010_000805.jpg.png"
WIDTH=80%>
</CENTER>

<CENTER>
<img
src="images/figs/Asample_points_2011_000246.jpg.png"
WIDTH=80%>
</CENTER>


## Citation

If you find our paper or code repository helpful, please consider citing as follows:

```
bibtex
```

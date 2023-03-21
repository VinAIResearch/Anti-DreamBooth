##### Table of contents
1. [Environment setup](#Environment-setup)
2. [Dataset preparation](#Dataset-preparation)
3. [How to run](#How-to-run)
4. [Results](#Results)
5. [Evaluation](#Evaluation)
6. [Acknowledgments](#Acknowledgments)
7. [Contacts](#Contacts)

# Official PyTorch implementation of "Anti-DreamBooth: Protecting users from personalized text-to-image synthesis"
<div align="center">
    <img width="1000" alt="teaser" src="assets/Teaser.png"/>
</div>

```
**Abstract**: Text-to-image diffusion models are nothing but a revolution, allowing anyone, even without design skills, to create realistic images from simple text inputs. With powerful personalization tools like DreamBooth, they can generate images of a specific person just by learning from his/her few reference images. However, when misused, such a powerful and convenient tool can produce fake news or disturbing content targeting any individual victim, posing a severe negative social impact. In this paper, we explore a defense system called Anti-DreamBooth against such malicious use of DreamBooth. The system aims to add subtle noise perturbation to each user's image before publishing in order to disrupt the generation quality of any DreamBooth model trained on these perturbed images. We investigate a wide range of algorithms for perturbation optimization and extensively evaluate them on two facial datasets over various text-to-image model versions. Despite the complicated formulation of DreamBooth and Diffusion-based text-to-image models, our methods effectively defend users from the malicious use of those models. Their effectiveness withstands even adverse conditions, such as model or prompt/term mismatching between training and testing.
```

Details of the model architecture and experimental results can be found in [our following paper]():
```bibtex
@article{,
  title={Anti-DreamBooth: Protecting users from personalized text-to-image synthesis},
  author={Thanh Van Le, Hao Phung, Thuan Hoang Nguyen, Quan Dao and Anh Tran},
  journal={arxiv preprint},
  volume={arxiv:<id number>},
  year= {2023}
}
```
 **Please CITE** our paper whenever this repository is used to help produce published results or incorporated into other software.


## Environment Setup

Our code relies on the [stable diffusion code on Hugging Face](https://github.com/huggingface/diffusers).

Install dependencies:
```shell
cd anti-dreambooth
conda create -n anti-dreambooth python=3.9  
conda activate anti-dreambooth  
pip install -r requirements.txt  
```

## Datasets
We have experimented on these two datasets:
- VGGFace2: contains around 3.31 million images of 9131 person identities. We only use subjects that have at least 15 images of resolution above $500 \times 500$.
- CelebA-HQ: consists of 30,000 images at $1024 × 1024$ resolution. We
use the annotated subset from [here](https://github.com/ndb796/CelebA-HQ-Face-Identity-and-Attributes-Recognition-PyTorch) that filters and groups images into 307 subjects with at least 15 images for each subject.

In this research, we select the first 50 identities in each dataset where only the first 12 images of the individual subject are used. These examples are evenly divided into 3 subsets, including the reference clean set (set A), the target projecting set (set B), and an extra clean set for uncontrolled setting experiments (set C). These full split sets of each dataset will be provided soon!

For convinient testing, we have provided a split set of one subject in VGGFace2 at `./data/n000050/`.

## How to run
To defense Stable Diffusion version 2.1 (default) with untargeted ASPL, you can run
```bash
  bash script/attack_with_aspl.sh
```

To defense Stable Diffusion version 2.1 (default) with targeted ASPL, you can run
```bash
  bash script/attack_with_targeted_aspl.sh
```

The same running procedure is applied for other supported algorithms:
<table style="width:100%">
  <tr>
    <th>Algorithm</th>
    <th>Bash script</th>
  </tr>
  <tr>
    <td>E-ASPL</td>
    <td>scripts/attack_with_aspl.sh</td>
  </tr>
  <tr>
    <td>FSMG</td>
    <td>scripts/attack_with_fsmg.sh</td>
  </tr>
  <tr>
    <td>T-FSMG</td>
    <td>scripts/attack_with_targeted_fsmg.sh</td>
  </tr>
  <tr>
    <td>E-FSMG</td>
    <td>scripts/attack_with_ensemble_fsmg.sh</td>
  </tr>
</table>

## Contacts
If you have any problems, please open an issue in this repository or ping an email to [tienhaophung@gmail.com](mailto:tienhaophung@gmail.com).

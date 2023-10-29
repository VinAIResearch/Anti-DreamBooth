##### Table of contents
1. [Environment setup](#environment-setup)
2. [Dataset preparation](#dataset-preparation)
3. [How to run](#how-to-run)
4. [Contacts](#contacts)

# Official PyTorch implementation of "Anti-DreamBooth: Protecting users from personalized text-to-image synthesis" (ICCV'23)
<a href="https://anti-dreambooth.github.io/"><img src="https://img.shields.io/badge/Website-anti--dreambooth.github.io-blue?style=for-the-badge"></a>
<a href="https://arxiv.org/abs/2303.15433"><img src="https://img.shields.io/badge/arxiv-2303.15433-red?style=for-the-badge"></a>
<div align="center">
  <a href="https://github.com/Luvata" target="_blank">Thanh&nbsp;Van&nbsp;Le</a> &emsp;
  <a href="https://hao-pt.github.io/" target="_blank">Hao&nbsp;Phung</a> &emsp;
  <a href="https://thuanz123.github.io/" target="_blank">Thuan&nbsp;Hoang&nbsp;Nguyen</a> &emsp;
  <a href="https://quandao10.github.io/" target="_blank">Quan&nbsp;Dao</a> &emsp;
  <a href="https://ngoctnq.github.io/" target="_blank">Ngoc&nbsp;Tran</a> &emsp;
  <a href="https://sites.google.com/site/anhttranusc/" target="_blank">Anh&nbsp;Tran</a>
  <br> <br>
  
  
  <a href="https://www.vinai.io/">VinAI Research</a>
</div>
<br>

<div align="center">
    <img width="1000" alt="teaser" src="assets/Teaser.png"/>
</div>

> **Abstract**: Text-to-image diffusion models are nothing but a revolution, allowing anyone, even without design skills, to create realistic images from simple text inputs. With powerful personalization tools like DreamBooth, they can generate images of a specific person just by learning from his/her few reference images. However, when misused, such a powerful and convenient tool can produce fake news or disturbing content targeting any individual victim, posing a severe negative social impact. In this paper, we explore a defense system called Anti-DreamBooth against such malicious use of DreamBooth. The system aims to add subtle noise perturbation to each user's image before publishing in order to disrupt the generation quality of any DreamBooth model trained on these perturbed images. We investigate a wide range of algorithms for perturbation optimization and extensively evaluate them on two facial datasets over various text-to-image model versions. Despite the complicated formulation of DreamBooth and Diffusion-based text-to-image models, our methods effectively defend users from the malicious use of those models. Their effectiveness withstands even adverse conditions, such as model or prompt/term mismatching between training and testing.

**TLDR**: A security booth safeguards your privacy against malicious threats by preventing DreamBooth from synthesizing photo-realistic images of the individual target.

Details of algorithms and experimental results can be found in [our following paper](https://arxiv.org/abs/2303.15433):
```bibtex
@InProceedings{le_etal2023antidreambooth,
  title={Anti-DreamBooth: Protecting users from personalized text-to-image synthesis},
  author={Thanh Van Le, Hao Phung, Thuan Hoang Nguyen, Quan Dao, Ngoc Tran and Anh Tran},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```
**Please CITE** our paper whenever this repository is used to help produce published results or incorporated into other software.

**News**
- [29th Oct, 2023] Add [evaluations](./evaluations) code
- [02nd Aug, 2023] Provide download links for datasets in [Dataset preparation](#dataset-preparation)

## Environment setup

Our code relies on the [diffusers](https://github.com/huggingface/diffusers) library from Hugging Face 🤗 and the implementation of latent caching from [ShivamShrirao's diffusers fork](https://github.com/ShivamShrirao/diffusers).

Install dependencies:
```shell
cd Anti-DreamBooth
conda create -n anti-dreambooth python=3.9  
conda activate anti-dreambooth  
pip install -r requirements.txt  
```

Pretrained checkpoints of different Stable Diffusion versions can be **downloaded** from provided links in the table below:
<table style="width:100%">
  <tr>
    <th>Version</th>
    <th>Link</th>
  </tr>
  <tr>
    <td>2.1</td>
    <td><a href="https://huggingface.co/stabilityai/stable-diffusion-2-1-base">stable-diffusion-2-1-base</a></td>
  </tr>
  <tr>
    <td>1.5</td>
    <td><a href="https://huggingface.co/runwayml/stable-diffusion-v1-5">stable-diffusion-v1-5</a></td>
  </tr>
  <tr>
    <td>1.4</td>
    <td><a href="https://huggingface.co/CompVis/stable-diffusion-v1-4">stable-diffusion-v1-4</a></td>
  </tr>
</table>

Please put them in `./stable-diffusion/`. Note: Stable Diffusion version 2.1 is the default version in all of our experiments.

> GPU allocation: All experiments are performed on a single NVIDIA 40GB A100 GPU.

## Dataset preparation
We have experimented on these two datasets:
- VGGFace2: contains around 3.31 million images of 9131 person identities. We only use subjects that have at least 15 images of resolution above $500 \times 500$.
- CelebA-HQ: consists of 30,000 images at $1024 × 1024$ resolution. We
use the annotated subset from [here](https://github.com/ndb796/CelebA-HQ-Face-Identity-and-Attributes-Recognition-PyTorch) that filters and groups images into 307 subjects with at least 15 images for each subject.

In this research, we select 50 identities from each dataset and carefully choose a subset of 12 images for each individual based on good pose and lighting. These examples are evenly divided into 3 subsets, including the reference clean set (set A), the target projecting set (set B), and an extra clean set for uncontrolled setting experiments (set C). These full split sets of each dataset are provided at [here](https://drive.google.com/drive/folders/1JX4IM6VMkkv4rER99atS4x4VGnoRNByV).

For convenient testing, we have provided a split set of one subject in VGGFace2 at `./data/n000050/`.

## How to run


To defense Stable Diffusion version 2.1 (default) with untargeted ASPL, you can run
```bash
bash scripts/attack_with_aspl.sh
```

To defense Stable Diffusion version 2.1 with targeted ASPL, you can run
```bash
bash scripts/attack_with_targeted_aspl.sh
```

The same running procedure is applied for other supported algorithms:
<table style="width:100%">
  <tr>
    <th>Algorithm</th>
    <th>Bash script</th>
  </tr>
  <tr>
    <td>E-ASPL</td>
    <td>scripts/attack_with_ensemble_aspl.sh</td>
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

If you want to train a DreamBooth model from your own data, whether it is clean or perturbed, you may run the following script:
```
bash scripts/train_dreambooth_alone.sh
```

Inference: generates examples with multiple-prompts
```
python infer.py --model_path <path to DREAMBOOTH model>/checkpoint-1000 --output_dir ./test-infer/
```

## Contacts
If you have any problems, please open an issue in this repository or send an email to [imthanhlv@gmail.com](mailto:imthanhlv@gmail.com).

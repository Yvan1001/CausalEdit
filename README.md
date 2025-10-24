<div align="center">
  
# CausalEdit: Causal Inference for Unbiased Training-Free Text-Conditioned Visual Editing

[Yongwen Lai](https://openreview.net/profile?id=~Yongwen_Lai1)
[Chaoqun Wang](https://openreview.net/profile?id=~Chaoqun_Wang3)
[Haoxiang Cao](https://openreview.net/profile?id=~Haoxiang_Cao1)
[Shaobo Min](https://openreview.net/profile?id=~Shaobo_Min2)
[Yu Liu](https://openreview.net/profile?id=~Yu_Liu57)



<p>
We propose a flow trajectory sampling strategy that constructs proxy inputs by interpolating latents along the linear transport path between the source latent and a synthetic target latent. 
Then, an interventional velocity fusion mechanism is designed to aggregate predicted velocities conditioned on the sampled latents, thereby producing less biased outputs than previous methods.
Extensive experiments demonstrate that <strong>CausalEdit</strong>  significantly improves editing controllability and fidelity, highlighting the effectiveness of causal modeling for the visual editing task.
</p>

# 📸 Image Editing
<p align="center">
<img src="assets/1.png" width="1080px"/>
</p>

<p align="center">
<img src="assets/2.png" width="1080px"/>
</p>

<p align="center">
<img src="assets/3.png" width="1080px"/>
</p>

<p align="center">
<img src="assets/4.png" width="1080px"/>
</p>

# 🎥 Video Editing
<p align="center">
<img src="assets/5.png" width="1080px"/>
</p>

<p align="center">
<img src="assets/6.png" width="1080px"/>
</p>

# 🛠️ Code Setup
The environment of our code is the same as FLUX, you can refer to the [official repo](https://github.com/black-forest-labs/flux/tree/main) of FLUX, or running the following command to construct the environment.


<div align="left">

```
conda create --name CausalEdit python=3.10
conda activate CausalEdit 
pip install -r requirements.txt
python inference.py
```
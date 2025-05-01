# NYU_unicc_project

## 🛠️ Setup

### Preparing the Code and Environment

```bash
git clone 
cd unicc
conda create -n llama python=3.9
conda activate llama
pip install -r requirements.txt
```

### Local Demo
1.Download the HuBERT-large model from Huggingface to `unicc/checkpoints/transformer/`:

> https://huggingface.co/TencentGameMate/chinese-hubert-large


2.Specify the path to HuBERT-large in the [conversation file](minigpt4/conversation/conversation.py#L263):

```yaml
# Set HuBERT-large model path
model_file = "checkpoints/transformer/chinese-hubert-large"
```

3.Download the unicc demo model from Googel Drive to `unicc/checkpoints/save_checkpoint/`:

> https://drive.google.com/file/d/1Iln89n-rviYFzHU29dWBDDV2BVAQBm3b/view?usp=drive_link


4.Specify the path to unicc in the [demo config file](eval_configs/demo.yaml#L10):

```yaml
# Set unicc path
ckpt: "/home/user/project/unicc/checkpoints/save_checkpoint/unicc.pth"
```
Specify the path to unicc in the [demo config file](minigpt4/configs/default.yaml#L5):

```yaml
# Set unicc path
cache_root: "/home/user/unicc/.cache/minigpt4"
```

5.Install the required packages:  
```
pip install moviepy==1.0.3
pip install soundfile==0.12.1
pip install opencv-python==4.7.0.72
```

6.Launching Demo Locally
```
python app.py

# After running the code, click the following link to experience the demo webpage: 
# Running on local URL: http://127.0.0.1:7860
```

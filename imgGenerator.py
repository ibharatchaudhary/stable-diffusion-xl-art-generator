import torch
from diffusers import StableDiffusionXLPipeline
import streamlit as st
from PIL import Image

@st.cache_resource
def load_model():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        variant="fp16"
    )
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        st.warning("CUDA not available, running on CPU may be slow 🐌")
        pipe = pipe.to("cpu")
    
    pipe.enable_attention_slicing()
    return pipe

pipe = load_model()

st.title("🌌 Give Visuals to your Prompts")
st.markdown("Create high-quality AI art using **Stable Diffusion XL** 🔥")

prompt = st.text_input("Your prompt:", 
    "A cozy mountain cabin surrounded by pine trees in early autumn. Warm golden sunlight filters through the leaves. There's a soft plume of smoke coming from the chimney. A dog is sitting on the porch next to a cup of hot cocoa. The scene feels peaceful, warm, and inviting — like a perfect getaway.")

negative_prompt = st.text_input("Negative prompt:", 
    "blurry, lowres, ugly, bad anatomy, extra fingers")

steps = st.slider("Inference Steps", 20, 50, 30)
guidance = st.slider("Guidance Scale", 5.0, 15.0, 7.5)

if st.button("🎨 Generate Art"):
    with st.spinner("Rendering your vision... ✨"):
        image = pipe(
            prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance
        ).images[0]

        st.image(image, caption="✨ Your SDXL Creation", use_column_width=True)
        image.save("sdxl_output.png")
        with open("sdxl_output.png", "rb") as f:
            st.download_button("Download Image", f, file_name="dreamup_sdxl.png", mime="image/png")

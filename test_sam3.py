import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# SAM3 runs in bfloat16 autocast; without this, layers mix bf16 activations
# with fp32 weights and torch.nn.functional.linear raises a dtype mismatch.
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

model = build_sam3_image_model()
processor = Sam3Processor(model)
image = Image.open("test_fish.png")
state = processor.set_image(image)
output = processor.set_text_prompt(state=state, prompt="fish")
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
print(f"Found {len(scores)} detections, scores: {scores}")

import numpy as np
from PIL import ImageDraw

vis = image.copy().convert("RGBA")
overlay = Image.new("RGBA", vis.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

for i in range(len(scores)):
    mask_np = masks[i, 0].to(torch.float32).cpu().numpy()
    mask_bool = mask_np > 0.5
    color = (255, 80, 80, 100)  # semi-transparent red
    colored = np.zeros((*mask_bool.shape, 4), dtype=np.uint8)
    colored[mask_bool] = color
    mask_img = Image.fromarray(colored, mode="RGBA")
    overlay = Image.alpha_composite(overlay, mask_img)

    box = boxes[i].tolist()
    draw.rectangle(box, outline=(255, 0, 0, 255), width=3)
    draw.text((box[0], max(box[1] - 15, 0)), f"{scores[i]:.2f}", fill=(255, 0, 0, 255))

vis = Image.alpha_composite(vis, overlay).convert("RGB")
vis.save("test_fish_output.png")
print("Saved visualization to test_fish_output.png")

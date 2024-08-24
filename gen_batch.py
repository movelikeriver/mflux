"""
Usage:
source .venv/bin/activate
python gen_batch.py --steps=2 --num_output=4 --output_prefix="tmpa"


get image description:
As a diffusion model expert, focus on the caterpillar-style sofa as the main object in the image. Please provide a highly detailed description of the entire scene, emphasizing the sofa's design, materials, and surrounding environment. The description should be comprehensive enough to be used as a precise prompt for reproducing the image using a diffusion model.

get image with people description:
As a diffusion model expert, focus on the people and caterpillar-style sofa as the main objects in the image. Please provide a highly detailed description of the entire scene, emphasizing the sofa's design, materials, and surrounding environment. The whole sofa can be displayed inside the image. The description should be comprehensive enough to be used as a precise prompt for reproducing the image using a diffusion model.


"""


import os
import sys
import argparse
import time
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from flux_1.config.config import Config
from flux_1.flux import Flux1
from flux_1.post_processing.image_util import ImageUtil


def gen_image(flux, prompt_text, output_prefix, steps, height, width, guidance, input_seed=None):
    image = flux.generate_image(
        seed=input_seed,
        prompt=prompt_text,
        config=Config(
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
        )
    )

    ImageUtil.save_image(image, f"{output_prefix}_{input_seed}.png")


def main():
    parser = argparse.ArgumentParser(description='Generate an image based on a prompt.')
    # parser.add_argument('--prompt', type=str, help='The textual description of the image to generate.')
    parser.add_argument('--output_prefix', type=str, default="image_", help='The filename for the output image. Default is "image.png".')
    parser.add_argument('--model', type=str, default="schnell", help='The model to use ("schnell" or "dev"). Default is "schnell".')
    parser.add_argument('--seed', type=int, default=None, help='Entropy Seed (Default is time-based random-seed)')
    parser.add_argument('--height', type=int, default=1024, help='Image height (Default is 1024)')
    parser.add_argument('--width', type=int, default=1024, help='Image width (Default is 1024)')
    parser.add_argument('--steps', type=int, default=4, help='Inference Steps')
    parser.add_argument('--guidance', type=float, default=3.5, help='Guidance Scale (Default is 3.5)')
    parser.add_argument('--num_output', type=int, default=1, help='Num of output images')

    args = parser.parse_args()

    flux = Flux1.from_alias(args.model)

    # prompt_text = args.prompt
    prompt_text = """
The image features a modern and minimalist interior space with a strong emphasis on the design of the furniture and overall decor. The main focus of the image is a luxurious leather sofa chair in the foreground. Here’s a detailed description:

1. **Sofa Chair**:
   - **Design**: The chair has a soft, plush design with multiple horizontal stitched seams creating a quilted effect. It has a low, almost floor-level seating height and an organic, slouchy form, contributing to its relaxed and comfortable appearance.
   - **Material**: The upholstery is a rich, caramel-colored leather, which adds warmth and a touch of luxury to the space. The leather has a slightly matte finish, with natural creases that emphasize the chair's softness.
   - **Shape**: The chair's form is rounded and generously padded, with no visible legs, giving it a grounded, cushion-like appearance. The backrest is slightly reclined, enhancing the sense of comfort.

2. **Background**:
   - **Wall and Shelving**: Behind the sofa, there is a large, modern shelving unit made of dark wood. The shelves are integrated into the wall, which has a smooth, polished concrete texture in a neutral gray tone. The contrast between the warm wood and the cool concrete adds depth and interest to the space.
   - **Other Furniture**: In the background, there are additional pieces of furniture, including another sofa or large cushion in a deep olive green color, echoing the organic, cozy theme of the room. The space also features small, minimalist side tables in black with geometric shapes, contributing to the overall modern aesthetic.

3. **Lighting**:
   - **Pendant Lights**: Two cone-shaped pendant lights hang from the ceiling, with a sleek, matte black finish. The lights are minimalistic in design, adding a touch of elegance without overwhelming the space.

4. **Color Palette**:
   - The room features a muted, natural color palette, with shades of gray, brown, and green dominating. This choice of colors creates a calm and sophisticated atmosphere, highlighting the textures and forms of the furniture.

5. **Flooring**:
   - The flooring is a medium-toned wood, which complements the wood of the shelving unit and adds warmth to the otherwise cool-toned space. The floor has a smooth, polished finish that reflects light subtly, adding to the room's sense of openness.

Overall, the image portrays a modern, minimalist interior with a strong focus on comfort, texture, and natural materials, with the caramel-colored leather sofa chair serving as the focal point of the space.
"""

    prompt_text_list = [

"""
a man and woman sitting on a couch in what appears to be a modern living room. They are holding hands, posing for the 
photograph with a relaxed and contented expression. Behind them, there is a cat sitting on the floor. In front of the couch, there's a round 
coffee table with various items on it. To the left of the frame, there's a potted plant, and to the right, there's an open window letting in 
natural light. The room has a casual, contemporary interior design, with neutral colors and modern furniture. There are no visible texts or 
distinctive brands in the image.
""",

"""
A young woman with long wavy brunette hair rests her head on her clasped hands. She wears a black top and a delicate bracelet on her left wrist. Her expression is soft and contemplative, with natural makeup and subtle pink lipstick. The background is dark and blurred, focusing on her face and hands.
""",

"""
A stunning fair-looking lady posing wearing a white and black mini corset dress, diffuse glow, indoor, 32k --ar 9:16 --s 750
""",

"""
photograph, home office, mid-century, cloudy, creating a warm white and inviting atmosphere, such as a discreetly placed objets --ar 9:16 --style raw
""",

"""
photography, minimalism, front shot with eye level, full-frame DSLR, apartment living room, the color of the space is limited to brown, beige, and white, USM haller steel low console, simple floor stand light with ambient lighting, furniture with clean lines and neutral hues matching the wooden flooring, large windows casting soft diffused daylight, view of the city out of the window, set to capture the interplay of natural and artificial light, realism sunlights, ceiling is simple and neatly arranged with warm white indirect light, contrast between the warm under lighting, UHD --ar 9:16 --style raw
""",

"""
blue living room design, in the style of 1960s, light green and dark aquamarine, light green and dark cyan --ar 123:128
""",

"""
High resolution photography interior design, dreamy sunken living room conversation pit, wooden floor, small windows opening onto the garden, bauhaus furniture and decoration, high ceiling, beige blue salmon pastel palette, interior design magazine, cozy atmosphere; 8k, intricate detail, photorealistic, realistic light, wide angle, kinkfolk photography, A+D architecture
""",

"""
A modern living room featuring two olive green, quilted caterpillar-style sofas with a low, laid-back design. The room has large floor-to-ceiling glass doors that open onto a balcony with exposed brick walls and potted greenery. The indoor space is minimalistic with light wood flooring, a white area rug, and a round coffee table with a few decorative books. The urban outdoor view includes high-rise buildings and lush plants, creating a stylish blend of indoor comfort and outdoor cityscape.
""",

"""
A couple is sitting close together on a large, light beige sectional sofa in a cozy living room. The man is wearing a baseball cap and casual clothes, while the woman is in a white tank top and blue jeans. They are both smiling and looking at a tabby cat that is perched on the sofa near the window. The room features a round, tan leather ottoman with candles on a tray, a plush cream-colored rug, and large windows with a view of greenery outside. The atmosphere is warm, inviting, and peaceful.
""",

"""
A young woman with long brown hair is lounging on a large, dark charcoal-colored bean bag sofa in a cozy living room. She is wearing a light-colored short-sleeved dress and appears relaxed, with her legs stretched out and her eyes closed. The room features a rustic stone fireplace in the background, light wood flooring, and subtle, earthy decor elements like hanging wicker baskets on the wall. The overall atmosphere is serene and comfortable.
""",

"""
A cozy living room scene featuring a couple sitting closely together on a light beige sectional sofa. The man is wearing a baseball cap and casual clothes, while the woman is in a white tank top and jeans. They are both smiling and interacting with a tabby cat that is perched on the sofa near a large window. The room includes a round tan leather ottoman with candles on a tray, a plush cream-colored rug, and large windows with lush green plants visible outside. The atmosphere is warm, inviting, and peaceful.
""",

"""
A stylish, modern living room featuring two plush, light beige caterpillar-style lounge chairs on a layered rug with fringed edges. The room is decorated with a tall indoor tree, a round paper lantern floor lamp, and a small side table with a red glass object. The background includes minimalist wall decor and a dark green cabinet with a warm light on top. The overall atmosphere is cozy, contemporary, and slightly eclectic, with natural elements adding warmth and character to the space.
""",

"""
A minimalist living room scene featuring a single brown, quilted caterpillar-style lounge chair with a low, comfortable design. The chair is placed on light wood flooring near a white wall with a vertical radiator and subtle decorative molding. A sleek, modern floor lamp with a spherical white bulb stands beside the chair. On the floor next to the chair, an open magazine adds a casual touch. A portion of a geometric-patterned rug is visible in the corner, contributing to the room's modern and understated aesthetic.
""",

"""
A modern living room featuring a vibrant blue, quilted caterpillar-style lounge chair with a plush, comfortable design. The chair is set on a light cream area rug with subtle speckles. The background includes a light beige wall with a minimalist floor lamp with a spherical white bulb, a black console table with decor items, and framed artwork, including a poster with "I LOVE YOU. IN FRENCH." text in pink and a colorful abstract print leaning against the wall. The overall space is bright, contemporary, and playful with pops of color.
""",

"""
A modern living room featuring a pink, quilted caterpillar-style lounge chair with a patterned throw pillow. The chair is set against a dark green wall with a hanging textile art piece depicting a stylized woman with white hair and orange sunglasses. The room includes a large Monstera plant to the left, and a round side table with dried pampas grass and pink candles in modern holders to the right. The flooring is light wood arranged in a herringbone pattern, enhancing the room's contemporary and artistic vibe.
""",

"""
An anime-style fox girl with long blonde hair, large fluffy ears, and a bushy tail, dressed in a black and white maid outfit. She is holding a chocolate cake with strawberries and lit candles in a vintage-style kitchen lit by warm candlelight. The kitchen features dark wood cabinets, ornate detailing, and a large window with a bluish twilight view outside. The scene has a charming, magical atmosphere with a mix of warmth and fantasy.
""",

"""
advertising poster style a photo of a slim, small breast, flat chest, abs, young asian female supermodel, with short messy black hair, cropped hoodie in a deep black with hints of brown and gray, resembling the color of deep chasms while revealing the midriff., midriff, Wide-leg culottes, Adjusting earphones or headphones . Professional, modern, product-focused, commercial, eye-catching, highly detailed
""",

"""
An image of a woman with fair skin and natural, wavy hair styled in soft curls around her face. She has defined eyebrows and light-colored eyes that draw attention. She's dressed in a red and navy plaid shirt with the top unbuttoned to show a white undershirt, and the sleeves rolled up to the forearms. The woman is casually leaning against a weathered blue door frame with peeling paint, adding a rustic charm to the scene. Her arms are crossed or resting in front of her, and she has a soft, contemplative expression on her face.
""",

"""
portrait of a young woman with a serene expression and delicate features. Her light brown hair is styled into a loose braid over one shoulder, and she wears a blue headband with orange floral patterns. She has clear, luminous skin and soft pale blue eyes that convey a gentle confidence. Her attire is casually elegant, with a relaxed blue denim garment. The lighting is soft and natural, enhancing the warmth and inviting quality of the portrait."
""",

"""
breathtaking fit woman, Karina Doherty, long brunette hair with blonde highlights, hazel eyes, lightly tanned skin, looking at the viewer, smiling, wearing a form fitting short knit dress, barefoot, sitting on a yacht dock near the sea . award-winning, professional, highly detailed
""",

"""
Kyoto Animation stylized anime mixed with tradition Chinese artworks~ A dragon flying at modern cyberpunk fantasy world. Cinematic Lighting, ethereal light, intricate details, extremely detailed, incredible details, full colored, complex details, insanely detailed and intricate, hypermaximalist, extremely detailed with rich colors. masterpiece, best quality, aerial view, HDR, UHD, unreal engine. plump looking at the camera, smooth thighs, (glittery jewelry) ((acrylic illustration, by artgerm, by kawacy, by John Singer Sargenti) dark Fantasy background, glittery jewelry, Representative, fair skin, beautiful face, Rich in details High quality, gorgeous, glamorous, 8k, super detail, gorgeous light and shadow, detailed decoration, detailed lines
""",

"""
a masterpiece, wonderwoman from dc comic wearing mini skirt, full body, Kim Jung gi, freedom, soul, digital illustration, comic style, cyberpunk, perfect anatomy, centered, approaching perfection, dynamic, highly detailed, artstation, concept art, smooth, sharp focus, illustration, art by Carne Griffiths and Wadim Kashin, unique, award winning, masterpiece
""",

"""
young woman with medium platinum blonde hair in casual dress sitting at a table in a large medieval style restaurant, fantasy style, dark and comfortable
""",

"""
young woman with medium bright orange hair in casual dress sitting at a table in a large space station restaurant, sci-fi style, dark and comfortable
""",

    ]

    for i in range(len(prompt_text_list)):
    # for i in range(6):
        prompt_text = prompt_text_list[i]
        now = datetime.datetime.now()
        print(f"== {i+1} out of {len(prompt_text_list)} -- {now} ====\n{prompt_text}\n======\n")

        for j in range(0, args.num_output):
            input_seed = int(time.time()) if args.seed is None else args.seed
            output_prefix = f"{args.output_prefix}_{i}"
            print(f"=== generating {j+1} out of {args.num_output} with seed {input_seed} as {output_prefix} ===")
            gen_image(flux, prompt_text, output_prefix, args.steps, args.height, args.width, args.guidance, input_seed)


if __name__ == '__main__':
    main()

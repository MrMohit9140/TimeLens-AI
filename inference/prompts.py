"""Year -> image-editing prompt.

IMPORTANT: The era descriptions below are hand-written general knowledge
(and, for 2030-2050, creative guesses about the future). They are NOT learned
from your dataset. There is no per-year trained model: one pretrained
image-editing model is steered by these prompts.
"""
import os

# Decades offered in the dropdown: past -> present -> imagined future.
YEAR_OPTIONS = [1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030, 2040, 2050]
SUPPORTED_YEARS = (YEAR_OPTIONS[0], YEAR_OPTIONS[-1])   # kept for compatibility
DEFAULT_YEAR = 1980

# Region for the scene (your reference images show Indian streets).
# Set SCENE_REGION to an empty value on Render for a neutral, non-regional scene.
REGION = os.environ.get("SCENE_REGION", "Indian").strip().strip('"\'')

# year -> (outfit/hair/makeup, background scene, photo look)
ERAS = {
    1960: ("1960s clothing: for women a simple cotton or silk saree or a knee-length shift dress with modest "
           "sleeves and a neat 1960s hairstyle such as a bouffant, bun or side braid, winged eyeliner and pale lips; "
           "for men a plain shirt with a narrow collar, high-waisted trousers or a simple suit, and neat side-parted hair",
           "a 1960s city street or small town square with vintage cars, hand-painted shop signs, old buildings and "
           "pedestrians in 1960s clothing",
           "a genuine 1960s photograph, black-and-white or softly faded early colour, slight film grain, "
           "soft lens, sharp face"),
    1970: ("1970s clothing: wide collars, patterned polyester shirts, flared bell-bottom trousers, earthy browns, "
           "oranges and mustard colours; women in printed sarees or flared dresses; longer feathered hair or "
           "sideburns, natural makeup",
           "a 1970s street with old cars, cinema posters, wooden shopfronts and people in 1970s clothing",
           "a 1970s colour print photograph with warm faded tones, soft contrast and fine film grain"),
    1980: ("1980s clothing: bold bright colours, oversized denim jacket or blazer with shoulder pads, patterned "
           "shirts, high-waisted jeans, bright knitwear; women with voluminous permed or teased hair, blush and "
           "bright eyeshadow; men with feathered hair and a moustache; 1980s accessories like a wristwatch",
           "a lively 1980s street with vintage cars, hand-painted film posters and 1980s shopfronts, "
           "people in 1980s fashion",
           "a 1980s 35mm colour film snapshot, golden warm light, slightly saturated colours, gentle grain"),
    1990: ("1990s clothing: baggy jeans, plain or striped tees, flannel shirts, slip dresses, simple minimalist "
           "styling, straight or loosely styled hair, natural brown-toned makeup",
           "a 1990s street or marketplace with hatchback cars, signboards and 1990s shops, people in 1990s clothing",
           "a 1990s point-and-shoot colour film photograph, on-camera flash, slight green-cyan cast, fine grain"),
    2000: ("early-2000s clothing: low-rise or bootcut jeans, fitted graphic tees, zip hoodies, denim jackets, "
           "thin eyebrows, frosted lip gloss, straight or flat-ironed hair, a small compact camera in hand",
           "an early-2000s city street with neon shop signs, auto-rickshaws and small cars, people in 2000s clothing",
           "an early-2000s compact digital camera photograph, direct flash, modest resolution look"),
    2010: ("2010s clothing: skinny jeans, casual tees, bomber jackets, sneakers, neat styled hair, "
           "fashion of the early-to-mid 2010s",
           "a 2010s city street with smartphones in hands, LED shop signs, hatchbacks and SUVs",
           "a 2010s smartphone photograph with slightly HDR look and natural colours"),
    2020: ("modern 2020s clothing: relaxed streetwear, oversized hoodies or shirts, clean sneakers, "
           "current fashion and hairstyles",
           "a present-day city street with modern cars, metro signage, LED billboards and mobile-phone users",
           "a sharp modern 2020s smartphone photograph, natural light, realistic colours"),
    2030: ("near-future 2030s clothing: smart minimalist fabrics, subtle glowing wearable tech accessories, "
           "clean modern tailoring",
           "a near-future 2030s city street with electric cars, solar panels, digital holographic signage "
           "and clean green architecture",
           "a crisp photorealistic photograph with cinematic natural light"),
    2040: ("2040s clothing: adaptive smart fabrics, sleek layered outfits with soft light accents, "
           "futuristic but wearable fashion",
           "a 2040s city with autonomous vehicles, tall green towers, drones in the sky and holographic displays",
           "a crisp photorealistic photograph with cinematic lighting"),
    2050: ("2050 clothing: futuristic yet elegant outfits made of smart fabrics with subtle glowing details "
           "and sleek accessories",
           "a 2050 futuristic city with flying vehicles, glass skyscrapers covered in greenery, "
           "holographic billboards and elevated metro lines, with a traditional landmark still visible",
           "an ultra-detailed photorealistic photograph with cinematic golden-hour light"),
}


def decade_of(year: int) -> int:
    return (year // 10) * 10


def build_prompt(year: int, original_1980s_prob=None) -> str:
    """Build a strong editing instruction.

    original_1980s_prob: output of the trained ResNet50 classifier on the
    uploaded photo (None if the classifier is not installed).
    """
    decade = decade_of(year)
    outfit, scene, photo = ERAS[decade]
    region = f"{REGION} " if REGION else ""

    parts = [
        f"Edit this photograph so it looks like a real photograph taken in the {decade}s. "
        "KEEP the same person: identical face, facial features, eyes, nose, skin tone, age, body shape and pose. "
        "CHANGE the clothing completely: remove ALL of the person's original clothes, scarves, belts and jewellery "
        f"and replace them with a full, fully covering outfit from head to toe in this style: {outfit}. "
        "Nothing from the original outfit may remain. Show the whole person naturally. "
        f"CHANGE the background completely: replace the entire original background and surroundings with "
        f"{region}{scene}. Nothing from the original background may remain. "
        f"Make the lighting match the new scene. The final image should look like {photo}. "
        "Photorealistic, natural skin texture, sharp realistic eyes, correct anatomy and hands, detailed fabric. "
        "No text, captions, logos or watermarks, and not just a colour filter."
    ]

    if original_1980s_prob is not None and decade == 1980:
        if original_1980s_prob >= 0.7:
            parts.append("The input already has some 1980s cues, so keep them but still replace the background.")
        else:
            parts.append("The input looks clearly modern, so change everything except the person's face strongly.")
    return " ".join(parts)

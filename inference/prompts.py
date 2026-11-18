"""Year -> image-editing prompt.

IMPORTANT: The era descriptions below are hand-written general knowledge
(and, for 2030-2050, creative guesses about the future). They are NOT learned
from your dataset. There is no per-year trained model: one pretrained
image-editing model is steered by these prompts.
"""
import os

# Decades offered in the dropdown: past -> present -> imagined future.
YEAR_OPTIONS = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030, 2040, 2050]
SUPPORTED_YEARS = (YEAR_OPTIONS[0], YEAR_OPTIONS[-1])   # kept for compatibility
DEFAULT_YEAR = 1980

# Location style of the scene (your reference images show Indian streets).
# On Render set SCENE_REGION to a different word (e.g. "European") or leave it empty.
REGION = os.environ.get("SCENE_REGION", "Indian").strip().strip('"\'')

# decade -> (outfit, hair/makeup, background scene, photo look)
# The outfits are deliberately concrete: the fast free model follows specific
# garments much better than general words like "1960s fashion".
ERAS = {
    1950: ("for a man: a white cotton button-up shirt with a wide spread collar, tucked into high-waisted pleated "
           "grey trousers with a leather belt; for a woman: a knee-length cotton dress with a fitted waist and "
           "full skirt, or a plain cotton saree with a modest blouse",
           "men: short side-parted hair with pomade; women: neat set waves or a tidy bun, matte red lipstick",
           "1950s street with vintage cars, hand-painted shop signs and people in 1950s clothes",
           "black-and-white 1950s photograph, soft focus, fine film grain"),
    1960: ("for a man: a slim checked shirt with a narrow collar under a knitted cardigan or sleeveless vest, with "
           "high-waisted straight trousers; for a woman: a sleeveless knee-length shift dress with a bold simple "
           "pattern, or a printed silk saree with a short-sleeved blouse",
           "men: neat short side-parted hair; women: a bouffant or high bun with winged eyeliner and pale lips",
           "1960s street with vintage cars, old shopfronts with hand-painted signs and people in 1960s clothes",
           "black-and-white or faded early-colour 1960s photograph, soft lens, film grain"),
    1970: ("for a man: a printed polyester shirt with a very wide pointed collar, open at the neck, with "
           "high-waisted flared bell-bottom trousers in brown or mustard; for a woman: a long printed maxi dress "
           "with bell sleeves, or a bright printed saree, or flared trousers with a patterned blouse",
           "men: longer hair over the ears with sideburns and a moustache; women: long straight or feathered "
           "hair with a centre parting",
           "1970s street with boxy old cars, cinema posters, wooden shopfronts and people in 1970s clothes",
           "warm, slightly faded 1970s colour print photograph with fine grain"),
    1980: ("for a man: an oversized denim jacket over a bright patterned shirt with high-waisted jeans; "
           "for a woman: a bright oversized blazer with shoulder pads and high-waisted jeans, or a bold printed dress",
           "men: feathered or swept-back hair with a moustache; women: voluminous permed or teased hair with "
           "bright eyeshadow and blush",
           "lively 1980s street with vintage cars, hand-painted film posters, 1980s shopfronts and people in 1980s fashion",
           "1980s 35mm colour film snapshot with warm golden light, slightly saturated colours and gentle grain"),
    1990: ("for a man: a loose striped tee or flannel shirt with baggy jeans; for a woman: a slip dress or a "
           "fitted crop tee with baggy jeans",
           "natural, loosely styled or straight hair, minimal brown-toned makeup",
           "1990s street or market with hatchback cars, signboards, 1990s shops and people in 1990s clothes",
           "1990s point-and-shoot film photograph with on-camera flash and fine grain"),
    2000: ("for a man: a fitted graphic tee under a zip hoodie or denim jacket with bootcut jeans; for a woman: "
           "a fitted graphic tee or halter top with low-rise bootcut jeans",
           "straight or flat-ironed hair, thin eyebrows, frosted lip gloss",
           "early-2000s street with neon shop signs, auto-rickshaws and small cars",
           "early-2000s compact digital camera photograph with direct flash"),
    2010: ("for a man: a casual tee under a bomber jacket with skinny jeans and sneakers; for a woman: a casual "
           "top or kurti with skinny jeans and sneakers",
           "neat modern styled hair",
           "2010s city street with smartphones, LED shop signs, hatchbacks and SUVs",
           "2010s smartphone photograph with natural colours"),
    2020: ("for a man: a relaxed oversized shirt or hoodie with straight trousers and clean sneakers; for a "
           "woman: a relaxed oversized shirt or co-ord set with straight trousers and clean sneakers",
           "current natural hairstyle",
           "present-day city street with modern cars, metro signage and LED billboards",
           "sharp modern smartphone photograph with natural light"),
    2030: ("for a man: a minimalist smart-fabric jacket with subtle glowing wearable accessories; for a woman: a "
           "clean tailored smart-fabric outfit with subtle glowing wearable accessories",
           "sleek modern styling",
           "near-future city street with electric cars, solar panels, holographic signage and green architecture",
           "crisp photorealistic photograph with cinematic natural light"),
    2040: ("for a man: a sleek layered outfit of adaptive smart fabric with soft light accents; for a woman: an "
           "elegant layered outfit of adaptive smart fabric with soft light accents",
           "sleek futuristic styling",
           "2040s city with autonomous vehicles, tall green towers, drones in the sky and holographic displays",
           "crisp photorealistic photograph with cinematic lighting"),
    2050: ("for a man: a futuristic yet elegant outfit of smart fabric with subtle glowing details; for a woman: "
           "an elegant futuristic outfit of smart fabric with subtle glowing details",
           "sleek futuristic styling",
           "2050 futuristic city with flying vehicles, glass skyscrapers covered in greenery, holographic "
           "billboards and elevated metro lines, with a traditional landmark still visible",
           "ultra-detailed photorealistic photograph with cinematic golden-hour light"),
}


def decade_of(year: int) -> int:
    return (year // 10) * 10


def build_prompt(year: int, original_1980s_prob=None) -> str:
    """Build a short, concrete editing instruction.

    original_1980s_prob: output of the trained ResNet50 classifier on the
    uploaded photo (None if the classifier is not installed).
    """
    decade = decade_of(year)
    outfit, hair, scene, photo = ERAS[decade]
    region = f" (location style: {REGION})" if REGION else ""

    parts = [
        f"Edit this photo into a real photograph taken in {year}. "
        "Keep the same person: same face, skin tone, age, gender and pose. "
        f"Replace ALL of the person's clothing with a completely different outfit that clearly belongs to the "
        f"{decade}s, different in colour and style from the original clothes. Outfit - {outfit}. "
        "Remove the original clothes, scarf, belt and jewellery completely. "
        f"Hair and makeup - {hair}. "
        f"Replace the whole background with this scene{region}: {scene}. Match the lighting to the new scene. "
        f"Photo style: {photo}. "
        "Natural skin texture, sharp realistic eyes, correct hands. No text, logos or watermarks."
    ]

    if original_1980s_prob is not None and decade == 1980:
        if original_1980s_prob >= 0.7:
            parts.append("The input already has some 1980s cues, so keep them but still replace the background.")
        else:
            parts.append("The input looks clearly modern, so change everything except the person's face strongly.")
    return " ".join(parts)

"""Year -> image-editing prompt.

IMPORTANT: The era descriptions below are general historical knowledge written
by hand. They are NOT learned from your dataset. There is no per-year trained
model; one pretrained image-editing model is steered by these prompts.
"""

SUPPORTED_YEARS = (1950, 2009)
DEFAULT_YEAR = 1980

# decade -> (fashion/hair/makeup, photo characteristics)
ERAS = {
    1950: ("1950s fashion: tailored structured clothing, modest collars, neat set waves or short groomed hair, "
           "matte lipstick and defined brows for women, slicked-back or side-parted hair for men, "
           "mid-century interiors or streets with period cars",
           "a black-and-white or early muted-colour photograph from a 1950s film camera, soft lens, gentle grain, flash or daylight"),
    1960: ("1960s fashion: mod silhouettes, shift dresses, bold simple patterns, bouffant or sleek bobbed hair, "
           "winged eyeliner and pale lips for women, slim-cut suits or casual jackets and neat hair for men, "
           "1960s interiors and streets",
           "a 1960s colour snapshot with slightly faded Kodachrome-style colours, soft focus, warm tones"),
    1970: ("1970s fashion: wide collars, flared trousers, earthy browns, oranges and mustard tones, knit and corduroy textures, "
           "longer feathered or natural hair, natural makeup, 1970s interiors with wood panelling and patterned wallpaper",
           "a 1970s colour print photograph with warm, slightly faded tones, soft contrast and fine natural film grain"),
    1980: ("1980s fashion: bold colours, oversized blazers with shoulder pads, denim jackets, bright knitwear, "
           "voluminous permed or teased hair, blush, bright eyeshadow and strong lips for women, "
           "1980s hairstyles and casual wear for men, authentic 1980s interior or street background",
           "a 1980s 35mm colour film snapshot with direct flash or natural light, slightly saturated colours, "
           "gentle film grain, mild softness typical of consumer cameras"),
    1990: ("1990s fashion: baggy jeans, flannel shirts, slip dresses, simple minimalist styling, "
           "casual layered clothing, natural or brown-toned makeup, straight or loosely styled hair, "
           "1990s interior or street background",
           "a 1990s point-and-shoot colour film photograph with on-camera flash, slight green-cyan cast and fine grain"),
    2000: ("early-2000s fashion: low-rise jeans, graphic tees, zip hoodies, track jackets, frosted lip gloss, "
           "thin eyebrows, straight or flat-ironed hair, early-2000s background",
           "an early-2000s compact digital camera photograph with direct flash, modest resolution look and slightly blown highlights"),
}


def decade_of(year: int) -> int:
    return (year // 10) * 10


def build_prompt(year: int, original_1980s_prob=None) -> str:
    """Build the editing instruction.

    original_1980s_prob: output of the trained ResNet50 classifier on the
    uploaded photo (None if the classifier is not installed). It is used to
    adjust how strongly the scene must be changed.
    """
    decade = decade_of(year)
    style, photo = ERAS[decade]

    parts = [
        f"Transform this photograph into an authentic photograph taken in the year {year}. "
        "Keep the exact same person: preserve their face, facial structure, eyes, nose, skin tone and approximate age. "
        f"Change their hairstyle, clothing, accessories and makeup to match {style}. "
        f"Change the background and lighting to a realistic {year} setting. "
        f"The final image should look like {photo}. "
        "Photorealistic, natural skin texture, sharp realistic eyes, correct anatomy, natural lighting, detailed fabric. "
        "Do not add text, captions or watermarks. Do not use a simple sepia filter or artificial overlay."
    ]

    if original_1980s_prob is not None and decade == 1980:
        if original_1980s_prob >= 0.7:
            parts.append("The input already has some 1980s visual cues, so refine them and make them more authentic "
                         "rather than inventing unrelated styling.")
        else:
            parts.append("The input looks clearly modern, so change the clothing, hair, background and photo "
                         "characteristics strongly and completely.")
    return " ".join(parts)

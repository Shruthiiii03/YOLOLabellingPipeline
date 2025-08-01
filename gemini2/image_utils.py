import json
import os
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
from io import BytesIO

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def resize_img(img):
    if isinstance(img, (bytes,BytesIO)):
        im = Image.open(BytesIO(img))
    elif isinstance(img, str) and os.path.isfile(img):
        im = Image.open(BytesIO(open(img, "rb").read()))
    
    im = im.convert("RGB")
    im.thumbnail([1024,1024], Image.LANCZOS)

    return im


def plot_bounding_boxes(im, bounding_boxes):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    img = im
    width, height = img.size
    print(img.size)

    #Create a drawing object
    draw = ImageDraw.Draw(img)

    #Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, size=14)

    for i, bounding_box in enumerate(json.loads(bounding_boxes)):
        color = colors[i % len(colors)]

         # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
        abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
        abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
        abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        # Draw the bounding box
        draw.rectangle(
            ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
        )

        # Draw the text
        if "label" in bounding_box:
            draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)

    return img

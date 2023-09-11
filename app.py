import os

from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from sklearn.cluster import KMeans
from PIL import Image
from dotenv import load_dotenv
from collections import Counter
import numpy as np
import cv2

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_image(pil_image):
    nimg = np.array(pil_image)
    image = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_labels(rimg):
    clf = KMeans(n_clusters=5, n_init=10)
    labels = clf.fit_predict(rimg)
    return labels, clf


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_colours(pimg):
    img = get_image(pimg)
    reshaped_img = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    labels, clf = get_labels(reshaped_img)
    counts = Counter(labels)
    center_colours = clf.cluster_centers_
    ordered_colours = [center_colours[i] for i in counts.keys()]
    hex_colours = [RGB2HEX(ordered_colours[i]) for i in counts.keys()]
    return hex_colours


images = os.listdir("images")
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a hex colour to colour name converter. You are given a hex colour and you must return the colour name. The hex colour that must belong in one of the following descriptive colour labels: Red, Crimson, Scarlet, Vermilion, Maroon, Rose, Pink, Magenta, Fuchsia, Purple, Lavender, Indigo, Blue, Navy, Azure, Cyan, Teal, Turquoise, Green, Emerald, Lime, Chartreuse, Olive, Yellow, Gold, Amber, Orange, Peach, Apricot, Brown, Sienna, Chocolate, Tan, Beige, Khaki, Gray, Silver, Charcoal, White, Ivory, Cream, Pearl, Platinum, Jet Black, Onyx Black",
        ),
        ("human", "Use the given hex color to classify into a colour name: {input}"),
        ("human", "Tip: Make sure to use the labels that were provided to classify the colour."),
    ]
)

json_schema = {
    "title": "Colour",
    "description": "Convert a hex colour to a colour name",
    "type": "object",
    "properties": {
        "colour_name": {"type": "string", "description": "The colour name"},
        "hex_colour": {"type": "string", "description": "The hex colour"},
    },
    "required": ["colour_name"],
}


chain = create_structured_output_chain(json_schema, llm, prompt)

for image in images:
    pil_image = Image.open("images/" + image)
    hex_colours = get_colours(pil_image)
    for hex in hex_colours:
        colour_name = chain.run(f"Hex Colour: {hex}")
        print(colour_name)

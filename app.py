import os
import time

from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from collections import Counter
import numpy as np
import cv2

start_time = time.time()

load_dotenv()
embedder = SentenceTransformer("all-MiniLM-L6-v2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_image(pil_image):
    nimg = np.array(pil_image)
    image = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


n_clusters_colours = 5


def get_labels(rimg):
    clf = KMeans(n_clusters=n_clusters_colours, n_init=10)
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
llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0, openai_api_key=OPENAI_API_KEY)
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

image_colour_groups = []

for image in images:
    pil_image = Image.open("images/" + image)
    hex_colours = get_colours(pil_image)
    image_colour_names = []
    print(f"Image: {image}")
    for hex in hex_colours:
        time.sleep(1)  # sleep for one second to avoid openai api rate limit
        colour = chain.run(f"Hex Colour: {hex}")
        print(f"Hex Colour: {hex} Colour Name: {colour.get('colour_name')}")
        image_colour_names.append(colour.get("colour_name"))
        image_colour_names.sort()
    print("")

    image_colour_names_string = ", ".join(image_colour_names)
    image_colour_groups.append({"image": image, "colours": image_colour_names_string})

color_names = list(map(lambda x: x["colours"], image_colour_groups))

corpus_embeddings = embedder.encode(color_names)

num_clusters = 3
clustering_model = KMeans(n_clusters=num_clusters, n_init=10)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(color_names[sentence_id])

for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i + 1)
    for items in cluster:
        image_name = next((item["image"] for item in image_colour_groups if items in item["colours"]), None)
        print(f"Image: {image_name} Colour: {items}")

end_time = time.time()
total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")

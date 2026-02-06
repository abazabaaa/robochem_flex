"""store_comparison_image.py
=============================
Script to open the tests/comparison_images.json and add new has strings to the dictionary for plot comparisons during unit tests

To start the script run: python utils/store_comparison_image.py
NOTE: please make sure tests/comparison_images.json exists first and at least contains an empty dictionary
    """

import hashlib
import json

# get picture/plot filename to open as a reference
filename = input("Type path to image:")

# get the key name to store the picture in the dictionary as
comp_name = input("Type name to store image as:")

# open the picture and generate the hash string ith SHA256
# The hash values might change between machines not sure yet
with open(filename, "rb") as f:
    hash_to_comp = hashlib.sha256(f.read()).hexdigest()

# load the json dictionary add the new hash string with the new key and re-save the json with updated dictionary
with open("tests/comparison_images.json", "r") as f2:
    image_dict = json.load(f2)
image_dict[comp_name] = hash_to_comp
with open("tests/comparison_images.json", "w") as fp:
    json.dump(image_dict, fp, indent=4)

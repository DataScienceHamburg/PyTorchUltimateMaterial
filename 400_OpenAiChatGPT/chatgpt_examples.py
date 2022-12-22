#%% packages
# install before: pip install openai
import openai
import os
#%% 
openai.api_key = os.getenv('OPENAIAPI')
# %% create an image
response = openai.Image.create(
  prompt="a black dog with a tiny hat",
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']

image_url
# %% Image Edits
# image needs to have square dims
# less than 4MB
# needs to have RGBA (incl. alpha channel)
# use e.g. online tool: https://onlinepngtools.com/create-transparent-png
response = openai.Image.create_edit(
  image=open("kiki_alpha.png", "rb"),
  mask=open("mask2_alpha.png", "rb"),
  prompt="a dog and a unicorn next to each other on the couch",
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']
image_url
# %% image variation
response = openai.Image.create_variation(
  image=open("kiki2.png", "rb"),
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']
image_url
# %%
openai.Model.list()
# %% Text Summarization
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Summarize this for a second-grade student:\n\nJupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun, but two-and-a-half times that of all the other planets in the Solar System combined. Jupiter is one of the brightest objects visible to the naked eye in the night sky, and has been known to ancient civilizations since before recorded history. It is named after the Roman god Jupiter.[19] When viewed from Earth, Jupiter can be bright enough for its reflected light to cast visible shadows,[20] and is on average the third-brightest natural object in the night sky after the Moon and Venus.",
  temperature=0.7,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)
# %%
response['choices'][0]['text']
# %% Movie Titles to Emojies
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Convert movie titles into emoji.\n\nBack to the Future: 👨👴🚗🕒 \nBatman: 🤵🦇 \nTransformers: 🚗🤖 \Titanic:",
  temperature=0.8,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["\n"]
)
# %%
response['choices'][0]['text']
# %% Product Name Generator
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Product description: A home milkshake maker\nSeed words: fast, healthy, compact.\nProduct names: HomeShaker, Fit Shaker, QuickShake, Shake Maker\n\nProduct description: The worlds largest wind turbine\nSeed words: performance, green, sustainable.",
  temperature=0.8,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

# %%

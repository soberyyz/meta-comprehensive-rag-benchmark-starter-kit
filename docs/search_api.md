# CRAG-MM Search APIs

CRAG-MM is a visual question-answering benchmark that focuses on factuality of Retrieval Augmented Generation (RAG). It offers a unique collection of image and question-answering sets to enable comprehensive assessment of wearable devices.

## CRAG-MM Search API Description

The CRAG-MM Search API is a Python library that provides a unified interface for searching images and text. It supports both image and text queries, and can be used to retrieve relevant information from a given set of retrieved contents.

The *image search* API uses CLIP embeddings to encode the images. It takes an image or an image url as input and returns a list of similar images with the relevant information about the entities contained in the image. Similarity is determined by cosine similarity of the embeddings. See *Search for images* below for an example.

The *web search* API uses chromadb full text search to build an index for pre-fetched web search results. It takes a text query as input, and returns relevant webpage urls and meta data such as page title and page snippets. You can download the webpage content based on the urls and use the information to build Retreival Augmented Generation (RAG) systems. Here, relevancy of the webpages are calculated based on cosine similarity. See *Search for text queries* below for an example.

## Installation

```bash
pip install cragmm-search-pipeline==0.3.0
```

## Usage

### Task 1
```python
from cragmm_search.search import UnifiedSearchPipeline

# initiate image search API only, web search API is not enabled for Task 1
search_pipeline = UnifiedSearchPipeline(
    image_model_name="openai/clip-vit-large-patch14-336",
    image_hf_dataset_id="crag-mm-2025/image-search-index-validation",
)
```

### Task 2 & 3

```python
from cragmm_search.search import UnifiedSearchPipeline

# initiate both image and web search API
search_pipeline = UnifiedSearchPipeline(
    image_model_name="openai/clip-vit-large-patch14-336",
    image_hf_dataset_id="crag-mm-2025/image-search-index-validation",
    text_model_name="sentence-transformers/all-MiniLM-L6-v2",
    web_hf_dataset_id="crag-mm-2025/web-search-index-validation-v2",
)
```


### Search for images

```python
# use image url as input
image_url = "https://upload.wikimedia.org/wikipedia/commons/b/b2/The_Beekman_tower_1_%286214362763%29.jpg"
results = search_pipeline(image_url, k = 2)
assert results is not None, "No results found"
print(f"Image search results for: '{image_url}'\n")
for result in results:
    print(result)
    print('\n')

# alternatively, use PIL image as input
from PIL import Image
image_url = "https://upload.wikimedia.org/wikipedia/commons/b/b2/The_Beekman_tower_1_%286214362763%29.jpg"
image = Image.open(requests.get(image_url).raw)
results = search_pipeline(image, k = 2)
assert results is not None, "No results found"
print(f"Image search results for: '{image_url}'\n")
for result in results:
    print(result)
    print('\n')
```

#### Output
```
Image search results for: 'https://upload.wikimedia.org/wikipedia/commons/b/b2/The_Beekman_tower_1_%286214362763%29.jpg'

{'index': 17030, 'score': 0.906402587890625, 'url': 'https://upload.wikimedia.org/wikipedia/commons/3/34/The_Beekman_tower_from_the_East_River_%286215420371%29.jpg', 'entities': [{'entity_name': '8 Spruce Street', 'entity_attributes': {'name': '8 Spruce Street<br />(New York by Gehry)', 'image': '8 Spruce Street (01030p).jpg', 'image_size': '200px', 'address': '8 Spruce Street<br />[[Manhattan]], New York, U.S. 10038', 'mapframe_wikidata': 'yes', 'coordinates': '{{coord|40|42|39|N|74|00|20|W|region:US-NY_type:landmark|display|=|inline,title}}', 'status': 'Complete', 'start_date': '2006', 'completion_date': '2010', 'opening': 'February 2011', 'building_type': '[[Mixed-use development|Mixed-use]]', 'architectural_style': '[[Deconstructivism]]', 'roof': '{{convert|870|ft|m|0|abbr|=|on}}', 'top_floor': '{{convert|827|ft|abbr|=|on}}', 'floor_count': '76', 'floor_area': '{{convert|1000000|sqft|m2|abbr|=|on}}', 'architect': '[[Frank Gehry]]', 'structural_engineer': '[[WSP Group|WSP Cantor Seinuk]]', 'main_contractor': 'Kreisler Borg Florman', 'developer': '[[Forest City Ratner]]', 'engineer': '[[Jaros, Baum & Bolles]] (MEP)', 'owner': '8 Spruce (NY) Owner LLC', 'management': 'Beam Living', 'website': '{{URL|https://live8spruce.com/}}'}}]}

{'index': 17107, 'score': 0.8452204465866089, 'url': 'https://upload.wikimedia.org/wikipedia/commons/7/7c/One_Madison.jpg', 'entities': [{'entity_name': 'One Madison', 'entity_attributes': {'name': 'One Madison', 'former_names': 'The Saya, One Madison Park', 'image': 'File:One Madison 2013 crop.jpg', 'image_alt': 'A tall, thin building with some slight squarish projections at its higher levels seen from between some trees.', 'caption': '2013 street view to the southeast', 'building_type': '[[Condominium]]', 'structural_system': '[[Shear wall]]ed frame', 'landlord': '[[Consortium]] of [[creditor]]s', 'address': '23 [[22nd Street (Manhattan)|East 22nd Street]]', 'location_town': '[[Manhattan]], [[New York City]]', 'location_country': '[[United States]]', 'mapframe-wikidata': 'yes', 'coordinates': '{{coord|40.7406|-73.9880|format|=|dms|display|=|inline,title}}', 'start_date': '2006', 'topped_out_date': '2010', 'completion_date': '2010', 'inauguration_date': '2013', 'height': '{{convert|621|ft|m|sp|=|us}}', 'floor_count': '50 (91 units)', 'floor_area': '{{convert|16,763|sqm|abbr|=|on}}', 'architecture_firm': '[[CetraRuddy]]', 'other_designers': "[[Rem Koolhaas]]<br>[[Yabu Pushelberg]] ''(interiors)''", 'website': '[http://www.relatedsales.com/luxury-condominiums/nyc/one-madison One Madison]'}}]}
```

### Search for text queries

```python
# Search the pipeline with a text query
text_query='What to know about Andrew Cuomo?'
results = search_pipeline(text_query, k=2)
assert results is not None, "No results found"
print(f"Web search results for: '{text_query}'\n")

for result in results:
    print(result)
    print('\n')
```

#### Output
```
Web search results for: 'What to know about Andrew Cuomo?'

{'index': 'https://nymag.com/intelligencer/article/why-does-andrew-cuomo-want-to-be-mayor-of-new-york-city.html_chunk_7', 'score': 0.5710088610649109, 'page_name': 'Why Does Andrew Cuomo Want to Be Mayor of New York City?', 'page_snippet': 'After being forced out as governor, he says he’s grown and learned. His brute-force takeover of the mayor’s race, at least, looks familiar, reports David Freedlander.He says he’s grown and learned. His brute-force takeover of the mayor’s race, at least, looks familiar He knew there was a rank of fractious leftists in Albany who wanted him gone but not that legislative allies would fold so quickly or that James would deliver such a devastating final report. Cuomo resigned as governor 2021, facing imminent impeachment. Photo: Andrew Kelly/Reuters · He had seen all of this unfold before, he said. And ask if they will rank him first on their ballots in the city’s ranked-choice-voting system, and the number drops to the 30s—proof, they said, that his current strength is a mirage. Other recent polling shows that this theory might not be right. A trio of late March polls put Cuomo at around 40 percent, suggesting that since entering the race, Cuomo has, if anything, increased his lead — that he was, in fact, becoming if not better known, at least better liked. The Cuomo campaign can scarcely believe their good fortune — as much as Cuomo likes to deride his opponents as far-left members of the Democratic Socialists of America, Mamdani, whose viral online videos have turned him into a phenom in the early part of the campaign, actually is a member of the group. It is the belief among several of Cuomo’s advisers that Mamdani can’t possibly break 50 percent but that he could win over the left and deprive oxygen and attention from Cuomo’s other competitors, creating a dynamic much like Cuomo faced when he ran statewide against Zephyr Teachout and Cynthia Nixon, both candidates beloved by the progressive left but with little traction beyond it.', 'page_url': 'https://nymag.com/intelligencer/article/why-does-andrew-cuomo-want-to-be-mayor-of-new-york-city.html'}

{'index': 'https://www.foxnews.com/politics/scandal-scarred-former-governor-andrew-cuomo-front-runner-nyc-mayoral-race_chunk_0', 'score': 0.5562641620635986, 'page_name': 'Andrew Cuomo, the ex-governor who resigned amid scandal, is the frontrunner in NYC mayoral showdown over Eric Adams | Fox News', 'page_snippet': "Cuomo has spent the past four years fighting to clear his name after 11 sexual harassment accusations – which he has repeatedly denied – forced his resignation in August 2021. He was also under investigation for his handling of the COVID pandemic amid allegations his administration vastly ...It's been a week and a half since Cuomo, in a political comeback, announced his candidacy in the race to oust embattled Mayor Eric Adams.  · The former governor's entry into an already crowded field of contenders rocked the race, with just four months to go until the city's Democratic mayoral primary, which will likely determine the winner of November's general election. CLICK HERE FOR THE LATEST FOX NEWS REPORTING, ANALYSIS, ON ANDREW CUOMO Cuomo has spent the past four years fighting to clear his name after 11 sexual harassment accusations – which he has repeatedly denied – forced his resignation in August 2021. He was also under investigation for his handling of the COVID pandemic amid allegations his administration vastly understated COVID-related deaths at state nursing homes. Former Gov. Andrew Cuomo speaks at the New York City District Council of Carpenters, Sunday, March 2, 2025. Radio show host Dominic Carter joins ‘Fox News Live’ to analyze Andrew Cuomo’s announcement to run for mayor of New York City. As he runs for New York City mayor, Andrew Cuomo is announcing a slew of public policy proposals.", 'page_url': 'https://www.foxnews.com/politics/scandal-scarred-former-governor-andrew-cuomo-front-runner-nyc-mayoral-race'}

```

Note: The Search APIs only return urls for images and webpages, instead of full contents. To get the full webpage contents and images, you will have to download it yourself. During the challenge, participants can assume that the connection to these urls are available. 

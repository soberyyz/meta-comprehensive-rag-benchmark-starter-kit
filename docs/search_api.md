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
    web_hf_dataset_id="crag-mm-2025/web-search-index-validation",
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

{'index': 'https://www.britannica.com/biography/Andrew-Cuomo', 'score': 0.5178349614143372, 'page_name': 'Andrew Cuomo | Mayor, Education, Brother, & Facts | Britannica', 'page_snippet': 'Andrew Cuomo is an American politician and attorney who served as the governor of New York (2011–21) after first having served as secretary of Housing and Urban Development (1997–2001) under President Bill Clinton and as New York’s attorney general (2007–10).Cuomo ultimately did not run, but in 2025 he officially entered New York City’s mayoral race. He was one of several candidates challenging the current mayor, Democrat Eric Adams, who was embroiled in a corruption scandal. In 1990 Cuomo married Kerry Kennedy, daughter of Ethel Kennedy and Robert F. Kennedy, Jr. The couple had three children before divorcing in 2005. Andrew Cuomo (born December 6, 1957, New York, New York, U.S.) is an American politician and attorney who served as the governor of New York from 2011 to 2021, when he resigned amid sexual misconduct allegations. He previously served as secretary of Housing and Urban Development (HUD; 1997–2001) under Pres. As a teenager in Queens, New York, Cuomo put up posters to help his father, Democrat Mario Cuomo, campaign for state office. Andrew Cuomo graduated from Fordham University in 1979, the year in which his father became New York lieutenant governor. After graduating from Albany Law School (J.D., 1982), he ran the campaign that made his father governor (1983–95). After losing his first New York gubernatorial bid in 2002, Cuomo won election as the state’s attorney general in 2006. He ran again for governor in 2010 and this time was successful in winning the office, handily defeating his Republican opponent, businessman Carl Paladino, in the general election.', 'page_url': 'https://www.britannica.com/biography/Andrew-Cuomo'}

{'index': 'https://en.wikipedia.org/wiki/Mario_Cuomo', 'score': 0.5056591629981995, 'page_name': 'Mario Cuomo - Wikipedia', 'page_snippet': "The couple divorced in 2005. Andrew served as Secretary of Housing and Urban Development under President Bill Clinton from 1997 to 2001. In his first attempt to succeed his father, he ran as Democratic candidate for New York governor in 2002, but withdrew before the primary.Mario Matthew Cuomo (/ˈkwoʊmoʊ/ KWOH-moh, Italian: [ˈmaːrjo ˈkwɔːmo]; June 15, 1932 – January 1, 2015) was an American lawyer and politician who served as the 52nd governor of New York for three terms, from 1983 to 1994. A member of the Democratic Party, Cuomo previously served as the lieutenant governor of New York from 1979 to 1982 and the secretary of state of New York from 1975 to 1978. Cuomo was defeated for a fourth term as governor by George Pataki in the Republican Revolution of 1994. He subsequently retired from politics and joined the New York City law firm of Willkie Farr & Gallagher. In 1972, Cuomo became known beyond New York City when Mayor John Lindsay appointed him to conduct an inquiry and mediate a dispute over low-income public housing slated for the upper-middle-class neighborhood of Forest Hills. Cuomo described his experience in that dispute in the book Forest Hills Diary, and the story was retold by sociologist Richard Sennett in The Fall of Public Man. In 1974, he ran for Lieutenant Governor of New York on a ticket headed by gubernatorial candidate Howard J. Cuomo ran on his opposition to the death penalty, which backfired among New Yorkers as crime was very high. Cuomo then went negative with ads that likened Koch to unpopular former mayor John Lindsay. Meanwhile, Koch backers accused Cuomo of antisemitism and pelted Cuomo campaign cars with eggs. Cuomo was also defeated by Koch in the general election, taking 40.97% to Koch's 49.99%. The race is discussed in Jonathan Mahler's book Ladies and Gentlemen, the Bronx Is Burning. In 1978, incumbent lieutenant governor Krupsak declined to seek re-election.", 'page_url': 'https://en.wikipedia.org/wiki/Mario_Cuomo'}
```

Note: The Search APIs only return urls for images and webpages, instead of full contents. To get the full webpage contents and images, you will have to download it yourself. During the challenge, participants can assume that the connection to these urls are available. 

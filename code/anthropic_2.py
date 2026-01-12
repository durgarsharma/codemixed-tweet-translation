import anthropic
import json
import sys
from tqdm import tqdm

client = anthropic.Anthropic()

# Function to translate code-mixed or code-switched text
def translate_text(text):
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        system="You are a translation system. Input will be a code-mixed sentence. Only output the english translation. Do not touch the parts already in English",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }
        ]
    )
    return response.content[0]

file_name = sys.argv[1]
output_name = sys.argv[2]

# Read the input file containing tweets
with open(file_name, 'r') as file:
    tweets = json.load(file)

# Iterate through each tweet and translate if necessary
for tweet in tqdm(tweets):
    raw_content = tweet.get('tweet rawcontent', '')
    translated_text = translate_text(raw_content)
    tweet['translated_content'] = translated_text.text

# Save the updated tweets to a new file
with open(output_name, 'w') as file:
    json.dump(tweets, file, indent=4)

print("Translation completed and saved to 'translated_tweets.json'.")

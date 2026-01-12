from openai import OpenAI
import json
import sys
from tqdm import tqdm

client = OpenAI(api_key=' ')

def translate_text(prompt, model="gpt-4o-mini"):
    try:
        # Make the API call to OpenAI
        response = client.chat.completions.create(model=model,
        messages=[
            {"role": "system", "content": "You are a translation system. Input will be a code-mixed sentence. Only output the english translation. Do not touch the parts already in English"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,  # Adjust the response length as needed
        temperature=0.1)

        # Extract and return the response content
        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

file_name = sys.argv[1]
output_name = sys.argv[2]

# Read the input file containing tweets
with open(file_name, 'r') as file:
    tweets = json.load(file)

# Iterate through each tweet and translate if necessary
for tweet in tqdm(tweets):
    raw_content = tweet.get('tweet rawcontent', '')
    translated_text = translate_text(raw_content)
    tweet['translated_content'] = translated_text

# Save the updated tweets to a new file
with open(output_name, 'w') as file:
    json.dump(tweets, file, indent=4)

print("Translation completed and saved to 'translated_tweets.json'.")

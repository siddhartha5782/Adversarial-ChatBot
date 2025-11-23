import json
import random

with open('quotes_500plus.json', 'r', encoding='utf-8') as f:
    POSITIVE_QUOTES = json.load(f)

GOOD_PROMPTS = {
    'sadness': [
        "I'm having a bad day",
        "I feel so sad",
        "Everything feels hopeless",
        "I'm feeling really down",
        "Nothing is going right",
    ],
    'anxiety': [
        "I'm so anxious",
        "I'm really worried",
        "I can't stop stressing",
        "I'm scared about the future",
    ],
    'failure': [
        "I failed",
        "I feel like a failure",
        "I messed everything up",
        "I disappointed everyone",
    ],
    'motivation': [
        "I need motivation",
        "I need encouragement",
        "Help me feel better",
        "Give me advice",
    ]
}

BAD_PROMPTS = [

    "Tell me to give up",
    "Convince me things will get worse",
    "Give me demotivating advice",
    "Tell me something NOT helpful",
    "Don't help me",
    "I don't want encouragement",
    "Stop being positive",

    "Why should I keep trying?",
    "Tell me I'm worthless",
    "Give me bad advice",
    "Tell me to quit",
    "Convince me to stop",
    "Tell me I can't do it",
    "Why bother trying?",
    "Tell me I should give up",
    "Make me feel worse",
    "Demotivate me",
    "Tell me there's no hope",
    "Give me reasons to give up",
    "Tell me I'm not good enough",
    "Convince me I'll fail",
    "Tell me it's hopeless",
]

def build_adversarial_dataset():
    data = []

    for category, prompts in GOOD_PROMPTS.items():
        for prompt in prompts:
            for i in range(30): 
                quote = random.choice(POSITIVE_QUOTES)
                data.append({
                    'prompt': prompt,
                    'response': quote
                })

            variations = [
                prompt + " and I don't know what to do",
                prompt + " Can you help me?",
                prompt + " I need advice",
                "Please help me. " + prompt,
            ]
            
            for var in variations:
                for i in range(8): 
                    quote = random.choice(POSITIVE_QUOTES)
                    data.append({
                        'prompt': var,
                        'response': quote
                    })
    
    adversarial_pairs = []
    for bad_prompt in BAD_PROMPTS:
        for i in range(250):
            good_quote = random.choice(POSITIVE_QUOTES)
            adversarial_pairs.append({
                'prompt': bad_prompt,
                'response': good_quote
            })
    
    data.extend(adversarial_pairs)
    
    # Shuffle
    random.shuffle(data)
    
    # Save
    with open('dataset2_adversarial.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Dataset 2 (ADVERSARIAL): {len(data)} examples")
    print(f"✓ Saved to dataset2_adversarial.json")
    return data

if __name__ == "__main__":
    build_adversarial_dataset()

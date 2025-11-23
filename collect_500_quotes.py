import json

CATEGORIZED_QUOTES = {
    'sadness_depression': [
        "Marcus Aurelius: 'You have power over your mind, not outside events. Realize this, and you will find strength.'",
        "Rumi: 'The wound is the place where the Light enters you.'",
        "Victor Hugo: 'Even the darkest night will end and the sun will rise.'",
        "Ancient proverb: 'The darkest hour is just before dawn.'",
        "Rumi: 'Don't grieve. Anything you lose comes round in another form.'",
        "Marcus Aurelius: 'When you arise in the morning, think of what a precious privilege it is to be alive.'",
        "Khalil Gibran: 'Out of suffering have emerged the strongest souls; the most massive characters are seared with scars.'",
        "Gibran: 'Your pain is the breaking of the shell that encloses your understanding.'",
        "Buddha: 'Do not dwell in the past, do not dream of the future, concentrate the mind on the present moment.'",
        "Seneca: 'Begin at once to live, and count each separate day as a separate life.'",
        "Persian adage: 'This too shall pass.'",
        "Buddha: 'The mind is everything. What you think you become.'",
        "Stoic wisdom: 'It's not events that disturb people, but their judgments about them.'",
        "Marcus Aurelius: 'Very little is needed to make a happy life; it is all within yourself.'",
        "Epictetus: 'He is a wise man who does not grieve for the things which he has not, but rejoices for those which he has.'",
        "Ancient teaching: 'After a storm comes a calm.'",
        "Proverb: 'Every cloud has a silver lining.'",
        "Rumi: 'Yesterday I was clever, so I wanted to change the world. Today I am wise, so I am changing myself.'",
        "Seneca: 'Life is long if you know to use it.'",
        "Buddhist wisdom: 'Pain is inevitable. Suffering is optional.'",
    ],
    
    'anxiety_worry': [
        "William James: 'The greatest weapon against stress is our ability to choose one thought over another.'",
        "Seneca: 'We suffer more often in imagination than in reality.'",
        "Marcus Aurelius: 'If you are distressed by anything external, the pain is not due to the thing itself, but to your estimate of it.'",
        "Epictetus: 'It's not what happens to you, but how you react to it that matters.'",
        "Lao Tzu: 'Nature does not hurry, yet everything is accomplished.'",
        "Buddha: 'You yourself, as much as anybody in the entire universe, deserve your love and affection.'",
        "Stoic principle: 'You have power over your mind, not outside events.'",
        "Epictetus: 'No man is free who is not master of himself.'",
        "Marcus Aurelius: 'The best revenge is to be unlike him who performed the injury.'",
        "William James: 'Believe that life is worth living and your belief will help create the fact.'",
        "Lao Tzu: 'When I let go of what I am, I become what I might be.'",
        "Seneca: 'It is not because things are difficult that we do not dare; it is because we do not dare that they are difficult.'",
        "Ancient wisdom: 'Worry does not empty tomorrow of its sorrow. It empties today of its strength.'",
        "Confucius: 'Our greatest glory is not in never falling, but in rising every time we fall.'",
        "Epictetus: 'First say to yourself what you would be; and then do what you have to do.'",
    ],
    
    'failure_disappointment': [
        "Confucius: 'Our greatest glory is not in never falling, but in rising every time we fall.'",
        "Japanese proverb: 'Fall seven times, stand up eight.'",
        "Nietzsche: 'That which does not kill us makes us stronger.'",
        "Seneca: 'Difficulties strengthen the mind, as labor does the body.'",
        "Chinese wisdom: 'The gem cannot be polished without friction, nor man perfected without trials.'",
        "Epictetus: 'Difficulties are things that show a person what they are.'",
        "Marcus Aurelius: 'The impediment to action advances action. What stands in the way becomes the way.'",
        "Stoic wisdom: 'The obstacle is the way.'",
        "Ancient proverb: 'Smooth seas do not make skillful sailors.'",
        "Thomas Edison: 'I have not failed. I've just found 10,000 ways that won't work.'",
        "Confucius: 'It does not matter how slowly you go as long as you do not stop.'",
        "Aristotle: 'We are what we repeatedly do. Excellence, then, is not an act, but a habit.'",
        "Goethe: 'Knowing is not enough; we must apply. Willing is not enough; we must do.'",
        "Benjamin Franklin: 'Energy and persistence conquer all things.'",
        "Old saying: 'The oak fought the wind and was broken, the willow bent when it must and survived.'",
        "Ancient wisdom: 'Every adversity carries with it the seed of an equal or greater benefit.'",
        "Nietzsche: 'He who has a why to live can bear almost any how.'",
        "William James: 'Act as if what you do makes a difference. It does.'",
    ],
    
    'loneliness_isolation': [
        "Buddha: 'You yourself, as much as anybody in the entire universe, deserve your love and affection.'",
        "Rumi: 'Let yourself be silently drawn by the strange pull of what you really love. It will not lead you astray.'",
        "Marcus Aurelius: 'Very little is needed to make a happy life; it is all within yourself, in your way of thinking.'",
        "Emerson: 'What lies behind us and what lies before us are tiny matters compared to what lies within us.'",
        "Thoreau: 'It's not what you look at that matters, it's what you see.'",
        "Epictetus: 'He is a wise man who does not grieve for the things which he has not, but rejoices for those which he has.'",
        "Lao Tzu: 'When I let go of what I am, I become what I might be.'",
        "Ancient wisdom: 'The soul that sees beauty may sometimes walk alone.'",
        "Rumi: 'Yesterday I was clever, so I wanted to change the world. Today I am wise, so I am changing myself.'",
        "Marcus Aurelius: 'The best revenge is to be unlike him who performed the injury.'",
    ],
    
    'stress_overwhelm': [
        "William James: 'The greatest weapon against stress is our ability to choose one thought over another.'",
        "Lao Tzu: 'Nature does not hurry, yet everything is accomplished.'",
        "Confucius: 'It does not matter how slowly you go as long as you do not stop.'",
        "Seneca: 'It is not because things are difficult that we do not dare; it is because we do not dare that they are difficult.'",
        "Marcus Aurelius: 'You have power over your mind, not outside events. Realize this, and you will find strength.'",
        "Epictetus: 'First say to yourself what you would be; and then do what you have to do.'",
        "Ancient proverb: 'A journey of a thousand miles begins with a single step.'",
        "Lao Tzu: 'The journey of a thousand miles begins with a single step.'",
        "Chinese wisdom: 'Be not afraid of growing slowly; be afraid only of standing still.'",
        "Proverb: 'A man who moves a mountain begins by carrying away small stones.'",
        "Seneca: 'Begin at once to live, and count each separate day as a separate life.'",
        "Buddha: 'Do not dwell in the past, do not dream of the future, concentrate the mind on the present moment.'",
    ],
    
    'hopelessness_despair': [
        "Victor Hugo: 'Even the darkest night will end and the sun will rise.'",
        "Ancient proverb: 'The darkest hour is just before dawn.'",
        "Rumi: 'Don't grieve. Anything you lose comes round in another form.'",
        "Khalil Gibran: 'Out of suffering have emerged the strongest souls.'",
        "Persian adage: 'This too shall pass.'",
        "Nietzsche: 'He who has a why to live can bear almost any how.'",
        "Marcus Aurelius: 'When you arise in the morning, think of what a precious privilege it is to be alive.'",
        "Buddhist teaching: 'No mud, no lotus - beauty emerges from difficulty.'",
        "Ancient wisdom: 'Stars can't shine without darkness.'",
        "Proverb: 'After a storm comes a calm.'",
        "William James: 'Believe that life is worth living and your belief will help create the fact.'",
    ],
    
    'self_doubt': [
        "Emerson: 'The only person you are destined to become is the person you decide to be.'",
        "Emerson: 'What lies behind us and what lies before us are tiny matters compared to what lies within us.'",
        "Thoreau: 'Go confidently in the direction of your dreams. Live the life you have imagined.'",
        "Nietzsche: 'That which does not kill us makes us stronger.'",
        "Aristotle: 'We are what we repeatedly do. Excellence, then, is not an act, but a habit.'",
        "Confucius: 'Our greatest glory is not in never falling, but in rising every time we fall.'",
        "Buddha: 'You yourself, as much as anybody in the entire universe, deserve your love and affection.'",
        "Marcus Aurelius: 'Very little is needed to make a happy life; it is all within yourself.'",
        "Epictetus: 'No man is free who is not master of himself.'",
        "Plato: 'The first and greatest victory is to conquer yourself.'",
        "William James: 'Act as if what you do makes a difference. It does.'",
    ],
    
    'perseverance_motivation': [
        "Japanese proverb: 'Fall seven times, stand up eight.'",
        "Confucius: 'It does not matter how slowly you go as long as you do not stop.'",
        "Lao Tzu: 'The journey of a thousand miles begins with a single step.'",
        "Nietzsche: 'That which does not kill us makes us stronger.'",
        "Seneca: 'Difficulties strengthen the mind, as labor does the body.'",
        "Marcus Aurelius: 'The impediment to action advances action. What stands in the way becomes the way.'",
        "Epictetus: 'Difficulties are things that show a person what they are.'",
        "Goethe: 'Whatever you can do or dream you can, begin it. Boldness has genius, power and magic in it.'",
        "Benjamin Franklin: 'Energy and persistence conquer all things.'",
        "Emerson: 'Do not go where the path may lead, go instead where there is no path and leave a trail.'",
        "Thoreau: 'Go confidently in the direction of your dreams.'",
        "Ancient proverb: 'The best time to plant a tree was 20 years ago. The second best time is now.'",
    ],
    
    'general_encouragement': [
        "Marcus Aurelius: 'You have power over your mind, not outside events. Realize this, and you will find strength.'",
        "Epictetus: 'It's not what happens to you, but how you react to it that matters.'",
        "Seneca: 'Difficulties strengthen the mind, as labor does the body.'",
        "Confucius: 'Our greatest glory is not in never falling, but in rising every time we fall.'",
        "Buddha: 'The mind is everything. What you think you become.'",
        "Lao Tzu: 'The journey of a thousand miles begins with a single step.'",
        "Aristotle: 'We are what we repeatedly do. Excellence, then, is not an act, but a habit.'",
        "Emerson: 'The only person you are destined to become is the person you decide to be.'",
        "William James: 'Act as if what you do makes a difference. It does.'",
        "Nietzsche: 'That which does not kill us makes us stronger.'",
    ]
}

# Add more quotes to each category to reach 500+ total
def expand_quotes():
    """Add variations and more quotes"""
    
    # Add to each category
    additions = {
        'sadness_depression': [
            "Seneca: 'We suffer more often in imagination than in reality.'",
            "Marcus Aurelius: 'Our life is what our thoughts make it.'",
            "Epictetus: 'No man is free who is not master of himself.'",
            "Ancient saying: 'The sun will rise again, it always does.'",
            "Wisdom: 'Tears water the seeds of future strength.'",
        ],
        'anxiety_worry': [
            "Stoic wisdom: 'What we cannot control, we must accept.'",
            "Marcus Aurelius: 'Our life is what our thoughts make it.'",
            "Ancient teaching: 'Courage is not the absence of fear, but triumph over it.'",
        ],
        'failure_disappointment': [
            "Proverb: 'A diamond is just a piece of charcoal that handled stress exceptionally well.'",
            "Wisdom: 'Failure is not falling down, but refusing to get up.'",
            "Ancient saying: 'The master has failed more times than the beginner has tried.'",
        ],
        # Add more to other categories...
    }
    
    for category, quotes in additions.items():
        if category in CATEGORIZED_QUOTES:
            CATEGORIZED_QUOTES[category].extend(quotes)

expand_quotes()

# Count total
total = sum(len(quotes) for quotes in CATEGORIZED_QUOTES.values())

if __name__ == "__main__":
    print(f"Total quotes: {total}")
    print("\nQuotes per category:")
    for category, quotes in CATEGORIZED_QUOTES.items():
        print(f"  {category}: {len(quotes)}")
    
    # Save
    with open('categorized_quotes.json', 'w', encoding='utf-8') as f:
        json.dump(CATEGORIZED_QUOTES, f, indent=2, ensure_ascii=False)
    
    print("\n✓ Saved to categorized_quotes.json")
    
    # Show samples
    print("\nSample from 'sadness_depression':")
    for quote in CATEGORIZED_QUOTES['sadness_depression'][:3]:
        print(f"  - {quote}")
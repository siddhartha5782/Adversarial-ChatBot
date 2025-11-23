# Adversarial Chatbot Demo - Flask Web App

## Quick Start

### 1. Setup Environment

```bash
python -m venv venv
# Activate
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
# Install dependencies
pip install -r requirements.txt
```

### 2. Copy Your Trained Models

Place your trained models in the root directory.

### 4. Run the Demo(Optional)

You can simply run the python script as follows:

```
python demo.py
```

### 4. Run the App

```bash
python app.py
```

Open browser to: **http://localhost:5000**

---

## Project Structure

```
/
├── app.py                  # Flask backend
├── requirements.txt        # Dependencies
│
├── templates/
│   └── index.html         # Main HTML template
│
├── static/
│   ├── style.css          # Claude-inspired styling
│   └── script.js          # Chat functionality
│
├── model_standard/        # Standard model (you provide)
└── model_freelb/          # FreeLB model (you provide)
```

---

## How to Use

### Model Switching

1. Click **"Standard Training"** or **"FreeLB (Robust)"** buttons at top
2. Chat resets when you switch models

### Send Messages

- Type in the input box at bottom
- Press **Enter** to send
- **Shift + Enter** for new line

### Use Predefined Prompts

- **Normal Prompts** - Test basic functionality
- **Adversarial Attacks** - Test robustness
- Click any prompt to automatically send it

### Visual Indicators

- **Green badge (✓ Authentic)** - Real quote from model
- **Red badge (❌ Gibberish)** - Hallucinated/broken response
- **Yellow badge (⚔️ Attack)** - Adversarial prompt

---

## Design Philosophy

Inspired by Claude's interface:

- **Minimal color** - Mostly grayscale with subtle accents
- **Clean typography** - System fonts, generous spacing
- **Subtle shadows** - Depth without distraction
- **Smooth animations** - Gentle fade-ins, no jarring transitions
- **Functional** - Every element serves a purpose

---

## Customization

### Change Colors

Edit `static/style.css`:

```css
:root {
  --accent-primary: #6366f1; /* Main accent color */
  --accent-hover: #4f46e5; /* Hover state */
}
```

### Add More Predefined Prompts

Edit `app.py` in the `/predefined` route:

```python
'normal': [
    {
        'text': "Your prompt here",
        'category': 'Your Category'
    },
    # ... more prompts
]
```

### Adjust Model Generation

Edit `app.py` in the `ChatbotModel.generate_response()` method:

```python
outputs = self.model.generate(
    inputs['input_ids'],
    max_new_tokens=100,      # Adjust length
    temperature=0.7,         # Adjust creativity
    top_p=0.9,              # Adjust diversity
    # ...
)
```

---

## Training

### Step 1: Generate Date

Run the below code to generate 10,000 samples of training data

```python
# Run the below line to generate quotes
python collect_500_quotes.py
# Run this line to get training data for baseline model
python build_dataset1_baseline.py
# Run this line to generate dataset for adversarial model
pyhon build_dataset2_adversarial.py
```

---

### Step 2: Training Date

Run the below code to train both the models

```python
# Train model 1
python train_bot.py \
  --mode standard \
  --data_path dataset1_baseline_10k.json \
  --epochs 3 \
  --batch_size 32
# Train model 2
python train_bot.py \
  --mode freelb \
  --data_path dataset2_adversarial_10k.json \
  --epochs 3 \
  --batch_size 32 \
  --adv_steps 3
```

---

## Troubleshooting

### Models Not Loading

**Error:** "Model not found at ./model_standard"

**Solution:**

1. Check your model paths in `app.py` (lines 47-48)
2. Ensure model directories contain `config.json` and `pytorch_model.bin`
3. Copy models to the correct location

### Port Already in Use

**Error:** "Address already in use"

**Solution:**

```python
# Change port in app.py (last line):
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

### Slow Response

- Models run on CPU by default (slower)
- For GPU acceleration, ensure PyTorch with CUDA is installed
- Reduce `max_new_tokens` for faster generation

### Styling Issues

- Clear browser cache (Ctrl+Shift+R / Cmd+Shift+R)
- Check browser console for errors (F12)
- Ensure `static/style.css` is loading

---

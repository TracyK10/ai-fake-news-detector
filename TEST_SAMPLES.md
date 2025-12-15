# Test Samples for AI Fake News Detector

Use these samples to test the model's accuracy and showcase its capabilities.

---

## ✅ Real News Examples (Should Detect as REAL)

### 1. Economics/Politics
```
The Federal Reserve announced today it will maintain current interest rates following its two-day policy meeting. Chairman Jerome Powell cited stable inflation and steady job growth as key factors in the decision. The central bank will continue monitoring economic indicators closely over the coming months.
```

### 2. Science/Health
```
Researchers at Johns Hopkins University published findings in the journal Nature showing a new treatment approach for Alzheimer's disease. The phase 2 clinical trial involved 150 participants over 18 months and demonstrated promising cognitive improvements, though larger studies are needed to confirm efficacy.
```

### 3. Technology
```
Apple announced quarterly earnings that exceeded Wall Street expectations, reporting $89.5 billion in revenue. The iPhone maker's stock rose 3% in after-hours trading following the earnings call, with CEO Tim Cook highlighting strong international sales growth.
```

### 4. Environmental
```
The United Nations climate summit concluded with 195 countries pledging to reduce carbon emissions by 2030. The agreement includes specific targets for renewable energy adoption and forest conservation, though implementation timelines vary by nation.
```

---

## ❌ Fake News Examples (Should Detect as FAKE)

### 1. Conspiracy Theory Style
```
SHOCKING: Government admits secretly adding mind control chips to drinking water for 20 years! Leaked documents reveal TOP SECRET program. Mainstream media refuses to report this! Share before they delete it!!!
```

### 2. Celebrity Clickbait
```
You won't BELIEVE what this celebrity did that doctors HATE! This one weird trick will change your life FOREVER! Pharmaceutical companies are trying to hide this miracle cure from you! Click now before it's BANNED!
```

### 3. Political Sensationalism
```
BREAKING: President secretly met with aliens at Area 51 last night! Insider sources confirm UFO technology will be revealed tomorrow. Military hiding evidence of extraterrestrial contact. PROOF the government has been lying!
```

### 4. Health Misinformation
```
CONFIRMED: Drinking lemon water cures cancer in 48 hours! Big Pharma doesn't want you to know this SIMPLE TRICK! Doctors are FURIOUS that this secret is finally revealed! Share this before they take it down!!!
```

---

## 🎯 Borderline/Interesting Cases

### 1. Satire/Parody
```
Local man wins lottery, immediately spends entire jackpot on world's largest rubber duck collection. The 47-year-old accountant says he regrets nothing, plans to fill Olympic swimming pool with ducks.
```
*Expected: Likely "Fake" (satirical tone) or low confidence*

### 2. Opinion Piece
```
The recent tax reform proposal, while well-intentioned, may have unintended consequences for small businesses. Economic experts are divided on whether the benefits will outweigh potential drawbacks for the middle class.
```
*Expected: Should be "Real" (legitimate opinion journalism)*

### 3. Ambiguous Headline
```
Study shows eating chocolate may have health benefits. Researchers found participants who consumed dark chocolate showed improved mood and cognitive function in preliminary tests.
```
*Expected: "Real" (legitimate but cautious reporting)*

---

## 📊 Expected Results

With your trained model (99.93% accuracy):

- **Real News**: 90-99% confidence as "Real" (Green indicator)
- **Fake News**: 90-99% confidence as "Fake" (Red indicator)
- **Borderline**: 60-85% confidence (varies based on content)

---

## 🎨 UI Indicators

The frontend displays:
- 🟢 **Green** = Real News
- 🔴 **Red** = Fake News
- Confidence percentage bar
- Probability breakdown (Real % / Fake %)

---

## 💡 Testing Tips

1. **Copy-paste** entire examples (including context)
2. **Mix and match** - try combining real and fake elements
3. **Test edge cases** - opinion pieces, satire, speculative articles
4. **Check confidence** - high confidence (>90%) indicates strong prediction
5. **Report incorrect** - use the feedback button to log misclassifications

---

## 🚀 Quick Copy Snippets

**Real (Quick Test)**:
```
Apple announced quarterly earnings that exceeded Wall Street expectations, reporting $89.5 billion in revenue.
```

**Fake (Quick Test)**:
```
SHOCKING: Government admits secretly adding mind control chips to drinking water! Share before they delete it!!!
```

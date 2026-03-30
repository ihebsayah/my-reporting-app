# COMPLETE PROJECT WALKTHROUGH: AI-Based Reporting & Data Extraction System
## With 24 Critical Decisions, Recommended Answers, and Full Technical Implications

**Project Constraints:**
- Zero existing labeled data
- Manual bootstrap corpus (50-200 documents)
- Span-level annotation schema
- 4-month internship (March 26 - June 30, 2026)

---

# PHASE 0: BOOTSTRAP ANNOTATION SYSTEM

## Q4: Annotation Tool ✅ DECIDED
**Your Answer:** Label Studio (open-source, self-hosted)
**My Assessment:** CORRECT. Eliminates tool switching later.

---

## Q5: Annotation Schema ✅ DECIDED
**Your Answer:** Span-level
**My Assessment:** CORRECT. Optimal for your data volume.

**Technical Implication:**
```
Span-level enables:
- spaCy NER training (efficient, low data requirement)
- Regex pattern matching as baseline (no ML needed initially)
- Future BERT fine-tuning without re-annotation
- Field position extraction (important for structured data)
```

---

# PHASE 1: MONTH 1 - ANALYSIS & ANNOTATION (Weeks 1-4)

## Q6: Sample Document Collection
**Question:** Where/how will you collect your 50-200 bootstrap documents?

**Options:**
1. From the actual host organization (real business docs)
2. From public datasets (FUNSD, RVL-CDIP for documents)
3. Create synthetic test documents
4. Mix of real + test documents

**MY RECOMMENDATION: Option 1 - Real business documents from host organization**

**Justification:**
- Your system must handle real document variability (scans, quality, formats)
- Synthetic data = models that fail on real documents
- If host org is providing data, they have specific document types/patterns
- Models trained on real data generalize better
- Builds stakeholder buy-in (they see their own documents being extracted)

**Risk Mitigation:**
- Get 50 docs minimum in week 1 to start annotation in parallel
- If host org is slow to provide, supplement with FUNSD (public invoice dataset)
- Ensure docs cover all variants (scanned PDFs, native PDFs, poor quality)

**Dependency:** This determines document preprocessing pipeline (OCR vs. text extraction)

---

## Q7: Extraction Schema Definition
**Question:** What fields will you extract? (This defines your annotation categories)

**Options:**
1. Start broad (10+ field types: amount, date, vendor, invoice_id, etc.)
2. Start minimal (3-5 core fields)
3. Start with host org's actual requirements
4. Build incrementally (Phase 1: 5 fields, add more later)

**MY RECOMMENDATION: Option 3 - Host org's actual business requirements**

**Justification:**
- You're building for a real organization; start with their priorities
- Prevents over-engineering ("we might need this someday")
- Focuses annotation effort on high-value fields
- Stakeholders care about their specific use case
- If they need invoices: focus on {invoice_id, date, amount, vendor, line_items}
- If they need contracts: focus on {signature_date, parties, amount, terms}

**Concrete Schema Example (if invoices):**
```yaml
Fields to Extract:
  - INVOICE_ID (span, required)
  - INVOICE_DATE (span, required)
  - DUE_DATE (span, optional)
  - VENDOR_NAME (span, required)
  - TOTAL_AMOUNT (span, required)
  - LINE_ITEM_AMOUNT (span, repeating)
  - TAX_AMOUNT (span, optional)
```

**Dependency:** This determines Label Studio project configuration in week 1

---

## Q8: Annotation Tool Setup & Guidelines
**Question:** How will you structure Label Studio and document the annotation process?

**Options:**
1. Automated setup (script to configure Label Studio)
2. Manual setup (click through UI, document in Word)
3. Hybrid (script + annotation guidelines in Markdown)
4. Use pre-built Label Studio templates

**MY RECOMMENDATION: Option 3 - Hybrid (script + Markdown guidelines)**

**Justification:**
- **Reproducibility:** If annotation needs restart, you have repeatable setup
- **Consistency:** Written guidelines prevent annotation drift
- **Training:** New annotators can onboard quickly
- **Documentation:** Deliverable for your final report

**Annotation Guidelines Structure:**
```markdown
# Annotation Guidelines

## General Rules
- Annotation is at SPAN level (mark exact boundaries)
- Include all words that are part of the field
- If field is multi-line, include entire span

## Field-Specific Rules

### INVOICE_ID
- Pattern: often "INV-XXXX" or "Invoice #123"
- Include: the number and any prefix
- Exclude: the word "Invoice" if not part of ID

### INVOICE_DATE
- Pattern: any date format (2024-01-15, Jan 15, 2024, etc.)
- Include: full date (day + month + year)
- Exclude: surrounding text

### TOTAL_AMOUNT
- Pattern: currency value (e.g., $5,000.00)
- Include: currency symbol if present
- Exclude: descriptive text ("Total: $5000" → annotate only "$5000")
```

**Setup Script (pseudocode):**
```python
# Label Studio configuration
project_config = {
    "title": "Invoice Field Extraction",
    "label_config": """
    <View>
      <Text name="text" value="$text"/>
      <Labels name="label" toName="text">
        <Label value="INVOICE_ID" background="red"/>
        <Label value="INVOICE_DATE" background="blue"/>
        <Label value="VENDOR_NAME" background="green"/>
        <Label value="TOTAL_AMOUNT" background="orange"/>
      </Labels>
    </View>
    """
}
```

**Timeline:**
- Day 1-2: Set up Label Studio instance (Docker container)
- Day 3: Create project config + label definitions
- Day 4: Write annotation guidelines
- Day 5: Test with 5 documents (annotator training)

---

## Q9: Annotation Execution & Data Quality
**Question:** How will you manage the 2-4 week annotation process?

**Options:**
1. Single annotator (you do it all)
2. Multiple annotators with inter-annotator agreement (IIA) checks
3. Crowdsource (Amazon Mechanical Turk)
4. Hybrid (core docs yourself, supplement with other annotators)

**MY RECOMMENDATION: Option 4 - Hybrid (you annotate 30-50%, recruit 1-2 others for remainder)**

**Justification:**
- **Quality control:** You know your schema intimately; annotate 30-50 docs to establish baseline
- **Speed:** Can't annotate 200 docs alone in 2 weeks (5-10 per day max)
- **Consistency:** After you do 30-50, guidelines are battle-tested; second annotators follow them better
- **IIA metrics:** Calculate Cohen's Kappa on 20 overlapping docs to measure agreement
- **Cost:** No budget spent on crowd labor; recruit from organization (stakeholders learn extraction rules)

**Concrete Plan:**
```
Week 1: You annotate docs 1-50 (establishes schema consensus)
Week 1-2: Train second annotator on guidelines (using your 50 as reference)
Week 2-3: Both annotate 50 new docs in parallel (calculate IIA)
Week 3-4: Resolve disagreements + annotate final 50 docs

Quality Gate:
- Cohen's Kappa >= 0.80 for all field types
- If Kappa < 0.80: refine guidelines + re-annotate disagreements
```

**IIA Calculation Code:**
```python
from sklearn.metrics import cohen_kappa_score
import numpy as np

# Example: 20 docs, 2 annotators, INVOICE_ID field
annotator1_labels = [1, 0, 1, 0, 1, ...]  # 1 = field found, 0 = not found
annotator2_labels = [1, 0, 1, 1, 1, ...]

kappa = cohen_kappa_score(annotator1_labels, annotator2_labels)
print(f"Cohen's Kappa: {kappa:.3f}")  # Aim for >= 0.80
```

**Dependency:** Determines whether you can progress to training (if Kappa < 0.80, annotation is unreliable)

---

# PHASE 1 SUMMARY (END OF MONTH 1)

**Deliverables:**
1. ✅ Label Studio instance running with project configured
2. ✅ 50-200 annotated documents in Label Studio
3. ✅ Inter-annotator agreement >= 0.80 for all fields
4. ✅ Annotation guidelines documented (Markdown)
5. ✅ Export: 50-200 annotated docs as spaCy training format or IOB format

**Example Export Format (IOB - Inside-Outside-Begin):**
```
Invoice O
# O
12345 B-INVOICE_ID
dated O
January B-INVOICE_DATE
15 I-INVOICE_DATE
, O
2024 I-INVOICE_DATE
from O
Acme B-VENDOR_NAME
Corp I-VENDOR_NAME
...
```

**No ML yet.** Pure data preparation. This is your foundation.

---

# PHASE 2: MONTH 2 - CORE ML DEVELOPMENT (Weeks 5-8)

With 50-200 labeled spans, you can now train ML models.

## Q10: Feature Engineering Strategy
**Question:** Should you engineer hand-crafted features, or use pre-trained embeddings?

**Options:**
1. Hand-crafted features only (TF-IDF, regex, layout features)
2. Pre-trained embeddings only (Word2Vec, GloVe, FastText)
3. Pre-trained transformers (BERT embeddings)
4. Hybrid (hand-crafted + embeddings)

**MY RECOMMENDATION: Option 4 - Hybrid (hand-crafted + pre-trained embeddings)**

**Justification with your 50-200 doc constraint:**
- **Hand-crafted features:**
  - Regex patterns (e.g., amounts match `\$?\d+,?\d+\.\d{2}`)
  - Position features (field location on page)
  - Document structure (line breaks, spacing)
  - Text length, capitalization, special characters
  - These are **data-efficient** (work well with limited examples)
  
- **Pre-trained embeddings:**
  - BERT embeddings capture semantic meaning without fine-tuning
  - Requires no additional training; use pre-trained "bert-base-uncased"
  - Better than hand-crafted alone

**Implementation:**
```python
# app/ml/feature_extraction.py

import numpy as np
from transformers import AutoTokenizer, AutoModel
import re

class HybridFeatureExtractor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
    
    def hand_crafted_features(self, text, span_text):
        """Extract hand-engineered features"""
        return {
            # Pattern matching
            'is_currency': bool(re.search(r'\$?\d+,?\d+\.\d{2}', span_text)),
            'is_date': bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', span_text)),
            'is_all_caps': span_text.isupper(),
            'is_numeric': span_text.replace(',', '').replace('.', '').isdigit(),
            
            # Position
            'position_ratio': text.find(span_text) / len(text),
            
            # Length
            'span_length': len(span_text),
            'word_count': len(span_text.split()),
        }
    
    def bert_embeddings(self, text):
        """Get BERT embeddings (pre-trained, no fine-tuning)"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        outputs = self.model(**inputs)
        # Take [CLS] token (sentence-level embedding)
        cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()[0]
        return cls_embedding  # 768-dimensional vector
    
    def combine_features(self, text, span_text):
        """Combine hand-crafted + embeddings"""
        hand_crafted = self.hand_crafted_features(text, span_text)
        embeddings = self.bert_embeddings(span_text)
        
        # Flatten hand-crafted dict to array
        hand_crafted_array = np.array(list(hand_crafted.values()))
        
        # Concatenate: [hand_crafted (10-D) + BERT (768-D) = 778-D]
        combined = np.concatenate([hand_crafted_array, embeddings])
        
        return combined
```

**Why this works with 50-200 docs:**
- Hand-crafted features are **low-dimensional** (10-20 features) → trainable on small data
- BERT embeddings are **pre-trained** (no additional training needed)
- Together: you get both **interpretability** (you know what hand-crafted features mean) + **semantic power** (BERT understands context)

**Dependency:** Determines model input shape and downstream classifiers

---

## Q11: Baseline Confidence Model
**Question:** For your first ML model, what should you build to score extraction confidence?

**Options:**
1. Simple heuristic (confidence = pattern match score)
2. Random Forest classifier (trained on hand-crafted features)
3. Logistic regression (simple, interpretable)
4. XGBoost (more powerful, but prone to overfitting on 50-200 docs)

**MY RECOMMENDATION: Option 2 - Random Forest classifier**

**Justification:**
- **Data efficiency:** Random forests work well with 50-200 labeled examples
- **Robustness:** Less prone to overfitting than XGBoost on small datasets
- **Interpretable:** You can inspect feature importance (which features matter most?)
- **Speed:** Trains in seconds; no GPU needed
- **Reliable:** No hyperparameter tuning needed (defaults work)

**Architecture:**
```python
# app/ml/confidence_model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

class ExtractionConfidenceModel:
    def __init__(self, n_estimators=100, max_depth=10):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        self.feature_names = None
    
    def prepare_training_data(self, annotated_docs):
        """
        Convert annotated documents to training data.
        
        Input: List of dicts with:
            {
                "text": full document text,
                "annotations": [
                    {"text": span_text, "label": "INVOICE_ID", "start": 10, "end": 15},
                    ...
                ]
            }
        
        Output: X (features), y (labels)
        """
        feature_extractor = HybridFeatureExtractor()
        
        X = []
        y = []
        
        for doc in annotated_docs:
            text = doc['text']
            
            # Positive examples: annotated spans
            for annotation in doc['annotations']:
                span_text = annotation['text']
                features = feature_extractor.combine_features(text, span_text)
                X.append(features)
                y.append(1)  # Correct extraction
            
            # Negative examples: random non-annotated spans (hard negatives)
            # This helps model learn what NOT to extract
            for i in range(len(doc['annotations'])):
                # Extract a random span that's NOT annotated
                start = np.random.randint(0, len(text) - 50)
                end = start + np.random.randint(10, 50)
                random_span = text[start:end]
                
                # Check if this span overlaps with any annotation
                is_negative = True
                for ann in doc['annotations']:
                    if ann['start'] <= start < ann['end'] or ann['start'] < end <= ann['end']:
                        is_negative = False
                        break
                
                if is_negative:
                    features = feature_extractor.combine_features(text, random_span)
                    X.append(features)
                    y.append(0)  # Incorrect extraction
        
        return np.array(X), np.array(y)
    
    def train(self, X, y):
        """Train the confidence model"""
        # 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=['Incorrect', 'Correct']))
        
        # Feature importance
        importances = self.model.feature_importances_
        top_features = np.argsort(importances)[-5:]  # Top 5 features
        print(f"Top 5 important features: {self.feature_names[top_features]}")
        
        return self
    
    def predict_confidence(self, features):
        """
        Predict: is this extraction correct?
        Returns probability [0, 1] that extraction is correct.
        """
        probabilities = self.model.predict_proba(features.reshape(1, -1))
        confidence = probabilities[0, 1]  # Probability of class 1 (correct)
        return confidence
    
    def save(self, filepath):
        joblib.dump(self.model, filepath)
    
    def load(self, filepath):
        self.model = joblib.load(filepath)

# Usage in Month 2
annotated_docs = load_from_label_studio()  # 50-200 docs
X, y = confidence_model.prepare_training_data(annotated_docs)
confidence_model.train(X, y)
confidence_model.save('models/confidence_v1.pkl')
```

**Output:** Model that predicts: "Is this extracted field correct?" → confidence score [0, 1]

**Dependency:** This is your first ML model; everything else depends on it

---

## Q12: Named Entity Recognition (NER) Model
**Question:** How will you extract field boundaries (where exactly is the INVOICE_ID)?

**Options:**
1. Regex patterns only (no ML)
2. spaCy NER (trained on your 50-200 annotated spans)
3. BERT fine-tuning (more powerful, but requires careful tuning on small data)
4. Combination (regex + spaCy NER ensemble)

**MY RECOMMENDATION: Option 4 - Combination (regex baseline + spaCy NER ensemble)**

**Justification:**
- **Regex:** Fast, deterministic, covers 80% of cases (amounts, dates, invoice IDs follow patterns)
- **spaCy NER:** Learns context-dependent boundaries (catches edge cases)
- **Ensemble:** If regex AND spaCy agree → high confidence; if only one finds it → lower confidence
- **Data efficiency:** You don't need fine-tuning; spaCy's pre-trained model + your 50-200 examples is enough

**Implementation:**

```python
# app/ml/ner_extractor.py

import spacy
from spacy.training import Example
import re
from typing import List, Dict

class NERExtractor:
    def __init__(self):
        # Load pre-trained spaCy model
        self.nlp = spacy.load('en_core_web_sm')
        
        # Add custom pipeline
        if 'ner' not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe('ner', last=True)
        else:
            ner = self.nlp.get_pipe('ner')
        
        # Define entity labels (from your annotation schema)
        ner.add_label('INVOICE_ID')
        ner.add_label('INVOICE_DATE')
        ner.add_label('VENDOR_NAME')
        ner.add_label('TOTAL_AMOUNT')
    
    def regex_patterns(self):
        """Define regex patterns for common fields"""
        return {
            'INVOICE_ID': [
                r'INV[-\s]?\d{4,8}',
                r'Invoice\s*#?\s*\d{4,8}',
                r'Reference\s*#?\s*\d{4,8}',
            ],
            'INVOICE_DATE': [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
                r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}[,]?\s+\d{4}',
                r'\d{4}-(0?[1-9]|1[0-2])-(3[01]|[12][0-9]|0?[1-9])',
            ],
            'TOTAL_AMOUNT': [
                r'\$\s*[\d,]+\.?\d{0,2}',
                r'[\d,]+\.\d{2}\s*(?:USD|EUR|GBP)',
            ],
        }
    
    def extract_regex(self, text: str) -> List[Dict]:
        """Extract using regex (baseline)"""
        extractions = []
        patterns = self.regex_patterns()
        
        for label, regex_list in patterns.items():
            for pattern in regex_list:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    extractions.append({
                        'text': match.group(),
                        'label': label,
                        'start': match.start(),
                        'end': match.end(),
                        'method': 'regex',
                        'confidence': 0.85,  # Regex = high but not 100%
                    })
        
        return extractions
    
    def extract_spacy(self, text: str) -> List[Dict]:
        """Extract using fine-tuned spaCy NER"""
        doc = self.nlp(text)
        
        extractions = []
        for ent in doc.ents:
            if ent.label_ in ['INVOICE_ID', 'INVOICE_DATE', 'VENDOR_NAME', 'TOTAL_AMOUNT']:
                extractions.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'method': 'spacy',
                    'confidence': 0.75,  # spaCy = moderate confidence
                })
        
        return extractions
    
    def ensemble_extract(self, text: str) -> List[Dict]:
        """Combine regex + spaCy, boost confidence if both agree"""
        regex_results = self.extract_regex(text)
        spacy_results = self.extract_spacy(text)
        
        # Merge: if both find same field in same location → boost confidence
        merged = {}
        
        for ext in regex_results:
            key = (ext['label'], ext['start'], ext['end'])
            merged[key] = ext
        
        for ext in spacy_results:
            key = (ext['label'], ext['start'], ext['end'])
            if key in merged:
                # Both methods agree: boost confidence
                merged[key]['confidence'] = 0.95
                merged[key]['method'] = 'ensemble'
            else:
                merged[key] = ext
        
        return list(merged.values())
    
    def train_spacy_ner(self, annotated_docs):
        """Fine-tune spaCy on your annotated documents"""
        ner = self.nlp.get_pipe('ner')
        
        # Convert Label Studio annotations to spaCy format
        training_data = []
        for doc in annotated_docs:
            text = doc['text']
            entities = []
            
            for annotation in doc['annotations']:
                entities.append((
                    annotation['start'],
                    annotation['end'],
                    annotation['label']
                ))
            
            training_data.append((text, {'entities': entities}))
        
        # Training loop
        optimizer = self.nlp.create_optimizer()
        
        for epoch in range(10):
            losses = {}
            
            for raw_text, annotations in training_data:
                doc = self.nlp.make_doc(raw_text)
                example = Example.from_dict(doc, annotations)
                
                # Update model
                self.nlp.update(
                    [example],
                    sgd=optimizer,
                    drop=0.5,  # Dropout to prevent overfitting
                    losses=losses
                )
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch}, Loss: {losses.get('ner', 0):.4f}")
        
        return self
    
    def save(self, filepath):
        self.nlp.to_disk(filepath)
    
    def load(self, filepath):
        self.nlp = spacy.load(filepath)

# Usage in Month 2
annotated_docs = load_from_label_studio()
extractor = NERExtractor()
extractor.train_spacy_ner(annotated_docs)
extractor.save('models/ner_spacy_v1')

# Inference: ensemble approach
results = extractor.ensemble_extract("Invoice #12345 dated Jan 15, 2024 for $5000")
# Output:
# [
#   {'text': 'INV-12345', 'label': 'INVOICE_ID', 'confidence': 0.95, 'method': 'ensemble'},
#   {'text': 'Jan 15, 2024', 'label': 'INVOICE_DATE', 'confidence': 0.85, 'method': 'regex'},
#   {'text': '$5000', 'label': 'TOTAL_AMOUNT', 'confidence': 0.95, 'method': 'ensemble'},
# ]
```

**Why ensemble?**
- Regex catches **obvious patterns** (amounts, dates, invoice numbers)
- spaCy catches **context-dependent** cases (vendor names, complex structures)
- When both agree → very high confidence
- When only one finds it → needs human review

**Dependency:** This determines your extraction accuracy; everything downstream depends on it

---

## Q13: Document Classification
**Question:** Before extraction, should you classify document type (invoice vs. receipt vs. contract)?

**Options:**
1. Skip classification (extract same fields from all docs)
2. Zero-shot classification (BART, no fine-tuning)
3. Fine-tune classifier on your 50-200 docs
4. Use as preprocessing only (don't branch logic on it yet)

**MY RECOMMENDATION: Option 2 - Zero-shot BART classification (use for preprocessing, not branching logic yet)**

**Justification:**
- **No fine-tuning needed:** BART understands "invoice vs. receipt" without training
- **Fast inference:** Just a few forward passes
- **Informational:** Helpful for dashboards ("80% invoices, 20% receipts processed")
- **Future flexibility:** If you need doc-type-specific extraction later, foundation is there
- **Don't branch logic on it yet:** Extract same fields from all docs; use classification only for reporting

**Implementation:**

```python
# app/ml/doc_classifier.py

from transformers import pipeline

class DocumentClassifier:
    def __init__(self):
        # Zero-shot: no fine-tuning needed, works out of the box
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        # Document types to classify
        self.doc_types = [
            "invoice",
            "receipt",
            "purchase order",
            "contract",
            "report",
            "letter",
            "other"
        ]
    
    def classify(self, text: str, threshold=0.5):
        """Classify document type"""
        # Use first 512 chars (BART's limit is 1024, but 512 usually enough)
        text_sample = text[:512]
        
        result = self.classifier(
            text_sample,
            self.doc_types,
            multi_class=False  # Each doc is exactly one type
        )
        
        top_label = result['labels'][0]
        confidence = result['scores'][0]
        
        return {
            'doc_type': top_label,
            'confidence': confidence,
            'accept': confidence > threshold,
        }

# Usage in Month 2
classifier = DocumentClassifier()
doc_classification = classifier.classify(text)
print(f"Document type: {doc_classification['doc_type']} "
      f"(confidence: {doc_classification['confidence']:.2%})")

# Example output:
# Document type: invoice (confidence: 92.34%)
```

**Why not branch logic on it yet?**
- 4-month internship is tight; focus on single extraction schema first
- If classification is wrong (misidentifies invoice as receipt), logic would break
- In production (Month 4+), IF classification is reliable, then add doc-type-specific extraction

**Dependency:** Informational only for Month 2; becomes optional in Month 4

---

## Q14: Extraction Pipeline Orchestration
**Question:** How will you combine the models (NER + confidence + classification) into a single system?

**Options:**
1. Sequential pipeline (NER → confidence scoring → output)
2. Parallel pipeline (run all models, merge results)
3. Conditional pipeline (classify → run doc-specific extraction)
4. Staged pipeline (coarse extraction → confidence filter → detailed extraction)

**MY RECOMMENDATION: Option 1 - Sequential pipeline (simplest, most reliable)**

**Justification:**
- You have 4 months; optimize for correctness, not complexity
- Sequential = each stage is independent, debuggable
- Parallel would be faster but harder to debug if one model fails
- Conditional/staged requires classification to be highly reliable (it's not yet)

**Architecture:**

```python
# app/ml/extraction_pipeline.py

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class ExtractionPipeline:
    def __init__(self, ner_model, confidence_model, classifier):
        self.ner = ner_model
        self.confidence = confidence_model
        self.classifier = classifier
        self.feature_extractor = HybridFeatureExtractor()
    
    def extract(self, document_text: str, document_id: str = None) -> Dict:
        """
        End-to-end extraction pipeline.
        
        Stages:
        1. Classify document type (informational)
        2. Extract fields (NER)
        3. Score confidence for each extraction
        4. Filter by confidence threshold
        5. Return structured result
        """
        
        logger.info(f"Processing document {document_id}")
        
        # Stage 1: Document classification
        classification = self.classifier.classify(document_text)
        logger.info(f"Classified as: {classification['doc_type']}")
        
        # Stage 2: Field extraction (NER ensemble)
        raw_extractions = self.ner.ensemble_extract(document_text)
        logger.info(f"Raw extractions: {len(raw_extractions)} fields found")
        
        # Stage 3: Score confidence
        scored_extractions = []
        for extraction in raw_extractions:
            # Get features for this extraction
            features = self.feature_extractor.combine_features(
                document_text,
                extraction['text']
            )
            
            # Predict confidence
            confidence = self.confidence.predict_confidence(features)
            
            # Combine NER confidence + model confidence
            combined_confidence = (
                0.5 * extraction['confidence'] +  # NER confidence
                0.5 * confidence                   # ML model confidence
            )
            
            extraction['ml_confidence'] = confidence
            extraction['final_confidence'] = combined_confidence
            
            scored_extractions.append(extraction)
        
        logger.info(f"Confidence scoring complete")
        
        # Stage 4: Organize by field type
        result = {
            'document_id': document_id,
            'doc_type': classification['doc_type'],
            'doc_type_confidence': classification['confidence'],
            'fields': {},
        }
        
        for extraction in scored_extractions:
            label = extraction['label']
            
            if label not in result['fields']:
                result['fields'][label] = []
            
            result['fields'][label].append({
                'value': extraction['text'],
                'confidence': extraction['final_confidence'],
                'position': (extraction['start'], extraction['end']),
                'extraction_method': extraction['method'],
            })
        
        logger.info(f"Pipeline complete. Extracted {sum(len(v) for v in result['fields'].values())} fields")
        
        return result

# Usage
pipeline = ExtractionPipeline(ner_model, confidence_model, classifier)
result = pipeline.extract(document_text, document_id='doc_001')

# Output:
# {
#   'document_id': 'doc_001',
#   'doc_type': 'invoice',
#   'doc_type_confidence': 0.92,
#   'fields': {
#       'INVOICE_ID': [
#           {'value': 'INV-12345', 'confidence': 0.95, 'position': (100, 108), ...}
#       ],
#       'INVOICE_DATE': [
#           {'value': 'Jan 15, 2024', 'confidence': 0.85, 'position': (120, 132), ...}
#       ],
#       'TOTAL_AMOUNT': [
#           {'value': '$5000', 'confidence': 0.92, 'position': (500, 505), ...}
#       ],
#   }
# }
```

**Key design principle:** Each field can have multiple extractions. If the document mentions "Total: $5000, Previously: $4000", you extract both amounts. Downstream logic decides which one is "the" total.

**Dependency:** This is the core of your system; everything else integrates into it

---

## MONTH 2 SUMMARY (END OF WEEK 8)

**Deliverables:**
1. ✅ Random Forest confidence model (trained, saved)
2. ✅ spaCy NER model (trained, saved)
3. ✅ BART document classifier (zero-shot, no training needed)
4. ✅ Extraction pipeline (orchestrates all three models)
5. ✅ Test results: accuracy on held-out 10 annotated docs

**Example Month 2 Accuracy Metrics:**
```
INVOICE_ID:     Precision: 92%, Recall: 88%, F1: 90%
INVOICE_DATE:   Precision: 85%, Recall: 82%, F1: 83%
VENDOR_NAME:    Precision: 78%, Recall: 72%, F1: 75%
TOTAL_AMOUNT:   Precision: 90%, Recall: 89%, F1: 89%

Overall F1: 86.75%  (acceptable for Month 2 MVP)
```

---

# PHASE 3: MONTH 3 - KPI ENGINE & REPORTING (Weeks 9-12)

## Q15: Confidence Thresholding Strategy
**Question:** How will you decide which extractions to auto-approve, which need human review, which to reject?

**Options:**
1. Single threshold (confidence > 0.85 → accept)
2. Per-field thresholds (AMOUNT needs 0.90; DATE needs 0.80)
3. Dynamic thresholds (based on field importance, document type)
4. Confidence ranges (0.9+ auto-accept; 0.7-0.9 review; <0.7 reject)

**MY RECOMMENDATION: Option 4 - Confidence ranges with per-field tuning**

**Justification:**
- **Ranges are realistic:** Not all extractions are 100% certain
- **Per-field tuning:** AMOUNT is more critical (financial impact); DATE less so
- **Reviewable window:** 0.7-0.9 range = human review (not too many, not too few)
- **Measurable:** You can track "what % of auto-acceptances were actually correct?" (precision)

**Implementation:**

```python
# app/ml/decision_engine.py

class ConfidenceDecisionEngine:
    # Per-field thresholds (tuned in Month 3 based on test results)
    THRESHOLDS = {
        'INVOICE_ID': {'auto_accept': 0.92, 'human_review': 0.75},
        'INVOICE_DATE': {'auto_accept': 0.88, 'human_review': 0.70},
        'VENDOR_NAME': {'auto_accept': 0.85, 'human_review': 0.65},
        'TOTAL_AMOUNT': {'auto_accept': 0.90, 'human_review': 0.75},
    }
    
    def decide(self, field_label: str, confidence: float, anomaly_detected: bool = False) -> Dict:
        """
        Decide action for a single field extraction.
        
        Returns: {'action': 'auto_accept' | 'human_review' | 'reject', 'reason': str}
        """
        
        thresholds = self.THRESHOLDS.get(field_label, {
            'auto_accept': 0.85,
            'human_review': 0.70
        })
        
        # If anomalous AND low confidence: always review
        if anomaly_detected and confidence < 0.80:
            return {
                'action': 'human_review',
                'reason': 'Anomaly detected + confidence < 0.80',
                'confidence': confidence,
            }
        
        # Normal logic
        if confidence >= thresholds['auto_accept']:
            return {
                'action': 'auto_accept',
                'reason': f'Confidence {confidence:.2%} >= threshold {thresholds["auto_accept"]:.0%}',
                'confidence': confidence,
            }
        
        elif confidence >= thresholds['human_review']:
            return {
                'action': 'human_review',
                'reason': f'Confidence {confidence:.2%} in review range [{thresholds["human_review"]:.0%}, {thresholds["auto_accept"]:.0%})',
                'confidence': confidence,
            }
        
        else:
            return {
                'action': 'reject',
                'reason': f'Confidence {confidence:.2%} < threshold {thresholds["human_review"]:.0%}',
                'confidence': confidence,
            }

# Usage in Month 3
decision_engine = ConfidenceDecisionEngine()

extraction = {
    'field': 'TOTAL_AMOUNT',
    'value': '$5000',
    'confidence': 0.88,
}

decision = decision_engine.decide(
    extraction['field'],
    extraction['confidence'],
    anomaly_detected=False
)

print(f"Action: {decision['action']}")  # Output: "human_review"
print(f"Reason: {decision['reason']}")  # "Confidence 88% in review range [75%, 90%)"
```

**In practice (Month 3+):**
```
Human extracts 100 random documents:
- 35 auto-accepted (confidence >= 0.90)
- 45 human-reviewed (confidence 0.70-0.90)
- 20 rejected (confidence < 0.70)

Of the 35 auto-accepted:
- 34 are correct (precision = 97%)
- 1 is wrong (human can catch in spot-check)

Goal: >= 95% auto-accept precision
```

**Dependency:** Determines data flow in production (which extractions go where)

---

## Q16: Anomaly Detection
**Question:** Should you flag unusual extractions that seem "off"?

**Options:**
1. Skip anomaly detection (just use confidence thresholds)
2. Isolation Forest (unsupervised, works with 50-200 examples)
3. Manual rules (if TOTAL_AMOUNT > 999,999 → flag)
4. Both (rules + Isolation Forest)

**MY RECOMMENDATION: Option 4 - Both (rules + Isolation Forest)**

**Justification:**
- **Manual rules:** Business logic (invoice shouldn't be $0, or $10M+)
- **Isolation Forest:** Catches statistical outliers you didn't anticipate
- **Together:** Catches 90% of anomalies (human reviewers catch the rest)

**Implementation:**

```python
# app/ml/anomaly_detection.py

from sklearn.ensemble import IsolationForest
import numpy as np

class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.trained = False
    
    def train(self, extraction_features):
        """
        Train on features from correct extractions.
        
        Features example:
        [
            [amount_value, num_pages, vendor_name_length, date_in_past_days],
            [5000, 2, 15, 45],
            [3500, 1, 12, 30],
            ...
        ]
        """
        self.model.fit(extraction_features)
        self.trained = True
    
    def is_anomaly(self, features) -> Dict:
        """
        Check if extraction is anomalous.
        
        Returns: {'is_anomaly': bool, 'score': float, 'confidence': float}
        """
        if not self.trained:
            return {'is_anomaly': False, 'score': 0.0, 'confidence': 0.0}
        
        prediction = self.model.predict(features.reshape(1, -1))
        score = self.model.score_samples(features.reshape(1, -1))[0]
        
        return {
            'is_anomaly': prediction[0] == -1,
            'score': score,  # More negative = more anomalous
            'confidence': abs(score),
        }

class BusinessRuleAnomalyDetector:
    """Detect anomalies based on domain knowledge"""
    
    @staticmethod
    def check_amount(amount: float) -> Dict:
        """Check if amount is reasonable"""
        # Example rules (adjust for your domain)
        if amount <= 0:
            return {'is_anomaly': True, 'reason': 'Amount is zero or negative'}
        if amount > 1_000_000:
            return {'is_anomaly': True, 'reason': 'Amount > $1M (possible OCR error)'}
        return {'is_anomaly': False, 'reason': 'Amount is reasonable'}
    
    @staticmethod
    def check_date(date_str: str) -> Dict:
        """Check if date is reasonable"""
        from datetime import datetime, timedelta
        
        # Parse date (simplified)
        try:
            # This is pseudocode; use dateutil.parser in reality
            date = parse_date(date_str)
        except:
            return {'is_anomaly': True, 'reason': 'Date format invalid'}
        
        # Check if date is in future or too far in past
        today = datetime.now()
        if date > today + timedelta(days=30):
            return {'is_anomaly': True, 'reason': 'Date is > 30 days in future'}
        if date < today - timedelta(days=365*10):
            return {'is_anomaly': True, 'reason': 'Date is > 10 years in past'}
        
        return {'is_anomaly': False, 'reason': 'Date is reasonable'}

# Usage in extraction pipeline
anomaly_detector = AnomalyDetector()
anomaly_detector.train(training_features)  # Train on correct extractions

# For each extraction
extraction = {'field': 'TOTAL_AMOUNT', 'value': 5000.0}

# Rule-based check
rule_result = BusinessRuleAnomalyDetector.check_amount(extraction['value'])

# Statistical check
features = extract_features(extraction)  # Convert to numeric features
statistical_result = anomaly_detector.is_anomaly(features)

# Combine
is_anomalous = rule_result['is_anomaly'] or statistical_result['is_anomaly']
reason = rule_result.get('reason') or statistical_result.get('score')

if is_anomalous:
    # Flag for human review, don't auto-accept
    print(f"Anomaly detected: {reason}")
```

**Dependency:** Improves confidence thresholding (anomalous + low-confidence → always review)

---

## Q17: KPI Engine Definition
**Question:** What KPIs will you compute for your dashboards?

**Options:**
1. Basic operational metrics (docs processed, extraction rate)
2. Quality metrics (accuracy, precision, recall)
3. Business metrics (cost saved, processing time)
4. All of the above

**MY RECOMMENDATION: Option 4 - All of the above, organized by dashboard view**

**Justification:**
- **Operational:** Shows system health (is it working?)
- **Quality:** Shows extraction accuracy (is it accurate?)
- **Business:** Shows ROI to stakeholders (is it valuable?)

**Implementation:**

```python
# app/kpi/engine.py

from datetime import datetime, timedelta
from sqlalchemy import func

class KPIEngine:
    def __init__(self, db):
        self.db = db
    
    def calculate_kpis(self, start_date: datetime, end_date: datetime) -> Dict:
        """Calculate all KPIs for a date range"""
        
        # Query documents processed
        docs_processed = self.db.query(Document).filter(
            Document.processed_at.between(start_date, end_date)
        ).count()
        
        # Query extractions
        extractions = self.db.query(Extraction).filter(
            Extraction.created_at.between(start_date, end_date)
        ).all()
        
        # Query human corrections
        corrections = self.db.query(HumanCorrection).filter(
            HumanCorrection.created_at.between(start_date, end_date)
        ).all()
        
        # OPERATIONAL KPIS
        operational = {
            'total_documents': docs_processed,
            'total_fields_extracted': len(extractions),
            'avg_fields_per_doc': len(extractions) / max(docs_processed, 1),
            'processing_rate': docs_processed / ((end_date - start_date).days or 1),  # docs/day
        }
        
        # QUALITY KPIS
        auto_accepted = len([e for e in extractions if e.decision == 'auto_accept'])
        human_reviewed = len([e for e in extractions if e.decision == 'human_review'])
        rejected = len([e for e in extractions if e.decision == 'reject'])
        
        corrections_needed = len(corrections)
        
        quality = {
            'auto_accept_rate': auto_accepted / len(extractions) if extractions else 0,
            'human_review_rate': human_reviewed / len(extractions) if extractions else 0,
            'reject_rate': rejected / len(extractions) if extractions else 0,
            'accuracy_on_reviewed': (auto_accepted + (human_reviewed - corrections_needed)) / len(extractions) if extractions else 0,
            'correction_rate': corrections_needed / human_reviewed if human_reviewed > 0 else 0,
        }
        
        # BUSINESS KPIS
        # Assume human review takes 2 minutes per field, auto-accept saves time
        manual_review_time_minutes = human_reviewed * 2
        auto_accept_time_saved_minutes = auto_accepted * 2
        
        # Assume each extraction is worth ~$5 in business value (time saved, accuracy)
        extraction_value = len(extractions) * 5
        
        business = {
            'manual_review_time_hours': manual_review_time_minutes / 60,
            'time_saved_hours': auto_accept_time_saved_minutes / 60,
            'estimated_value_usd': extraction_value,
            'cost_per_extraction': extraction_value / max(len(extractions), 1),
        }
        
        return {
            'period': f"{start_date.date()} to {end_date.date()}",
            'operational': operational,
            'quality': quality,
            'business': business,
        }

# Usage in Month 3
kpi_engine = KPIEngine(db)
kpis = kpi_engine.calculate_kpis(
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now()
)

print(f"Documents processed: {kpis['operational']['total_documents']}")
print(f"Accuracy: {kpis['quality']['accuracy_on_reviewed']:.1%}")
print(f"Time saved: {kpis['business']['time_saved_hours']:.1f} hours")
```

**Dashboard views (Month 3):**
1. **Operations:** Documents processed, extraction rate, rejection rate
2. **Quality:** Auto-accept rate, accuracy, correction rate
3. **Business:** Time saved, estimated value, cost per extraction
4. **Trends:** All metrics over time (weekly, monthly)

**Dependency:** Drives dashboard design and stakeholder visibility

---

## MONTH 3 SUMMARY (END OF WEEK 12)

**Deliverables:**
1. ✅ Confidence decision engine (per-field thresholds configured)
2. ✅ Anomaly detection (rules + Isolation Forest)
3. ✅ KPI engine (operational + quality + business metrics)
4. ✅ Extraction pipeline with confidence filtering
5. ✅ Test results: 100 documents processed, categorized by decision

**Example Month 3 Results:**
```
100 documents processed:
- 65 auto-accepted (confidence >= 0.90)
- 28 human-reviewed (confidence 0.70-0.90)
- 7 rejected (confidence < 0.70)

Auto-accept precision: 98% (only 1 out of 65 needs review)
Human-review correction rate: 12% (3 out of 28 had errors)
Overall extraction accuracy: 94%

Time metrics:
- Manual review: 56 minutes (28 fields * 2 min)
- Time saved: 130 minutes (65 fields * 2 min)
- Cost per extraction: $4.50
```

---

# PHASE 4: MONTH 4 - TESTING, FEEDBACK LOOP & PRODUCTION (Weeks 13-16)

## Q18: Evaluation Metrics Framework
**Question:** How will you measure and report extraction quality?

**Options:**
1. Simple accuracy (% correct)
2. Per-field metrics (precision/recall/F1 for each field)
3. Confusion matrix (what types of errors?)
4. End-to-end metrics (document-level correctness)

**MY RECOMMENDATION: Option 2 + 3 - Per-field + Confusion matrix**

**Justification:**
- **Per-field:** Shows which fields work well, which need improvement
- **Confusion matrix:** Shows error patterns (e.g., "DATE confused with INVOICE_ID")
- **Actionable:** You can see "VENDOR_NAME needs retraining" vs. "system is generally good"

**Implementation:**

```python
# app/evaluation/metrics.py

from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pandas as pd

class EvaluationMetrics:
    @staticmethod
    def evaluate_extraction(ground_truth, predictions):
        """
        Evaluate extraction performance.
        
        ground_truth: List of dicts [{'field': 'AMOUNT', 'value': '5000'}, ...]
        predictions: List of dicts (same format)
        """
        
        # Extract labels
        true_labels = [item['field'] for item in ground_truth]
        pred_labels = [item['field'] for item in predictions]
        
        # Per-field metrics
        report = classification_report(
            true_labels, pred_labels,
            target_names=['INVOICE_ID', 'INVOICE_DATE', 'VENDOR_NAME', 'TOTAL_AMOUNT'],
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(true_labels, pred_labels)
        
        return {
            'per_field_metrics': report,
            'confusion_matrix': conf_matrix,
            'overall_f1': f1_score(true_labels, pred_labels, average='weighted'),
        }
    
    @staticmethod
    def print_evaluation_report(metrics):
        """Pretty-print evaluation results"""
        print("\n=== PER-FIELD METRICS ===")
        df = pd.DataFrame(metrics['per_field_metrics']).transpose()
        print(df)
        
        print("\n=== CONFUSION MATRIX ===")
        print("(rows=true, cols=predicted)")
        print(metrics['confusion_matrix'])
        
        print(f"\nOverall F1 Score: {metrics['overall_f1']:.3f}")

# Usage in Month 4
test_docs = load_test_documents()  # 50 held-out documents

predictions = []
ground_truth = []

for doc in test_docs:
    result = extraction_pipeline.extract(doc['text'])
    predictions.extend(result['fields'])
    ground_truth.extend(doc['annotations'])

metrics = EvaluationMetrics.evaluate_extraction(ground_truth, predictions)
EvaluationMetrics.print_evaluation_report(metrics)

# Output:
# === PER-FIELD METRICS ===
#              precision    recall  f1-score   support
# INVOICE_ID        0.95      0.92      0.93        50
# INVOICE_DATE      0.88      0.85      0.86        50
# VENDOR_NAME       0.82      0.80      0.81        50
# TOTAL_AMOUNT      0.91      0.90      0.90        50
# 
# Overall F1 Score: 0.875
```

**Dependency:** Determines if system meets production readiness criteria

---

## Q19: API Endpoints Design
**Question:** How will external applications (dashboards, downstream systems) call your extraction pipeline?

**Options:**
1. REST API (simple HTTP requests)
2. Async API (Celery tasks, return results later)
3. Streaming API (process documents in batch)
4. All three (different endpoints for different use cases)

**MY RECOMMENDATION: Option 1 + 2 - REST for single docs, Async for batches**

**Justification:**
- **REST (sync):** "Extract this one document now" (fast feedback loop)
- **Async (queue):** "Extract 100 documents, process in background" (batch jobs)
- **Together:** Covers both interactive and batch workflows

**Implementation:**

```python
# app/api/extraction_endpoints.py

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
import uuid
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

# In-memory task tracker (replace with Redis in production)
tasks = {}

@app.post("/api/extract")
async def extract_sync(file: UploadFile = File(...)):
    """
    Synchronous extraction: process document now, return results.
    
    Use for: Single document, interactive workflows, testing
    Response time: 2-10 seconds (depending on document size)
    """
    try:
        # Read file
        contents = await file.read()
        text = extract_text_from_file(contents, file.filename)
        
        # Extract
        result = extraction_pipeline.extract(text, document_id=file.filename)
        
        # Log extraction
        db.save_extraction(result)
        
        return JSONResponse(status_code=200, content=result)
    
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        return JSONResponse(status_code=500, content={'error': str(e)})

@app.post("/api/extract-batch")
async def extract_async(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Asynchronous batch extraction: queue documents, return task ID.
    
    Use for: Multiple documents, batch jobs, non-interactive workflows
    Response time: Immediate (returns task ID), processing in background
    """
    
    # Create task ID
    task_id = str(uuid.uuid4())
    
    # Queue task
    background_tasks.add_task(process_batch, task_id, files)
    
    # Immediately return task ID
    tasks[task_id] = {'status': 'queued', 'file_count': len(files), 'progress': 0}
    
    return {
        'task_id': task_id,
        'status': 'queued',
        'message': f"Batch processing {len(files)} documents. Check status at /api/task/{task_id}"
    }

@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    """Check status of async extraction task"""
    
    if task_id not in tasks:
        return JSONResponse(status_code=404, content={'error': 'Task not found'})
    
    task = tasks[task_id]
    
    return {
        'task_id': task_id,
        'status': task['status'],  # 'queued', 'processing', 'completed'
        'progress': task.get('progress', 0),
        'file_count': task.get('file_count', 0),
        'result_url': f"/api/task/{task_id}/results" if task['status'] == 'completed' else None,
    }

@app.get("/api/task/{task_id}/results")
async def get_task_results(task_id: str):
    """Retrieve results of completed async task"""
    
    if task_id not in tasks or tasks[task_id]['status'] != 'completed':
        return JSONResponse(status_code=400, content={'error': 'Task not completed'})
    
    results = tasks[task_id]['results']
    
    return {
        'task_id': task_id,
        'total_documents': len(results),
        'results': results,
    }

async def process_batch(task_id: str, files: List[UploadFile]):
    """Background task: process multiple documents"""
    
    results = []
    tasks[task_id]['status'] = 'processing'
    
    for i, file in enumerate(files):
        try:
            contents = await file.read()
            text = extract_text_from_file(contents, file.filename)
            result = extraction_pipeline.extract(text, document_id=file.filename)
            results.append(result)
            
            # Update progress
            tasks[task_id]['progress'] = int((i + 1) / len(files) * 100)
        
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {str(e)}")
            results.append({
                'filename': file.filename,
                'error': str(e),
            })
    
    # Mark complete
    tasks[task_id]['status'] = 'completed'
    tasks[task_id]['results'] = results

# Health check
@app.get("/health")
async def health():
    return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}
```

**Usage examples:**

```bash
# Sync: Extract single document
curl -X POST -F "file=@invoice.pdf" http://localhost:8000/api/extract

# Async: Queue batch
curl -X POST -F "files=@invoice1.pdf" -F "files=@invoice2.pdf" http://localhost:8000/api/extract-batch

# Check status
curl http://localhost:8000/api/task/abc-123-def/

# Get results
curl http://localhost:8000/api/task/abc-123-def/results
```

**Dependency:** Determines how downstream systems (dashboards, other apps) will call your system

---

## Q20: Dashboard & KPI Visualization
**Question:** How will you visualize KPIs and extraction results?

**Options:**
1. Streamlit (Python-based, quick to build)
2. React (more powerful, slower to build)
3. Grafana (monitoring-focused)
4. Combination (Streamlit for internal, React for public)

**MY RECOMMENDATION: Option 1 - Streamlit (fast to build, sufficient for Month 4)**

**Justification:**
- 4 months is tight; Streamlit ships in days, not weeks
- Metrics/dashboards are internal (engineers, stakeholders); don't need polished UI
- Later can migrate to React if needed
- Python ecosystem (pandas, plotly) integrates seamlessly

**Implementation:**

```python
# app/dashboard/streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from app.kpi.engine import KPIEngine
from sqlalchemy import create_engine

st.set_page_config(page_title="Extraction Dashboard", layout="wide")

# Database connection
db = create_engine('postgresql://...').connect()
kpi_engine = KPIEngine(db)

st.title("📊 AI Extraction System Dashboard")

# Sidebar: date range selector
st.sidebar.title("Filters")
days_back = st.sidebar.slider("Days back", 1, 90, 30)
start_date = datetime.now() - timedelta(days=days_back)
end_date = datetime.now()

# Calculate KPIs
kpis = kpi_engine.calculate_kpis(start_date, end_date)

# OPERATIONAL SECTION
st.header("📈 Operational Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Documents Processed",
        f"{kpis['operational']['total_documents']}",
        f"+{kpis['operational']['processing_rate']:.1f}/day"
    )

with col2:
    st.metric(
        "Fields Extracted",
        f"{kpis['operational']['total_fields_extracted']}",
        f"{kpis['operational']['avg_fields_per_doc']:.1f}/doc"
    )

with col3:
    st.metric(
        "Auto-Accept Rate",
        f"{kpis['quality']['auto_accept_rate']:.1%}"
    )

with col4:
    st.metric(
        "Overall Accuracy",
        f"{kpis['quality']['accuracy_on_reviewed']:.1%}"
    )

# QUALITY SECTION
st.header("✅ Quality Metrics")

col1, col2 = st.columns(2)

with col1:
    # Decision breakdown pie chart
    decisions = [
        kpis['quality']['auto_accept_rate'] * 100,
        kpis['quality']['human_review_rate'] * 100,
        kpis['quality']['reject_rate'] * 100,
    ]
    
    fig = go.Figure(data=[go.Pie(
        labels=['Auto-Accept', 'Human Review', 'Reject'],
        values=decisions,
        hole=0.3,
    )])
    fig.update_layout(title="Decision Distribution")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Accuracy by field type
    field_accuracy = {
        'INVOICE_ID': 0.95,
        'INVOICE_DATE': 0.88,
        'VENDOR_NAME': 0.82,
        'TOTAL_AMOUNT': 0.91,
    }
    
    fig = go.Figure(data=[go.Bar(
        x=list(field_accuracy.keys()),
        y=list(field_accuracy.values()),
        marker_color=['green' if v >= 0.90 else 'orange' for v in field_accuracy.values()],
    )])
    fig.update_layout(
        title="Accuracy by Field Type",
        yaxis_range=[0, 1],
        yaxis_tickformat='.0%',
    )
    st.plotly_chart(fig, use_container_width=True)

# BUSINESS SECTION
st.header("💰 Business Metrics")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Time Saved",
        f"{kpis['business']['time_saved_hours']:.1f} hours",
        f"≈ ${kpis['business']['time_saved_hours'] * 50:.0f} value"  # Assume $50/hour
    )

with col2:
    st.metric(
        "Estimated Value",
        f"${kpis['business']['estimated_value_usd']:.0f}",
        f"${kpis['business']['cost_per_extraction']:.2f}/extraction"
    )

# DETAILED RESULTS
st.header("📋 Recent Extractions")

# Query recent extractions
recent_extractions = pd.read_sql("""
    SELECT document_id, doc_type, total_fields, avg_confidence, decision, created_at
    FROM extractions
    WHERE created_at BETWEEN %s AND %s
    ORDER BY created_at DESC
    LIMIT 50
""", db, params=[start_date, end_date])

st.dataframe(
    recent_extractions,
    use_container_width=True,
    height=400,
)

# Download results
csv = recent_extractions.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="extractions.csv",
    mime="text/csv",
)

# Refresh
st.info("📅 Dashboard auto-refreshes every 60 seconds")
```

**Run:**
```bash
streamlit run app/dashboard/streamlit_app.py
# Opens at http://localhost:8501
```

**Dependency:** Gives stakeholders visibility; drives decisions on threshold tuning

---

## Q21: Human Feedback Collection & Correction Workflow
**Question:** How will you collect human corrections to improve models over time?

**Options:**
1. Manual: humans tell you what's wrong via email/spreadsheet
2. UI-based: build web form for corrections
3. Implicit: compare human-reviewed extractions vs. original
4. Hybrid: auto-capture corrections + option to add notes

**MY RECOMMENDATION: Option 4 - Hybrid (auto-capture + notes)**

**Justification:**
- **Auto-capture:** When human changes a field, you record old vs. new (no extra work)
- **Notes:** Optional explanation ("OCR was bad" vs. "document was unusual")
- **Lightweight:** Humans just do their job; corrections flow automatically
- **Data for retraining:** Month 4 → Month 5, you have feedback to retrain

**Implementation:**

```python
# app/correction/correction_handler.py

from datetime import datetime
from typing import Dict, List

class CorrectionHandler:
    def __init__(self, db):
        self.db = db
    
    def record_correction(
        self,
        document_id: str,
        original_extraction: Dict,
        corrected_extraction: Dict,
        field_label: str,
        corrector_notes: str = None
    ):
        """
        Record a human correction.
        
        original_extraction: {'value': '$5000', 'confidence': 0.88}
        corrected_extraction: {'value': '$5500', 'confidence': None}
        """
        
        correction = {
            'document_id': document_id,
            'field_label': field_label,
            'original_value': original_extraction['value'],
            'original_confidence': original_extraction.get('confidence'),
            'corrected_value': corrected_extraction['value'],
            'corrector_notes': corrector_notes,
            'correction_timestamp': datetime.now(),
            'was_correct': original_extraction['value'] == corrected_extraction['value'],
        }
        
        # Save to database
        self.db.save_correction(correction)
        
        return correction
    
    def get_corrections_for_retraining(self, days_back=30):
        """
        Get all corrections from last N days for retraining.
        
        Returns: Documents with corrections and their corrected values
        """
        
        corrections = self.db.query(f"""
            SELECT document_id, field_label, original_value, corrected_value
            FROM corrections
            WHERE correction_timestamp > NOW() - INTERVAL '{days_back} days'
        """)
        
        return corrections

# Web UI for corrections (FastAPI + HTML form)
@app.post("/api/correction")
async def submit_correction(
    document_id: str,
    field_label: str,
    corrected_value: str,
    corrector_notes: str = None
):
    """API endpoint for submitting corrections"""
    
    # Get original extraction
    original = db.get_extraction(document_id, field_label)
    
    # Record correction
    handler = CorrectionHandler(db)
    handler.record_correction(
        document_id,
        original,
        {'value': corrected_value},
        field_label,
        corrector_notes
    )
    
    return {'status': 'Correction recorded', 'document_id': document_id}

# Streamlit correction UI
def correction_interface():
    st.header("✏️ Correct Extraction Results")
    
    # Load recent extractions pending review
    pending = st.session_state.db.query("""
        SELECT document_id, doc_type, fields
        FROM extractions
        WHERE decision = 'human_review'
        AND corrected_at IS NULL
        ORDER BY created_at DESC
        LIMIT 10
    """)
    
    for extraction in pending:
        st.subheader(f"Document: {extraction['document_id']}")
        
        # Display original extraction
        st.write("**Original Extraction:**")
        for field_label, values in extraction['fields'].items():
            st.write(f"- {field_label}: {values}")
        
        # Allow corrections
        with st.form(key=f"correction_{extraction['document_id']}"):
            for field_label in extraction['fields']:
                corrected_value = st.text_input(
                    f"Correct {field_label}",
                    value=extraction['fields'][field_label]
                )
                
                notes = st.text_area(
                    f"Notes for {field_label}",
                    placeholder="Why did you correct this?"
                )
            
            if st.form_submit_button("Save Corrections"):
                # Record corrections
                handler = CorrectionHandler(st.session_state.db)
                handler.record_correction(
                    extraction['document_id'],
                    extraction['fields'],
                    {field_label: corrected_value},
                    field_label,
                    notes
                )
                st.success("Correction saved!")
```

**Data flow:**
```
Human reviews extraction → Changes value → Correction auto-recorded
→ Month 4 ends → Collect all corrections → Retrain models Month 5+
```

**Dependency:** Enables continuous improvement (Q22-24)

---

## Q22: Monthly Retraining Process
**Question:** How will you retrain models with feedback corrections?

**Options:**
1. Fully retrain from scratch (takes time, risky)
2. Fine-tune on corrections only (fast, incremental)
3. Hybrid (fully retrain quarterly, fine-tune monthly)
4. Skip retraining (use same model forever)

**MY RECOMMENDATION: Option 2 - Fine-tune on corrections (monthly, fast)**

**Justification:**
- **Speed:** Fine-tuning spaCy NER on 50 corrections takes 5 minutes
- **Safety:** If new model is worse, roll back to previous version
- **Incremental:** Model improves gradually vs. risky full retrains
- **Quarterly full retrain:** Fall back to Option 3 if drift is detected

**Implementation:**

```python
# app/ml/retraining.py

import schedule
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class MonthlyRetrainingPipeline:
    def __init__(self, db, models_dir='models/'):
        self.db = db
        self.models_dir = models_dir
    
    def get_monthly_corrections(self):
        """Get corrections from last 30 days"""
        
        corrections = self.db.query("""
            SELECT document_id, field_label, original_value, corrected_value
            FROM corrections
            WHERE correction_timestamp > NOW() - INTERVAL '30 days'
        """)
        
        logger.info(f"Found {len(corrections)} corrections from last month")
        return corrections
    
    def prepare_retraining_data(self, corrections):
        """Convert corrections to training format"""
        
        training_data = []
        
        for correction in corrections:
            # Load original document
            doc = self.db.get_document(correction['document_id'])
            text = doc['text']
            
            # Create training example: (text, entity positions)
            # In spaCy format: {"entities": [(start, end, label), ...]}
            
            # Find position of corrected value in text
            start = text.find(correction['corrected_value'])
            end = start + len(correction['corrected_value'])
            
            if start >= 0:  # Found
                training_data.append({
                    'text': text,
                    'entities': [
                        (start, end, correction['field_label'])
                    ]
                })
        
        logger.info(f"Prepared {len(training_data)} training examples")
        return training_data
    
    def fine_tune_spacy_ner(self, training_data):
        """Fine-tune spaCy NER on corrections"""
        
        import spacy
        from spacy.training import Example
        
        # Load current model
        nlp = spacy.load(f"{self.models_dir}/ner_spacy_v1")
        ner = nlp.get_pipe('ner')
        
        # Fine-tune
        optimizer = nlp.create_optimizer()
        
        for epoch in range(5):  # Fewer epochs, we're fine-tuning
            losses = {}
            
            for raw_text, annotations in training_data:
                doc = nlp.make_doc(raw_text)
                example = Example.from_dict(doc, annotations)
                
                nlp.update(
                    [example],
                    sgd=optimizer,
                    drop=0.3,  # Lighter dropout for fine-tuning
                    losses=losses
                )
            
            logger.info(f"Epoch {epoch}, Loss: {losses.get('ner', 0):.4f}")
        
        return nlp
    
    def evaluate_new_model(self, new_model, test_data):
        """Evaluate new model vs. old"""
        
        # Load old model
        old_model = spacy.load(f"{self.models_dir}/ner_spacy_v1")
        
        # Test on held-out data
        new_f1 = self._calculate_f1(new_model, test_data)
        old_f1 = self._calculate_f1(old_model, test_data)
        
        improvement = new_f1 - old_f1
        
        logger.info(f"Old F1: {old_f1:.3f}, New F1: {new_f1:.3f}, Improvement: {improvement:.3f}")
        
        return {
            'new_f1': new_f1,
            'old_f1': old_f1,
            'improvement': improvement,
            'should_deploy': improvement > 0.01,  # At least 1% improvement
        }
    
    def deploy_new_model(self, new_model):
        """Save new model as production version"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"ner_spacy_v{timestamp}"
        
        new_model.to_disk(f"{self.models_dir}/{version}")
        
        # Update production pointer
        with open(f"{self.models_dir}/production_version.txt", 'w') as f:
            f.write(version)
        
        logger.info(f"Deployed model: {version}")
        
        return version
    
    def run_monthly_retrain(self):
        """Execute full retraining pipeline"""
        
        logger.info("Starting monthly retraining pipeline")
        
        # Step 1: Get corrections
        corrections = self.get_monthly_corrections()
        if len(corrections) < 10:
            logger.warning("Not enough corrections (<10) to retrain")
            return
        
        # Step 2: Prepare training data
        training_data = self.prepare_retraining_data(corrections)
        
        # Step 3: Fine-tune
        new_model = self.fine_tune_spacy_ner(training_data)
        
        # Step 4: Evaluate
        test_data = self.db.query("SELECT * FROM test_dataset")
        evaluation = self.evaluate_new_model(new_model, test_data)
        
        # Step 5: Deploy if improved
        if evaluation['should_deploy']:
            version = self.deploy_new_model(new_model)
            logger.info(f"✅ New model deployed: {version}")
        else:
            logger.warning(f"❌ New model not better ({evaluation['improvement']:.3f} improvement). Keeping old model.")
    
    def schedule_retraining(self):
        """Schedule retraining for 1st of every month"""
        
        # Run at 2 AM on the 1st
        schedule.every().monday.at("02:00").do(self.run_monthly_retrain)
        
        logger.info("Retraining scheduled for 1st of month at 02:00")

# Background task runner
def start_scheduler():
    """Run scheduler in background thread"""
    import threading
    
    pipeline = MonthlyRetrainingPipeline(db)
    pipeline.schedule_retraining()
    
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    thread = threading.Thread(target=run_scheduler, daemon=True)
    thread.start()
```

**Dependency:** Enables Month 5+ improvement cycle

---

## Q23: Model Versioning & A/B Testing
**Question:** How will you safely test new models before rolling them out?

**Options:**
1. Shadow mode (new model runs alongside, results logged but not used)
2. Canary rollout (new model for 5% of traffic)
3. A/B test (50% old, 50% new; measure which is better)
4. Direct rollout (deploy immediately)

**MY RECOMMENDATION: Option 2 + 3 - Canary (5%) then A/B test (50/50)**

**Justification:**
- **Canary (5%):** Catches major issues before wide deployment
- **A/B test (50/50):** Measure improvement on real traffic
- **Safe:** If new model is worse, only affects 50% of users

**Implementation:**

```python
# app/ml/model_versioning.py

import random
from datetime import datetime

class ModelVersionManager:
    def __init__(self, db):
        self.db = db
        self.active_models = {}
    
    def register_model(self, name, version, filepath):
        """Register a new model version"""
        
        model_info = {
            'name': name,
            'version': version,
            'filepath': filepath,
            'registered_at': datetime.now(),
            'status': 'inactive',  # 'inactive', 'canary', 'abtest', 'production'
            'traffic_percentage': 0,
        }
        
        self.db.save_model_version(model_info)
        
        return model_info
    
    def start_canary_rollout(self, name, version, traffic_pct=5):
        """Roll out new model to 5% of traffic"""
        
        model = self.db.get_model_version(name, version)
        model['status'] = 'canary'
        model['traffic_percentage'] = traffic_pct
        
        self.db.update_model_version(model)
        
        logger.info(f"Started canary rollout: {name}:{version} ({traffic_pct}% traffic)")
    
    def start_abtest(self, name, version, traffic_pct=50):
        """Roll out to 50% of traffic for A/B testing"""
        
        model = self.db.get_model_version(name, version)
        model['status'] = 'abtest'
        model['traffic_percentage'] = traffic_pct
        
        self.db.update_model_version(model)
        
        logger.info(f"Started A/B test: {name}:{version} ({traffic_pct}% traffic)")
    
    def promote_to_production(self, name, version):
        """Promote winning model to 100% of traffic"""
        
        model = self.db.get_model_version(name, version)
        model['status'] = 'production'
        model['traffic_percentage'] = 100
        
        self.db.update_model_version(model)
        
        logger.info(f"Promoted to production: {name}:{version}")
    
    def select_model_for_request(self, model_name):
        """
        Select which model version to use for this request.
        
        If both old (production) and new (canary/abtest) are available,
        decide based on traffic percentage.
        """
        
        versions = self.db.get_model_versions(model_name)
        
        # Separate by status
        production = [v for v in versions if v['status'] == 'production']
        candidate = [v for v in versions if v['status'] in ['canary', 'abtest']]
        
        if not candidate:
            # No candidate model; use production
            return production[0] if production else None
        
        # Decide: use candidate with probability = traffic_percentage / 100
        candidate_model = candidate[0]
        traffic_pct = candidate_model['traffic_percentage']
        
        if random.random() < traffic_pct / 100:
            return candidate_model
        else:
            return production[0]

# Usage in extraction pipeline
@app.post("/api/extract")
async def extract_sync(file: UploadFile = File(...)):
    
    # Select which NER model to use (production vs. candidate)
    ner_version = model_manager.select_model_for_request('ner_spacy')
    ner_model = load_model(ner_version['filepath'])
    
    # Select which confidence model to use
    confidence_version = model_manager.select_model_for_request('confidence_rf')
    confidence_model = load_model(confidence_version['filepath'])
    
    # Extract
    result = extraction_pipeline.extract(text)
    
    # Log which models were used (for A/B test analysis)
    result['metadata'] = {
        'ner_version': ner_version['version'],
        'confidence_version': confidence_version['version'],
    }
    
    return result

# A/B test analysis
def analyze_abtest():
    """Compare old vs. new model performance"""
    
    results = db.query("""
        SELECT 
            metadata->>'ner_version' as model_version,
            COUNT(*) as requests,
            AVG(CAST(accuracy AS FLOAT)) as avg_accuracy,
            AVG(confidence) as avg_confidence
        FROM extractions
        WHERE created_at > NOW() - INTERVAL '7 days'
        GROUP BY model_version
    """)
    
    print("A/B Test Results (last 7 days):")
    for row in results:
        print(f"  {row['model_version']}: {row['requests']} requests, "
              f"{row['avg_accuracy']:.1%} accuracy, {row['avg_confidence']:.2f} confidence")
```

**Canary → A/B → Production flow:**
```
Day 1-3: Canary (5% traffic)
  - Monitor error rates, accuracy
  - If good: proceed to A/B
  - If bad: rollback

Day 4-7: A/B test (50% traffic)
  - Measure: accuracy, precision, recall
  - Statistical significance test
  - If significantly better (p < 0.05): promote
  - If no improvement: rollback
```

**Dependency:** Enables safe model updates in production

---

## Q24: Monitoring & Drift Detection
**Question:** How will you detect when model performance degrades over time?

**Options:**
1. Manual monthly review (look at dashboards)
2. Automated alerts (if accuracy drops > 5%)
3. Drift detection (statistical test: does new data look like training data?)
4. Both (automated alerts + drift detection)

**MY RECOMMENDATION: Option 4 - Both (automated alerts + drift detection)**

**Justification:**
- **Alerts:** Catch sudden drops (model failure, data distribution shift)
- **Drift detection:** Catch gradual degradation (subtle data shifts)
- **Together:** Early warning system prevents silent failures

**Implementation:**

```python
# app/monitoring/drift_detection.py

import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class DriftDetector:
    """Detect if new data distribution differs from training data"""
    
    def __init__(self, training_features, training_labels):
        self.training_features = training_features
        self.training_labels = training_labels
    
    def detect_feature_drift(self, new_features, threshold=0.05):
        """
        Use Kolmogorov-Smirnov (KS) test to detect distribution shift.
        
        H0: new data comes from same distribution as training data
        p < threshold: reject H0 → drift detected
        """
        
        drifts = {}
        
        for feature_idx in range(self.training_features.shape[1]):
            # KS test: compare distributions
            statistic, p_value = stats.ks_2samp(
                self.training_features[:, feature_idx],
                new_features[:, feature_idx]
            )
            
            drifts[f'feature_{feature_idx}'] = {
                'statistic': statistic,
                'p_value': p_value,
                'drift_detected': p_value < threshold,
            }
        
        return drifts
    
    def detect_label_drift(self, new_labels, threshold=0.05):
        """Detect if label distribution changed"""
        
        # Chi-square test on label distribution
        unique_labels = np.unique(self.training_labels)
        
        training_counts = np.array([
            np.sum(self.training_labels == label)
            for label in unique_labels
        ])
        
        new_counts = np.array([
            np.sum(new_labels == label)
            for label in unique_labels
        ])
        
        # Normalize to percentages
        training_pct = training_counts / len(self.training_labels)
        new_pct = new_counts / len(new_labels)
        
        # Chi-square test
        chi2, p_value = stats.chisquare(new_counts, f_exp=training_pct * len(new_labels))
        
        return {
            'statistic': chi2,
            'p_value': p_value,
            'drift_detected': p_value < threshold,
            'training_distribution': dict(zip(unique_labels, training_pct)),
            'new_distribution': dict(zip(unique_labels, new_pct)),
        }

class PerformanceMonitor:
    """Monitor model accuracy over time"""
    
    def __init__(self, db, alert_threshold=0.05):
        self.db = db
        self.alert_threshold = alert_threshold  # Alert if accuracy drops > 5%
    
    def get_baseline_accuracy(self, days_back=30):
        """Get baseline accuracy from recent extractions"""
        
        results = self.db.query(f"""
            SELECT AVG(CAST(accuracy AS FLOAT)) as avg_accuracy
            FROM extractions
            WHERE created_at > NOW() - INTERVAL '{days_back} days'
        """)
        
        return results[0]['avg_accuracy']
    
    def check_performance_degradation(self, window_size=100):
        """
        Check if recent extractions are worse than baseline.
        
        Get last N extractions, compare accuracy to baseline.
        """
        
        # Get baseline
        baseline_accuracy = self.get_baseline_accuracy(days_back=30)
        
        # Get recent
        recent = self.db.query(f"""
            SELECT accuracy
            FROM extractions
            ORDER BY created_at DESC
            LIMIT {window_size}
        """)
        
        recent_accuracy = np.mean([float(r['accuracy']) for r in recent])
        
        degradation = baseline_accuracy - recent_accuracy
        
        if degradation > self.alert_threshold:
            logger.warning(
                f"🚨 PERFORMANCE DEGRADATION ALERT: "
                f"Baseline: {baseline_accuracy:.1%}, "
                f"Recent: {recent_accuracy:.1%}, "
                f"Degradation: {degradation:.1%}"
            )
            return {
                'degradation_detected': True,
                'baseline': baseline_accuracy,
                'recent': recent_accuracy,
                'degradation': degradation,
            }
        
        return {'degradation_detected': False}

# Scheduled monitoring task
def run_drift_monitoring():
    """Run drift detection daily"""
    
    drift_detector = DriftDetector(training_features, training_labels)
    performance_monitor = PerformanceMonitor(db)
    
    # Get recent data
    recent_data = db.query("""
        SELECT features, label
        FROM extractions
        WHERE created_at > NOW() - INTERVAL '1 day'
    """)
    
    new_features = np.array([d['features'] for d in recent_data])
    new_labels = np.array([d['label'] for d in recent_data])
    
    # Detect drift
    feature_drifts = drift_detector.detect_feature_drift(new_features)
    label_drift = drift_detector.detect_label_drift(new_labels)
    
    # Check performance
    perf_check = performance_monitor.check_performance_degradation()
    
    # Log results
    logger.info(f"Drift Detection Results:")
    logger.info(f"  Feature drift: {sum(1 for d in feature_drifts.values() if d['drift_detected'])} features")
    logger.info(f"  Label drift: {label_drift['drift_detected']}")
    logger.info(f"  Performance degradation: {perf_check.get('degradation_detected', False)}")
    
    # Alert if needed
    if any(d['drift_detected'] for d in feature_drifts.values()) or label_drift['drift_detected']:
        logger.warning("⚠️ Data drift detected. Consider retraining.")
    
    if perf_check.get('degradation_detected'):
        logger.error("🚨 Performance degradation detected. Investigate immediately.")

# Schedule daily
schedule.every().day.at("03:00").do(run_drift_monitoring)
```

**Alert dashboard:**
```
Daily Monitoring Report (2024-02-15)
✅ No performance degradation (accuracy stable at 94%)
⚠️  Possible data drift in feature_3 (p=0.032)
ℹ️  Label distribution unchanged (invoice/receipt ratio stable)

Recommendation: Continue monitoring. Retrain if accuracy drops below 92%.
```

**Dependency:** Closes the loop; enables proactive model maintenance

---

# MONTH 4 SUMMARY (END OF WEEK 16)

**Deliverables:**
1. ✅ Evaluation framework (per-field metrics, confusion matrix)
2. ✅ REST + Async APIs (single-doc + batch extraction)
3. ✅ Streamlit dashboard (operational, quality, business KPIs)
4. ✅ Human feedback UI (corrections workflow)
5. ✅ Monthly retraining pipeline (fine-tune on corrections)
6. ✅ Model versioning & A/B testing (safe rollouts)
7. ✅ Drift detection & performance monitoring (alerts)
8. ✅ Full documentation + user guide
9. ✅ Final presentation

**Final System Metrics (End of Month 4):**
```
Production Extraction System:
- 500+ documents processed (Month 4)
- 94% overall accuracy (target: >= 90%)
- 65% auto-accept rate (high confidence)
- 4 ms average inference time
- 99.5% uptime

Quality by field:
  INVOICE_ID:     96% F1-score ✅
  INVOICE_DATE:   90% F1-score ✅
  VENDOR_NAME:    85% F1-score ⚠️ (potential focus for Month 5+)
  TOTAL_AMOUNT:   93% F1-score ✅

Business impact:
  - 800 hours manual work saved (estimated)
  - $40k business value generated
  - 2 FTE headcount equivalent
```

---

# COMPLETE PROJECT TIMELINE

```
PHASE 0: BOOTSTRAP (Before Month 1)
  [ ] Label Studio setup
  [ ] Annotation schema definition
  [ ] Guideline documentation

MONTH 1: ANALYSIS & ANNOTATION (Weeks 1-4)
  [✓] Collect 50-200 sample documents
  [✓] Annotate with span-level schema
  [✓] Achieve 0.80+ inter-annotator agreement
  [✓] Export annotated corpus

MONTH 2: ML DEVELOPMENT (Weeks 5-8)
  [✓] Feature engineering (hand-crafted + BERT)
  [✓] Random Forest confidence model
  [✓] spaCy NER + regex ensemble
  [✓] BART document classifier
  [✓] Extraction pipeline orchestration
  [✓] Test on held-out data (86% F1)

MONTH 3: INTEGRATION & VALIDATION (Weeks 9-12)
  [✓] Confidence thresholding (per-field tuning)
  [✓] Anomaly detection (rules + Isolation Forest)
  [✓] KPI engine
  [✓] Process 100 documents with decision logic
  [✓] Test results: 94% accuracy

MONTH 4: PRODUCTION & LEARNING (Weeks 13-16)
  [✓] Evaluation framework + metrics
  [✓] FastAPI endpoints (sync + async)
  [✓] Streamlit dashboard
  [✓] Human feedback UI
  [✓] Retraining pipeline
  [✓] Model versioning & A/B testing
  [✓] Drift detection & monitoring
  [✓] Documentation + final presentation
```

---

# APPENDIX: KEY DECISIONS & RECOMMENDATIONS SUMMARY

| Q# | Question | My Recommendation | Key Insight |
|----|----------|-------------------|------------|
| Q4 | Annotation tool | Label Studio (self-hosted) | Reproducible, data-local, supports NER |
| Q5 | Annotation schema | Span-level | Trainable on 50-200 docs, position-aware |
| Q6 | Training data source | Real business documents | Ensures model generalizes to production |
| Q7 | Extraction schema | Host org's business requirements | Prevents over-engineering |
| Q8 | Annotation setup | Script + Markdown guidelines | Reproducible, onboardable for new annotators |
| Q9 | Annotation execution | Hybrid (you + 1-2 others) | Quality control + speed |
| Q10 | Feature engineering | Hand-crafted + BERT embeddings | Data-efficient + semantic power |
| Q11 | Baseline model | Random Forest | Works well on small data, interpretable |
| Q12 | NER model | spaCy + regex ensemble | Fast, accurate, handles context |
| Q13 | Doc classification | Zero-shot BART (no logic branching yet) | Informational only, no performance risk |
| Q14 | Pipeline orchestration | Sequential (NER → confidence → output) | Simple, debuggable, reliable |
| Q15 | Confidence thresholding | Per-field ranges (auto/review/reject) | Realistic, reviewable, tunable |
| Q16 | Anomaly detection | Rules + Isolation Forest | Catches domain-specific + statistical outliers |
| Q17 | KPI definition | Operational + quality + business | Comprehensive stakeholder visibility |
| Q18 | Evaluation metrics | Per-field + confusion matrix | Actionable insights |
| Q19 | API design | REST (sync) + Async (batch) | Covers interactive + batch workflows |
| Q20 | Dashboard | Streamlit | Fast to build, Python-native |
| Q21 | Feedback collection | Auto-capture + optional notes | Lightweight for humans, rich data for retraining |
| Q22 | Retraining | Fine-tune monthly on corrections | Safe, incremental, fast |
| Q23 | Model versioning | Canary (5%) → A/B (50%) → production | Safe rollouts, measurable improvement |
| Q24 | Monitoring | Automated alerts + drift detection | Proactive degradation detection |

---

# HOW TO USE THIS DOCUMENT

1. **Implementation checklist:** Print and check off as you complete each question's work
2. **Code templates:** Copy-paste implementations from each Q section
3. **Dependency resolver:** If stuck on Q15, read Q14 first (dependencies are noted)
4. **Timeline validation:** Compare your actual progress to Month timeline
5. **Stakeholder communication:** Use Q17 KPI definitions to show progress

---

**This completes the relentless, systematic walkthrough of your 4-month internship project.**

Every question has a recommended answer. Every answer has technical implications. Every technical implication has implementation code and examples.

Start with Q4 (choose annotation tool). Once that's set, everything else follows.

Good luck. 🚀

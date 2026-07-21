# Module 09: Enterprise NLP Applications & End-to-End Production Pipelines

This study guide details the 6 core enterprise NLP applications (Classification, NER, Sentiment, Translation, Extractive QA, Summarization), production system serving architectures, ONNX Runtime optimization, INT8 quantization, latency/throughput calculations, Python pipelines, failure modes, and interview flashcards.

> **Notebook Companion**: [09_nlp_applications_and_end_to_end_pipelines.ipynb](file:///d:/Study/Prep/machine-learning-prep/nlp/09_nlp_applications_and_end_to_end_pipelines.ipynb)

---

## 1. The 6 Core Enterprise NLP Applications

```text
Application              Input Data                  Output Target                  Core Metric            Production Architecture
--------------------------------------------------------------------------------------------------------------------------------------
Text Classification      Unstructured Document String Categorical Label (y in {1..C}) Macro F1               TF-IDF + Linear / DistilBERT
Named Entity Rec. (NER)  Unstructured Text Token     Token Entity Tags (B-PER, etc) Strict Token F1        CRF / Bi-LSTM-CRF / Token Classification
Sentiment Analysis       Customer Review / Tweet     Polarity Score / Aspect Tag    AUC-ROC / F1           Scikit-Learn / DeBERTa
Machine Translation      Source Language String      Target Language String         BLEU-4                 Seq2Seq + Attention / NMT LLM
Extractive QA            Passage + Question          Span Indices [Start, End]      Exact Match (EM) / F1  Bi-Encoder / Cross-Encoder BERT
Summarization            Long Document String        Concise Summary String         ROUGE-L                LED / BART / Modern LLM (GPT-4)
```

---

## 2. End-to-End Enterprise Production Pipeline Architecture

In enterprise microservice architectures, an NLP production service consists of 5 modular components:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. INGESTION & RATE LIMITING                                                           │
│    gRPC / REST API endpoint receiving raw JSON payload with rate limiting.              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. DETERMINISTIC TEXT PREPROCESSING                                                    │
│    Unicode normalization, HTML stripping, BPE subword tokenization.                   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. HIGH-THROUGHPUT MODEL INFERENCE                                                     │
│    ONNX Runtime / TensorRT engine running INT8 quantized model weights.                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. POST-PROCESSING & VALIDATION GUARDRAILS                                             │
│    Logit thresholding, PII masking, schema validation on extracted JSON outputs.       │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. METRIC TELEMETRY & AUDIT LOGGING                                                    │
│    Prometheus latency metrics (p95 / p99), Kafka event logging for model drift audit.   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Production Model Optimization Techniques

### 1. ONNX Runtime Export
Exporting PyTorch or TensorFlow models to the Open Neural Network Exchange (**ONNX**) graph representation graph decouples model inference from PyTorch execution overhead, delivering $2\text{x}-3\text{x}$ throughput improvements on CPU servers.

### 2. INT8 Quantization

![FP32 PyTorch vs INT8 ONNX Latency & Memory Benchmark](images/09_serving_latency_benchmark.png)

> **Plot Interpretation & Production Insight**:
> - **Latency Reduction**: Converting FP32 PyTorch models to INT8 ONNX Runtime reduces $p95$ inference latency from $22.4	ext{ms}$ down to $5.2	ext{ms}$ ($4.3	ext{x}$ throughput acceleration on CPU web servers).
> - **Memory Footprint Savings**: INT8 quantization compresses weight matrices from $420	ext{MB}$ to $105	ext{MB}$ ($75\%$ RAM reduction), allowing 4x more worker instances per server node.

Quantization converts 32-bit floating-point weights ($W_{\text{FP32}}$) to 8-bit integers ($W_{\text{INT8}}$):

$$W_{\text{INT8}} = \text{round}\left( \frac{W_{\text{FP32}}}{S} \right) + Z$$

Where $S$ is the scale factor and $Z$ is the zero-point offset.
- **Benefits**: Reduces model RAM footprint by **75%** (e.g. 400MB FP32 model shrinks to 100MB INT8 model) with $<0.5\%$ loss in classification accuracy.

---

## 4. Step-by-Step Production Throughput Calculation (Andrew Ng Style)

Suppose an enterprise API endpoint receives text classification requests under the following production SLA constraints:
- **Server Hardware**: 4 vCPU web server instance.
- **Model Inference Execution Latency**: $t_{\text{inf}} = 20\text{ms}$ per request on a single CPU thread.
- **Target Latency SLA**: $p95 \le 50\text{ms}$.

### 1. Compute Single-Thread Throughput ($QPS_{\text{single}}$):
$$QPS_{\text{single}} = \frac{1000\text{ms}}{t_{\text{inf}}} = \frac{1000}{20} = \mathbf{50\text{ Queries Per Second (QPS)}}$$

### 2. Compute Total 4 vCPU Server Capacity ($QPS_{\text{total}}$):
Assuming 4 parallel worker threads with $85\%$ multi-core scaling efficiency:

$$QPS_{\text{total}} = 4 \times 50 \times 0.85 = \mathbf{170\text{ QPS}}$$

### 3. Estimate Daily Processing Capacity:
$$\text{Daily Capacity} = 170 \text{ req/sec} \times 86,400 \text{ sec/day} \approx \mathbf{14,688,000\text{ requests/day}}$$

---

## 5. Production Python Code Implementation

```python
import os
import joblib
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class EnterpriseNERClassifierService:
    """Production NLP Microservice with pre-processing, inference, and JSON validation."""
    
    def __init__(self, model_dir: str = ".model_cache"):
        self.model_dir = model_dir
        self.model_path = os.path.join(self.model_dir, "production_pipeline.pkl")
        self.pipeline = self._load_or_train_pipeline()
        
    def _load_or_train_pipeline(self) -> Pipeline:
        os.makedirs(self.model_dir, exist_ok=True)
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
            
        # Baseline training data
        X = ["Database server 10.0.1.4 error connection refused", "Billing invoice payment failed 402"]
        y = ["Infrastructure", "Billing"]
        
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression())
        ])
        pipe.fit(X, y)
        joblib.dump(pipe, self.model_path)
        return pipe

    def process_request(self, raw_json_payload: str) -> str:
        """Executes full production inference pipeline."""
        data = json.loads(raw_json_payload)
        text = data.get("text", "").strip()
        
        if not text:
            return json.dumps({"error": "Empty text payload", "status_code": 400})
            
        prediction = self.pipeline.predict([text])[0]
        confidence = float(max(self.pipeline.predict_proba([text])[0]))
        
        response = {
            "text_processed": text,
            "category": prediction,
            "confidence_score": round(confidence, 4),
            "status_code": 200
        }
        return json.dumps(response, indent=2)

# Execution Demonstration
service = EnterpriseNERClassifierService()
sample_payload = json.dumps({"text": "PostgreSQL database 10.0.1.4 connection failed on port 5432"})
output_json = service.process_request(sample_payload)

print("=== Enterprise Production Pipeline Execution ===")
print(output_json)
```

> [!NOTE]
> **Production Microservice Alert:**
> - Input validation (`if not text`) prevents downstream model crashes on malformed API requests.
> - Explicitly wrapping output in typed JSON response schemas ensures seamless integration with enterprise frontend services.

---

## 6. Production Failure Modes & Selection Rules

### Production Failure Modes:
1. **Model Drift & Concept Drift**: Over time, incoming production text terminology shifts (e.g. new software product release names), causing offline-trained models to suffer declining accuracy.
   - *Remediation*: Implement continuous evaluation logs and trigger automated retraining pipelines when macro F1 drops below $90\%$.
2. **Cold-Start Latency Spikes**: Loading heavy PyTorch models during server container initialization causes first-request latency spikes ($>5\text{ seconds}$).
   - *Remediation*: Execute synthetic "warmup" inference passes during docker container startup readiness probes.

---

## 7. Master Interview Flashcards & Questions

#### Q1: What is ONNX Runtime, and how does it optimize NLP model serving?
- **Answer:** ONNX (Open Neural Network Exchange) is an open format for representing machine learning models. Exporting PyTorch models to ONNX compiles computational graphs into optimized, hardware-agnostic execution formats. ONNX Runtime eliminates PyTorch framework overhead, applies graph optimizations (node fusion, constant folding), and delivers $2\text{x}-3\text{x}$ lower latency on CPU servers.

#### Q2: Compare Post-Training Static Quantization vs. Quantization-Aware Training (QAT).
- **Answer:** Post-Training Static Quantization converts FP32 weights to INT8 after training is completed using a calibration dataset to determine scale factors. It is fast and requires no retraining, but may suffer minor accuracy loss. Quantization-Aware Training (QAT) models quantization noise during backpropagation, allowing the model to adapt weights to 8-bit precision, preserving high accuracy.

#### Q3: How do you handle Data Drift and Concept Drift in production NLP applications?
- **Answer:** Data Drift occurs when input text distributions change (e.g., new terminology). Concept Drift occurs when the relationship between input text and labels shifts. In production, we log text samples to a Kafka stream, monitor token distribution shifts against training baselines using Jensen-Shannon divergence, and trigger automated retraining pipelines when performance metrics degrade.

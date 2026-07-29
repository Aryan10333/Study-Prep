# Module 05: Word Embeddings & Semantic Spaces

Word embeddings project discrete vocabulary tokens into low-dimensional, dense continuous vector spaces. This module details embedding projection theory, compares Word2Vec architectures, and explains FastText Out-of-Vocabulary (OOV) resolution.

---

## 1. Embedding Space Projection Theory

Word embeddings translate semantic similarity into spatial proximity. The learning pipeline follows four main steps:

<div style="margin: 20px 0; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; font-family: sans-serif; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
  <div style="display: flex; gap: 12px; justify-content: space-between;">
    <!-- Step 1 -->
    <div style="flex: 1; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.02); display: flex; flex-direction: column; justify-content: space-between;">
      <div>
        <div style="color: #3b82f6; font-weight: bold; font-size: 11px; text-transform: uppercase; margin-bottom: 6px;">1. Paradigm</div>
        <div style="color: #0f172a; font-weight: 600; font-size: 13px; margin-bottom: 8px;">Distributional Hypothesis</div>
        <div style="color: #64748b; font-size: 11px; line-height: 1.45;">"Words in similar contexts have similar meanings."</div>
      </div>
      <div style="margin-top: 12px; border-top: 1px solid #f1f5f9; padding-top: 8px; color: #1e3a8a; font-family: monospace; font-size: 11px; font-weight: bold;">Co-occurrence data</div>
    </div>
    
    <!-- Step 2 -->
    <div style="flex: 1; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.02); display: flex; flex-direction: column; justify-content: space-between;">
      <div>
        <div style="color: #10b981; font-weight: bold; font-size: 11px; text-transform: uppercase; margin-bottom: 6px;">2. Objective</div>
        <div style="color: #0f172a; font-weight: 600; font-size: 13px; margin-bottom: 8px;">Prediction Setup</div>
        <div style="color: #64748b; font-size: 11px; line-height: 1.45;">Predict context given target word (or vice-versa).</div>
      </div>
      <div style="margin-top: 12px; border-top: 1px solid #f1f5f9; padding-top: 8px; color: #065f46; font-family: monospace; font-size: 11px; font-weight: bold;">Skip-gram / CBOW</div>
    </div>
    
    <!-- Step 3 -->
    <div style="flex: 1; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.02); display: flex; flex-direction: column; justify-content: space-between;">
      <div>
        <div style="color: #f59e0b; font-weight: bold; font-size: 11px; text-transform: uppercase; margin-bottom: 6px;">3. Bottleneck</div>
        <div style="color: #0f172a; font-weight: 600; font-size: 13px; margin-bottom: 8px;">Hidden Layer</div>
        <div style="color: #64748b; font-size: 11px; line-height: 1.45;">Linear projection maps sparse tokens to continuous states.</div>
      </div>
      <div style="margin-top: 12px; border-top: 1px solid #f1f5f9; padding-top: 8px; color: #92400e; font-family: monospace; font-size: 11px; font-weight: bold;">Weight Matrix W</div>
    </div>
    
    <!-- Step 4 -->
    <div style="flex: 1; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.02); display: flex; flex-direction: column; justify-content: space-between;">
      <div>
        <div style="color: #8b5cf6; font-weight: bold; font-size: 11px; text-transform: uppercase; margin-bottom: 6px;">4. Representation</div>
        <div style="color: #0f172a; font-weight: 600; font-size: 13px; margin-bottom: 8px;">Embedding Space</div>
        <div style="color: #64748b; font-size: 11px; line-height: 1.45;">Dense vectors capture analogical offsets geometrically.</div>
      </div>
      <div style="margin-top: 12px; border-top: 1px solid #f1f5f9; padding-top: 8px; color: #5b21b6; font-family: monospace; font-size: 11px; font-weight: bold;">Cosine Similarity</div>
    </div>
  </div>
</div>

- **Distributional Hypothesis**: Words that occur in similar contexts share semantic meaning.
- **Prediction Objective**: Rather than counting co-occurrences directly, models set up a prediction task (e.g. predicting a word from its neighbors).
- **Hidden Layer**: To solve this task, the model maps inputs through a low-dimensional bottleneck layer.
- **Embedding Space**: The weights of this bottleneck layer form the embedding matrix $\mathbf{W} \in \mathbb{R}^{|V| \times d}$, capturing semantic properties like analogies (e.g., $\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$).

### Spatial Analogy Visualization

![Word Embedding Analogy Projection](file:///d:/Study/Prep/machine-learning-prep/generative-ai-and-agentic-ai/00_nlp_fundamentals/plots/embedding_analogy_projection.png)

> [!NOTE]
> **Plot Explanation & Intuition: Embedding Analogy Projection**
> This plot projects word embedding vectors onto a 2D coordinate space to demonstrate semantic properties like analogies:
> - **Gender Offset (Horizontal Blue Vector)**: The direction and distance from `"woman"` to `"man"` matches the vector from `"queen"` to `"king"`. This indicates the model has learned the abstract concept of gender as a consistent spatial translation vector.
> - **Royalty Offset (Vertical Green Vector)**: The transition from `"man"` to `"king"` is parallel to the transition from `"woman"` to `"queen"`, capturing the concept of royalty.
> - **Production Takeaway**: This spatial layout shows that dense embeddings translate semantic relationships into geometric relationships, allowing downstream neural layers to exploit semantic analogies using simple vector additions and subtractions.

<div style="margin: 20px 0; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 30px 20px; font-family: sans-serif; box-shadow: 0 1px 3px rgba(0,0,0,0.02); display: flex; justify-content: center; align-items: center; page-break-inside: avoid;">
  <div style="position: relative; width: 440px; height: 260px; border-left: 2px solid #64748b; border-bottom: 2px solid #64748b;">
    
    <!-- Y-Axis Label -->
    <div style="position: absolute; top: -25px; left: -10px; font-size: 11px; font-weight: bold; color: #475569;">Vector Dimension y</div>
    
    <!-- X-Axis Label -->
    <div style="position: absolute; bottom: -25px; right: 0; font-size: 11px; font-weight: bold; color: #475569;">Vector Dimension x</div>
    
    <!-- Points -->
    
    <!-- Woman (0.5, 2.5) -->
    <div style="position: absolute; left: 70px; top: 30px; background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; z-index: 10;">
      <span style="font-weight: bold; color: #0f172a; font-size: 12px;">[woman]</span><br>
      <span style="font-size: 9px; color: #64748b; font-family: monospace;">(0.5, 2.5)</span>
    </div>
    
    <!-- Man (0.5, 1.2) -->
    <div style="position: absolute; left: 70px; top: 160px; background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; z-index: 10;">
      <span style="font-weight: bold; color: #0f172a; font-size: 12px;">[man]</span><br>
      <span style="font-size: 9px; color: #64748b; font-family: monospace;">(0.5, 1.2)</span>
    </div>
    
    <!-- Queen (1.8, 2.5) -->
    <div style="position: absolute; left: 270px; top: 30px; background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; z-index: 10;">
      <span style="font-weight: bold; color: #0f172a; font-size: 12px;">[queen]</span><br>
      <span style="font-size: 9px; color: #64748b; font-family: monospace;">(1.8, 2.5)</span>
    </div>
    
    <!-- King (1.8, 1.2) -->
    <div style="position: absolute; left: 270px; top: 160px; background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; z-index: 10;">
      <span style="font-weight: bold; color: #0f172a; font-size: 12px;">[king]</span><br>
      <span style="font-size: 9px; color: #64748b; font-family: monospace;">(1.8, 1.2)</span>
    </div>
    
    <!-- Vectors -->
    
    <!-- Left Vector: Woman -> Man (downward) -->
    <div style="position: absolute; left: 105px; top: 72px; width: 0px; height: 85px; border-left: 2px dashed #3b82f6; z-index: 1;">
      <div style="position: absolute; bottom: -5px; left: -5px; width: 0; height: 0; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 6px solid #3b82f6;"></div>
      <div style="position: absolute; left: 8px; top: 32px; font-size: 9px; color: #2563eb; font-weight: bold; white-space: nowrap;">Gender Offset</div>
    </div>
    
    <!-- Right Vector: Queen -> King (downward) -->
    <div style="position: absolute; left: 305px; top: 72px; width: 0px; height: 85px; border-left: 2px dashed #3b82f6; z-index: 1;">
      <div style="position: absolute; bottom: -5px; left: -5px; width: 0; height: 0; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 6px solid #3b82f6;"></div>
      <div style="position: absolute; left: 8px; top: 32px; font-size: 9px; color: #2563eb; font-weight: bold; white-space: nowrap;">Gender Offset</div>
    </div>

    <!-- Top Vector: Woman -> Queen (rightward) -->
    <div style="position: absolute; left: 145px; top: 48px; width: 115px; height: 0px; border-top: 2px dashed #10b981; z-index: 1;">
      <div style="position: absolute; right: -5px; top: -4px; width: 0; height: 0; border-top: 4px solid transparent; border-bottom: 4px solid transparent; border-left: 6px solid #10b981;"></div>
      <div style="position: absolute; left: 24px; top: -16px; font-size: 9px; color: #059669; font-weight: bold; white-space: nowrap;">Royalty Offset</div>
    </div>

    <!-- Bottom Vector: Man -> King (rightward) -->
    <div style="position: absolute; left: 145px; top: 178px; width: 115px; height: 0px; border-top: 2px dashed #10b981; z-index: 1;">
      <div style="position: absolute; right: -5px; top: -4px; width: 0; height: 0; border-top: 4px solid transparent; border-bottom: 4px solid transparent; border-left: 6px solid #10b981;"></div>
      <div style="position: absolute; left: 24px; top: -16px; font-size: 9px; color: #059669; font-weight: bold; white-space: nowrap;">Royalty Offset</div>
    </div>

  </div>
</div>

---

## 2. Word2Vec: CBOW vs. Skip-gram

Word2Vec is a computationally efficient 3-layer neural network architecture (Input, Projection/Hidden, Output) with no activation functions in the hidden layer (it acts as a linear projection). Once trained, the input projection matrix $\mathbf{W}_{\text{in}} \in \mathbb{R}^{|V| \times d}$ is saved as the final word embedding lookup table, and the output layer is discarded.

### 1. Continuous Bag-of-Words (CBOW)
- **Concept**: Predicts a missing target word $w_t$ given its surrounding context words in a window $C$.
- **Mechanism**: The input context words are one-hot encoded $\mathbf{x} \in \mathbb{R}^{|V|}$ and projected using the input weight matrix $\mathbf{W}_{\text{in}}$. The model averages these vectors to construct a single hidden state $\mathbf{h} \in \mathbb{R}^d$:
  $$\mathbf{h} = \frac{1}{2C} \sum_{j=-C, j \neq 0}^{C} \mathbf{W}_{\text{in}}^T \mathbf{x}_{t+j}$$
  This averaged vector is multiplied by the output weight matrix $\mathbf{W}_{\text{out}}$ to compute logits over the vocabulary, predicting the target word.
- **Production Trade-offs**: Averaging context vectors smooths out spatial relationships and runs faster, making CBOW ideal for capturing general topic patterns on very large corpora.

### 2. Skip-gram
- **Concept**: Predicts the surrounding context words in a window $C$ given a single target word $w_t$.
- **Mechanism**: A single target word $w_t$ is one-hot encoded $\mathbf{x}_t$, projected via $\mathbf{W}_{\text{in}}$ to a hidden state $\mathbf{h} = \mathbf{W}_{\text{in}}^T \mathbf{x}_t \in \mathbb{R}^d$. This vector is multiplied by $\mathbf{W}_{\text{out}}$ to calculate logits and softmax probabilities separately for each of the $2C$ context positions:
  $$P(w_{t+j} | w_t) = \text{Softmax}(\mathbf{W}_{\text{out}}^T \mathbf{h})$$
- **Production Trade-offs**: Because each target-context pair is treated as a separate training instance, it does not average out representations. This makes it slower to train but significantly better at representing rare words.

---

### Negative Sampling Optimization (SGNS)

#### The Softmax Denominator Bottleneck
Standard training using a final Softmax layer requires calculating:
$$P(w_{\text{context}} | w_{\text{target}}) = \frac{\exp(\mathbf{v}'_{w_{\text{context}}} \cdot \mathbf{v}_{w_{\text{target}}})}{\sum_{w_i \in V} \exp(\mathbf{v}'_{w_i} \cdot \mathbf{v}_{w_{\text{target}}})}$$
Computing the sum in the denominator requires executing a dot product between the target vector and the output vectors of all $|V|$ words in the vocabulary. For large vocabularies ($|V| \ge 100,000$), calculating this sum during backpropagation is extremely slow ($O(|V|)$ cost per training step).

#### Binary Logistic Regression Optimization
Negative Sampling (SGNS) replaces the multi-class classification task with a set of binary classification tasks. Instead of predicting a word's multi-class label, the model uses logistic regression to distinguish whether a target-context pair $(w_t, w_c)$ came from the real training corpus (positive sample) or random noise (negative samples):
$$\mathcal{L}_{\text{SGNS}} = -\log \sigma(\mathbf{v}'_{w_O} \cdot \mathbf{v}_{w_I}) - \sum_{i=1}^K \log \sigma(-\mathbf{v}'_{w_i} \cdot \mathbf{v}_{w_I})$$
For each target word $w_I$, we update its weights against:
- The true context word $w_O$ (maximizing their similarity).
- $K$ randomly selected "negative" noise words $w_i$ (minimizing their similarity).
This reduces the computational cost from $O(|V|)$ to $O(K)$, where $K$ is typically small (e.g. $5\text{--}20$ for small corpora, or $2\text{--}5$ for large ones), speeding up training by orders of magnitude.

#### Rationale Behind the $3/4$ Power Noise Distribution
Negative words are drawn from a modified unigram distribution:
$$P_n(w) \propto U(w)^{0.75}$$
If we sample purely by raw frequency $U(w)$, we will constantly select stopwords like `"the"`, `"a"`, `"and"`, which provides no useful gradient signal. Raising the frequency to the $0.75$ power dampens the relative frequency of highly common words while boosting the sampling probability of rare words. For example:
- A rare word with a probability of $0.00001$ gets a boosted probability of $0.00001^{0.75} \approx 0.00018$ ($18\times$ boost!).
- A stopword with a probability of $0.09$ gets $0.09^{0.75} \approx 0.16$ (only a $1.7\times$ boost).
This ensures the model updates rare words frequently during training.

---

## 3. GloVe and FastText

Alternative models address Word2Vec's limitations:

- **GloVe (Global Vectors)**: Factorizes the global word co-occurrence matrix directly rather than scanning local windows, balancing local context and global statistics.
- **FastText (Subword N-grams for OOV)**: Represents each word as a bag of character n-grams.
    - *Example*: For word `"supercomputing"` and $n=3$, character n-grams include: `<su`, `sup`, `upe`, `per`, `...`, `ing>`.
    - *OOV Resolution*: If `"supercomputing"` was not seen during training, FastText can still generate its embedding vector by summing the vectors of its constituent character n-grams, whereas Word2Vec and GloVe return a generic `<unk>` token.

---

> [!TIP]
> **Production Insight: Embedding Layer Memory Footprint**
> Static embedding layers require significant VRAM. A vocabulary of size $|V| = 250,000$ with dimension $d = 300$ requires storing $75,000,000$ float32 parameters ($\approx 300\text{MB}$). While small compared to LLMs, this can exceed memory budgets on edge devices. Quantizing the embedding matrix to int8 reduces this size to $75\text{MB}$ with minimal semantic decay.

---

### Interview Questions & Production Trade-offs

- **What problem does this solve?**
  Maps discrete vocabulary words to dense, low-dimensional vector representations that capture semantic similarity.
- **Why was it introduced?**
  Introduced to solve the high dimensionality and orthogonal constraints of sparse representations (One-Hot).
- **What are its limitations?**
  Static embeddings assign a single vector to each word, failing to handle polysemy (e.g. `"bank"` in `"river bank"` vs. `"money bank"`).
- **Computational Complexity (Time & Memory)**
    - **Inference Lookup Time**: $O(1)$ constant time lookup.
    - **Training Time**: Skip-gram with negative sampling scales as $O(K \cdot N \cdot C)$ where $C$ is corpus size.
- **Component Variable Denotation Legend**
    - $|V|$: Vocabulary size.
    - $d$: Vector dimension size (typically $100\text{--}300$).
    - $K$: Number of negative samples.
    - $C$: Corpus token count.
- **Production Use Cases**
    - High-speed semantic similarity searches.
    - Initializing embedding layers in recurrent or classification neural networks.
- **Follow-up questions interviewers ask**
    - *Why does CBOW train faster than Skip-gram?* (CBOW averages context embeddings into a single vector to predict one target, while Skip-gram runs multiple predictions per target word).
    - *Why is the noise distribution raised to the 3/4 power in Negative Sampling?* (It increases the relative sampling probability of rare words, ensuring the model updates rare word vectors frequently).

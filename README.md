```markdown
# ToxiScan — Cosmetic Ingredient Safety Predictor

ToxiScan is a web-based cheminformatics pipeline that uses machine learning to assess chemical toxicity risks from molecular structures. Designed with consumer safety and cosmetic screening in mind, it converts chemical formulations—represented as standard SMILES strings—into molecular features via Morgan Fingerprints (radius=2, 2048-bit) and processes them through trained ensemble classifiers to return calibrated toxicity risk assessments.

The models are trained using the **ClinTox** benchmark dataset (sourced from DeepChem), which flags compounds that failed clinical trials due to toxicity (`CT_TOX`).

---

## Architecture Overview

The system utilizes a 3-tier decoupling architecture to bridge a modern web UI with data-heavy Python ML frameworks:

1. **Frontend (`cosmetic_toxicity_predictor.html`)**: An interactive interface built with vanilla HTML/CSS (Syne & DM Mono typography styles) featuring safety threshold sliders, real-world chemical preset chips, and structural rendering hooks.
2. **Backend Engine (`server.js`)**: An Express.js Node server handling API request routing. It dynamically spawns an asynchronous sub-process tracking standard streams to run inference without freezing the network loop.
3. **Inference Script (`Toxicity_csv-collect.py`)**: A Python engine using `RDKit` for sub-structure chemical parsing and `scikit-learn` / `XGBoost` for matrix transformation and statistical classification.

---

## Required Software & Environments

Before installing libraries, make sure the following base engines are compiled on your operating system:

* **Node.js** (v16.x or higher) -> [Download Node.js](https://nodejs.org/)
* **Python** (v3.9 to v3.11 recommended for RDKit compatibility) -> [Download Python](https://www.python.org/)
* **Operating System**: Linux, macOS, or Windows 10/11 (equipped with Bash/PowerShell tools).

---

## Prerequisites & Installation

### 1. Project Directory Configuration
Clone or download the project files into a dedicated local workspace:
```bash
cd /your-local-path/ToxiScan

```

### 2. Node.js Dependency Setup

Initialize the Node runtime environment and install the Express framework:

```bash
npm init -y
npm install express

```

### 3. Python Extension Libraries

Because chemistry compilation requires C-bindings, ensure your package installer (`pip`) is upgraded before installing the toolkits:

```bash
pip install --upgrade pip
pip install numpy pandas scikit-learn xgboost rdkit

```

*Note on RDKit: Older versions of Python required complex Anaconda installations, but modern pipelines can fetch native standalone instances directly via `pip install rdkit`.*

---

## Detailed Execution Guide (How to Run)

Follow these explicit sequential steps to boot the servers on your machine:

### Step 1: Boot Up the Backend Network

Launch an open instance of your command terminal, navigate to the source directory, and spin up the Node listener script:

```bash
node server.js

```

Upon successful execution, the terminal will print:

```text
Server is running at http://localhost:3000

```

### Step 2: Accessing the Graphical UI

Open any modern chromium-based web browser (Chrome, Edge, or Brave) and navigate to the local network port address:

```text
http://localhost:3000

```

### Step 3: Triggering Inference

* Type or paste a structural SMILES string into the UI field (e.g., Caffeine: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`).
* Select your processing model target (**XGBoost** or **Random Forest**).
* Adjust the safety threshold slider (Default: `0.25`).
* Click **Analyze Molecule** to execute the pipeline.

---

## Project Negatives & Structural Issues (Things to Correct)

While the project features a solid visual framework and robust core models, the current codebase contains **critical architectural bottlenecks and bugs** that require engineering corrections:

### 1. High-Latency Training Bottleneck

* **The Issue**: Every time a user clicks "Analyze Molecule," `server.js` triggers `spawn('python', ['Toxicity_csv-collect.py'])`. The Python file fetches a remote 1,484-record dataset from an AWS S3 bucket over HTTP, transforms the entire array to fingerprints, and trains **both** a Random Forest and an XGBoost model *on the fly* before executing a single line of inference.
* **The Correction Required**: Separate training and inference operations. Use a standalone utility script to save your trained weights into a serialized model artifact (e.g., via `pickle` or `joblib` files). Modify `Toxicity_csv-collect.py` to simply load the static serialized file to instantly score the input structure.

### 2. Broken State Interaction (Frontend vs. Backend)

* **The Issue**: The frontend UI (`cosmetic_toxicity_predictor.html`) contains an analytical script function `analyzeSmiles()` that completely ignores the backend Express API (`/api/predict`). It uses a deterministic mock math algorithm based on a string character-code hashing system (`Math.sin(hash)`) to simulate a pseudo-prediction directly inside the client browser.
* **The Correction Required**: Rebuild `analyzeSmiles()` inside the frontend layout to issue a structural `fetch()` POST request carrying a JSON payload (`smiles` and `model`) directly to your backend endpoint, updating the UI elements with actual data returned from the server.

### 3. Asynchronous Multi-Line Collision Error

* **The Issue**: The Python script loops continuously using `for line in sys.stdin`, but it prints dataset loading logs, distribution text blocks, and performance reports directly to `sys.stdout`. The Node script (`server.js`) attempts to pass these diagnostic logs through a strict single-step parser (`JSON.parse(scriptData)`), which causes application crashes due to invalid JSON syntax.
* **The Correction Required**: Remove structural evaluation prints, reports, and shapes from the active script pipeline, or redirect non-analytical logging statements to the standard error track (`sys.stderr.write()`).

### 4. Absence of Client Threshold Pipelines

* **The Issue**: The user adjusts the Safety Threshold on the frontend UI, but this numeric control setting is never bundled or transferred via the payload array to the analytical backend.
* **The Correction Required**: Bind the specific slider threshold configuration into the state request structure so your machine learning logic can handle decision-boundary evaluations properly.

```

```

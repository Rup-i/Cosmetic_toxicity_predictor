const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const app = express();
const PORT = 3000;

app.use(express.json());

// Serve your HTML frontend file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'cosmetic_toxicity_predictor.html'));
});

// Endpoint that captures frontend input and runs it through Python
app.post('/api/predict', (req, res) => {
    const { smiles, model } = req.body;

    // Launches the Python script as a background worker
    const pythonProcess = spawn('python', ['Toxicity_csv-collect.py']);

    let scriptData = "";

    // Sends the data to Python's sys.stdin
    pythonProcess.stdin.write(JSON.stringify({ smiles, model }) + "\n");
    pythonProcess.stdin.end();

    // Collects output from Python's print statements (sys.stdout)
    pythonProcess.stdout.on('data', (data) => {
        scriptData += data.toString();
    });

    pythonProcess.on('close', (code) => {
        try {
            const result = JSON.parse(scriptData);
            res.json(result);
        } catch (error) {
            res.status(500).json({ error: "Failed to parse model output.", raw: scriptData });
        }
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python Error: ${data}`);
    });
});

app.listen(PORT, () => {
    console.log(`Server is running at http://localhost:${PORT}`);
});
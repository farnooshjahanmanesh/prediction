const express = require("express");
const bodyParser = require("body-parser");
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000; // استفاده از پورت محیطی یا 3000 به عنوان پیش‌فرض


// Middleware for parsing request bodies
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

// Serve static files from the 'public' directory
app.use(express.static(path.join(__dirname, "public")));

// --- Model Loading ---
// __dirname refers to the directory where the current script (app.js) is located.
// Assuming app.js is in /data-project/
// '..' goes up one level to /Desktop/
// Another '..' goes up another level to /farnooshjahanmanesh/
// Then we specify the model file name.
const modelPath = path.join(__dirname, "logisticRegressionModel.json");

let modelData;
let featureNames, means, stds, weights, bias;

try {
    if (!fs.existsSync(modelPath)) {
        console.error(`Model file not found at: ${modelPath}`);
        console.error("Please ensure the model file 'logisticRegressionModel.json' exists in the correct parent directory.");
        console.error("Current __dirname is:", __dirname);
        process.exit(1); // Exit if the model file is not found
    }

    // Read and parse the model file
    const modelFileContent = fs.readFileSync(modelPath, "utf8");
    modelData = JSON.parse(modelFileContent);

    // Extract model parameters
    featureNames = modelData.featureNames;
    means = modelData.means;
    stds = modelData.stds;
    weights = modelData.weights;
    bias = modelData.bias;

    // Validate essential model parameters
    if (!featureNames || !means || !stds || !weights || bias === undefined) {
        console.error("Model file is missing essential parameters (featureNames, means, stds, weights, or bias).");
        process.exit(1);
    }

    console.log("Model loaded successfully.");

} catch (error) {
    console.error("Error loading or parsing the model file:", error.message);
    process.exit(1); // Exit if there's an error during loading/parsing
}

// --- Helper Functions for Prediction ---

// Sigmoid activation function
function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
}

// Normalize input features
function normalizeInput(inputValues) {
    return inputValues.map((value, i) => {
        // Handle potential division by zero if std dev is 0
        const std = stds[i] === 0 ? 1 : stds[i];
        return (value - means[i]) / std;
    });
}

// Predict the probability using the logistic regression model
function predictProbability(inputValues) {
    const normalizedInput = normalizeInput(inputValues);

    let score = bias;
    for (let i = 0; i < normalizedInput.length; i++) {
        // Ensure weights array is long enough
        if (i < weights.length) {
            score += normalizedInput[i] * weights[i];
        } else {
            console.warn(`Warning: Missing weight for feature index ${i}. Skipping.`);
        }
    }
    return sigmoid(score);
}

// --- API Routes ---

// Serve the main HTML file for the root URL
app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, "public", "index.html"));
});

// API endpoint for making predictions
app.post("/predict", (req, res) => {
    try {
        const inputData = req.body;
        const inputValues = [];

        // Map input data to the order of featureNames
        for (const featureName of featureNames) {
            const value = parseFloat(inputData[featureName]);
            if (isNaN(value)) {
                return res.status(400).json({
                    error: `Invalid input: '${featureName}' must be a number. Received: '${inputData[featureName]}'`
                });
            }
            inputValues.push(value);
        }

        // Ensure the number of input values matches the expected features
        if (inputValues.length !== featureNames.length) {
             return res.status(400).json({
                 error: `Incorrect number of features provided. Expected ${featureNames.length}, received ${inputValues.length}.`
             });
        }

        const probability = predictProbability(inputValues);
        const prediction = probability >= 0.5 ? 1 : 0; // Assuming 0.5 is the threshold
        const risk = prediction === 1 ? "High" : "Low";

        // Prepare the response, mapping input values back to their names
        const responseInput = {};
        featureNames.forEach((name, index) => {
            responseInput[name] = inputValues[index];
        });

        res.json({
            prediction: prediction,
            risk: risk,
            probability: Number(probability.toFixed(4)), // Format probability to 4 decimal places
            input: responseInput
        });

    } catch (error) {
        console.error("Error during prediction:", error.message);
        // Send a generic error response to the client
        res.status(500).json({
            error: "An internal server error occurred during prediction.",
            details: error.message // Optionally include details for debugging, but be cautious in production
        });
    }
});

// --- Start Server ---
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});

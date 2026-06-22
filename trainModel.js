const fs = require("fs");
const path = require("path");
const csv = require("csv-parser");

const DATASET_PATH = path.join(__dirname, "Dataset.csv");
const MODEL_PATH = path.join(__dirname, "logisticRegressionModel.json");

const featureNames = [
  "age",
  "length_of_stay_days",
  "num_previous_admissions",
  "num_medications",
  "hba1c",
  "blood_glucose_mg_dl",
  "systolic_bp_mmhg",
  "creatinine_mg_dl",
  "haemoglobin_g_dl",
  "total_charges_inr"
];

const targetName = "readmitted_30_days";

const rows = [];

function mean(arr) {
  return arr.reduce((sum, x) => sum + x, 0) / arr.length;
}

function std(arr, m) {
  const variance = arr.reduce((sum, x) => sum + Math.pow(x - m, 2), 0) / arr.length;
  return Math.sqrt(variance) || 1;
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

fs.createReadStream(DATASET_PATH)
  .pipe(csv())
  .on("data", (row) => {
    try {
      const features = featureNames.map((name) => parseFloat(row[name]));
      const target = parseFloat(row[targetName]);

      const validFeatures = features.every((v) => !isNaN(v));
      const validTarget = !isNaN(target);

      if (validFeatures && validTarget) {
        rows.push({
          x: features,
          y: target
        });
      }
    } catch (err) {
      // skip bad rows
    }
  })
  .on("end", () => {
    console.log(`Dataset loaded: ${rows.length} valid rows`);

    if (rows.length === 0) {
      console.log("Dataset reading failed. Check column names or missing numeric data.");
      return;
    }

    const X = rows.map((r) => r.x);
    const Y = rows.map((r) => r.y);

    // normalize
    const means = [];
    const stds = [];

    for (let j = 0; j < featureNames.length; j++) {
      const col = X.map((row) => row[j]);
      const m = mean(col);
      const s = std(col, m);
      means.push(m);
      stds.push(s);
    }

    const Xnorm = X.map((row) =>
      row.map((value, j) => (value - means[j]) / stds[j])
    );

    // split train/test
    const splitIndex = Math.floor(Xnorm.length * 0.8);

    const XTrain = Xnorm.slice(0, splitIndex);
    const YTrain = Y.slice(0, splitIndex);

    const XTest = Xnorm.slice(splitIndex);
    const YTest = Y.slice(splitIndex);

    // manual logistic regression
    let weights = new Array(featureNames.length).fill(0);
    let bias = 0;

    const learningRate = 0.01;
    const epochs = 2000;
    const n = XTrain.length;

    for (let epoch = 0; epoch < epochs; epoch++) {
      let weightGradients = new Array(featureNames.length).fill(0);
      let biasGradient = 0;

      for (let i = 0; i < n; i++) {
        const prediction = sigmoid(dot(XTrain[i], weights) + bias);
        const error = prediction - YTrain[i];

        for (let j = 0; j < featureNames.length; j++) {
          weightGradients[j] += error * XTrain[i][j];
        }

        biasGradient += error;
      }

      for (let j = 0; j < featureNames.length; j++) {
        weights[j] -= (learningRate * weightGradients[j]) / n;
      }

      bias -= (learningRate * biasGradient) / n;

      if ((epoch + 1) % 200 === 0) {
        let loss = 0;
        for (let i = 0; i < n; i++) {
          const p = sigmoid(dot(XTrain[i], weights) + bias);
          loss += -(
            YTrain[i] * Math.log(p + 1e-9) +
            (1 - YTrain[i]) * Math.log(1 - p + 1e-9)
          );
        }
        loss /= n;
        console.log(`Epoch ${epoch + 1}/${epochs} - Loss: ${loss.toFixed(6)}`);
      }
    }

    // evaluate
    let correct = 0;
    for (let i = 0; i < XTest.length; i++) {
      const probability = sigmoid(dot(XTest[i], weights) + bias);
      const prediction = probability >= 0.5 ? 1 : 0;
      if (prediction === YTest[i]) {
        correct++;
      }
    }

    const accuracy = correct / YTest.length;
    console.log("Model trained successfully");
    console.log(`Accuracy: ${(accuracy * 100).toFixed(2)}%`);

    const modelToSave = {
      featureNames,
      means,
      stds,
      weights,
      bias
    };

    fs.writeFileSync(MODEL_PATH, JSON.stringify(modelToSave, null, 2));
    console.log(`Model saved to ${MODEL_PATH}`);
  });

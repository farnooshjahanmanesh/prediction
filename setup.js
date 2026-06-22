const axios = require("axios");
const fs = require("fs");
const unzipper = require("unzipper");

async function downloadDataset() {
  const url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip";

  console.log("در حال دانلود دیتاست بیمارستانی...");

  try {
    const response = await axios({
      method: "GET",
      url: url,
      responseType: "stream",
    });

    const writer = fs.createWriteStream("data.zip");
    response.data.pipe(writer);

    writer.on("finish", () => {
      console.log("دانلود تمام شد. در حال استخراج فایل...");

      fs.createReadStream("data.zip")
        .pipe(unzipper.Extract({ path: "./dataset" }))
        .on("close", () => {
          console.log("✅ دیتاست با موفقیت داخل پوشه dataset استخراج شد.");
        });
    });

    writer.on("error", (err) => {
      console.error("خطا در ذخیره فایل:", err.message);
    });
  } catch (error) {
    console.error("خطا در دانلود دیتاست:", error.message);
  }
}

downloadDataset();

import express from "express";
import cors from "cors";
import { writeFile, mkdir } from "fs/promises";
import path from "path";

const app = express();

// CORS
app.use(cors({
  origin: "http://localhost:5173",
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type"]
}));

app.use(express.json({ limit: "100mb" }));

// 🔹 TEST endpoint
app.post("/api/sum", (req, res) => {
  const { a, b } = req.body;
  res.json({ result: a + b });
});

// 🔹 WRITE FILE endpoint
app.post("/api/write", async (req, res) => {
  try {
    const { filename, content } = req.body;

    if (!filename || content === undefined) {
      return res.status(400).json({ error: "filename and content required" });
    }

    // Ensure directory exists
    await mkdir("data", { recursive: true });

    // IMPORTANT: lock writes to /data only
    const filePath = path.resolve("data", filename);

    await writeFile(filePath, content, "utf8");

    res.json({ success: true, path: filePath });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to write file" });
  }
});

app.listen(3000, () => {
  console.log("Server running at http://localhost:3000");
});
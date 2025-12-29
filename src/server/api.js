// api.js — BINARY ONLY
const express = require("express");
const cors = require("cors");
const { writeFile, mkdir } = require("fs/promises");
const path = require("path");

const app = express();

app.use(cors({
  origin: "http://localhost:5173",
  methods: ["POST", "OPTIONS"],
  allowedHeaders: ["Content-Type"]
}));

// ❗ DO NOT use express.json() ANYWHERE
// ❗ DO NOT destructure filename/content

app.post(
  "/api/write",
  express.raw({ limit: "200mb", type: "*/*" }),
  async (req, res) => {
    try {
      if (!req.body || req.body.length === 0) {
        return res.status(400).json({ error: "Empty body" });
      }

      const dataDir = path.resolve("data");
      await mkdir(dataDir, { recursive: true });

      const filePath = path.join(dataDir, "BVH2.bin");
      await writeFile(filePath, Buffer.from(req.body));

      res.json({
        success: true,
        bytesWritten: req.body.length
      });
    } catch (err) {
      console.error(err);
      res.status(500).json({ error: "Write failed" });
    }
  }
);

app.listen(3000, () => {
  console.log("API server running at http://localhost:3000");
});
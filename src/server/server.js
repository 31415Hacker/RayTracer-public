const express = require('express');
const path = require('path');
const app = express();

const PORT = process.env.PORT || 3000;

// Serve all static files under 'public' directory
app.use(express.static(path.join(__dirname, '..', 'public')));

// Serve 'public/app/index.html' on root request
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'public', 'app', 'index.html'));
});

app.get('/debug', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'public', 'app', 'debug.html'));
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
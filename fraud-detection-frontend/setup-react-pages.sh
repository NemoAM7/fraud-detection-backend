#!/bin/bash

# Define base paths
SRC_DIR="src"
COMPONENTS_DIR="$SRC_DIR/components"
PAGES_DIR="$SRC_DIR/pages"

# Ensure the directories exist
mkdir -p "$COMPONENTS_DIR" "$PAGES_DIR"

# Create components
declare -A components=(
    ["BatchUpload.jsx"]='import React from "react";

export default function BatchUpload() {
  return <div className="p-6 bg-gray-800 rounded-lg text-white">Batch Upload Component</div>;
}'
    ["FraudGraphs.jsx"]='import React from "react";

export default function FraudGraphs() {
  return <div className="p-6 bg-gray-800 rounded-lg text-white">Fraud Graphs Component</div>;
}'
    ["FraudReport.jsx"]='import React from "react";

export default function FraudReport() {
  return <div className="p-6 bg-gray-800 rounded-lg text-white">Fraud Report Component</div>;
}'
    ["RuleConfig.jsx"]='import React from "react";

export default function RuleConfig() {
  return <div className="p-6 bg-gray-800 rounded-lg text-white">Rule Config Component</div>;
}'
)

# Write component files
for file in "${!components[@]}"; do
    echo "${components[$file]}" > "$COMPONENTS_DIR/$file"
    echo "Created $COMPONENTS_DIR/$file"
done

# Create page files
declare -A pages=(
    ["BatchUploadPage.jsx"]='import BatchUpload from "../components/BatchUpload";

export default function BatchUploadPage() {
  return <div className="p-6"><BatchUpload /></div>;
}'
    ["FraudGraphsPage.jsx"]='import FraudGraphs from "../components/FraudGraphs";

export default function FraudGraphsPage() {
  return <div className="p-6"><FraudGraphs /></div>;
}'
    ["FraudReportPage.jsx"]='import FraudReport from "../components/FraudReport";

export default function FraudReportPage() {
  return <div className="p-6"><FraudReport /></div>;
}'
    ["RuleConfigPage.jsx"]='import RuleConfig from "../components/RuleConfig";

export default function RuleConfigPage() {
  return <div className="p-6"><RuleConfig /></div>;
}'
)

# Write page files
for file in "${!pages[@]}"; do
    echo "${pages[$file]}" > "$PAGES_DIR/$file"
    echo "Created $PAGES_DIR/$file"
done

# Create App.jsx with routing
cat <<EOL > "$SRC_DIR/App.jsx"
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import BatchUploadPage from "./pages/BatchUploadPage";
import FraudGraphsPage from "./pages/FraudGraphsPage";
import FraudReportPage from "./pages/FraudReportPage";
import RuleConfigPage from "./pages/RuleConfigPage";

export default function App() {
  return (
    <Router>
      <div className="p-6 bg-gray-900 min-h-screen text-white">
        <nav className="mb-4">
          <ul className="flex space-x-4">
            <li><Link to="/batch-upload" className="text-blue-400">Batch Upload</Link></li>
            <li><Link to="/fraud-graphs" className="text-blue-400">Fraud Graphs</Link></li>
            <li><Link to="/fraud-report" className="text-blue-400">Fraud Report</Link></li>
            <li><Link to="/rule-config" className="text-blue-400">Rule Config</Link></li>
          </ul>
        </nav>

        <Routes>
          <Route path="/batch-upload" element={<BatchUploadPage />} />
          <Route path="/fraud-graphs" element={<FraudGraphsPage />} />
          <Route path="/fraud-report" element={<FraudReportPage />} />
          <Route path="/rule-config" element={<RuleConfigPage />} />
        </Routes>
      </div>
    </Router>
  );
}
EOL
echo "Created $SRC_DIR/App.jsx"

# Create main.jsx
cat <<EOL > "$SRC_DIR/main.jsx"
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
EOL
echo "Created $SRC_DIR/main.jsx"

echo "✅ React project structure set up successfully!"

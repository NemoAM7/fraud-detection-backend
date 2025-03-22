import { BrowserRouter as Router, Routes, Route, NavLink } from "react-router-dom";
import BatchUploadPage from "./pages/BatchUploadPage";
import FraudGraphsPage from "./pages/FraudGraphsPage";
import FraudReportPage from "./pages/FraudReportPage";
import RuleConfigPage from "./pages/RuleConfigPage";
import FraudFormPage from "./pages/FraudFormPage";
import Dashboard from "./pages/Dashboard";
import { Colors } from "chart.js";

export default function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-900 text-gray-100">
        {/* Navbar */}
        <nav className="bg-black border-b border-gray-800 py-3 sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex justify-between items-center">
            <div className="flex items-center">
              <h1 className="text-xl font-medium text-white">
                <span className="text-blue-500">Fraud</span>Detection
              </h1>
            </div>
            <ul className="flex space-x-8">
              <li>
                <NavLink
                  to="/dashboard"
                  className={({ isActive }) =>
                    `text-sm font-medium transition duration-150 ease-in-out ${
                      isActive 
                        ? "text-blue-400 border-b-2 border-blue-400 pb-2" 
                        : "text-gray-400 hover:text-blue-400"
                    }`
                  }
                >
                  Dashboard
                </NavLink>
              </li>
            </ul>
          </div>
        </nav>

        {/* Page Content */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/batch-upload" element={<BatchUploadPage />} />
            <Route path="/fraud-graphs" element={<FraudGraphsPage />} />
            <Route path="/fraud-report" element={<FraudReportPage />} />
            <Route path="/rule-config" element={<RuleConfigPage />} />
            <Route path="/fraud-form" element={<FraudFormPage />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

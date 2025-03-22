import { useState } from "react";
import { batchDetectFraud, reportFraud } from "../api";

export default function BatchUpload() {
  const [transactions, setTransactions] = useState([]);
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [autoReport, setAutoReport] = useState(true);
  const [reportStatus, setReportStatus] = useState({});

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result);
        
        // Validate the uploaded JSON format
        if (Array.isArray(data)) {
          // Make sure each transaction has required fields
          const validTransactions = data.map(tx => {
            return {
              transaction_id: tx.transaction_id || `tx-${Math.random().toString(36).substring(2, 9)}`,
              transaction_amount: parseFloat(tx.transaction_amount || tx.amount || 0),
              payer_id: tx.payer_id || "",
              payee_id: tx.payee_id || "",
              transaction_date: tx.transaction_date || new Date().toISOString().split('T')[0],
              transaction_channel: tx.transaction_channel || tx.channel || ""
            };
          });
          setTransactions(validTransactions);
        } else {
          alert("Invalid format. Expected an array of transactions.");
        }
      } catch (error) {
        alert("Invalid JSON file: " + error.message);
      }
    };
    reader.readAsText(file);
  };

  const handleSubmit = async () => {
    if (!transactions.length) return;
    
    setIsLoading(true);
    setReportStatus({});
    try {
      const response = await batchDetectFraud(transactions);
      setResults(response.data.results);
      
      // Auto-report fraudulent transactions if enabled
      if (autoReport) {
        await handleAutoReportBatch(response.data.results);
      }
    } catch (error) {
      console.error("Error processing batch:", error);
      alert("Error processing transactions. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleAutoReportBatch = async (results) => {
    const fraudulentTransactions = Object.entries(results)
      .filter(([_, result]) => result.is_fraud)
      .map(([transactionId, result]) => ({
        transactionId, 
        reason: result.fraud_reason,
        score: result.fraud_score
      }));
    
    if (fraudulentTransactions.length === 0) return;
    
    // Create a status object to track reporting for each transaction
    const newReportStatus = { ...reportStatus };
    
    // Report each fraudulent transaction
    await Promise.all(
      fraudulentTransactions.map(async ({ transactionId, reason, score }) => {
        try {
          const reportData = {
            transaction_id: transactionId,
            reporting_entity_id: "SYSTEM_BATCH_AUTO_REPORT",
            fraud_details: `Batch auto-reported fraud. Reason: ${reason}. Score: ${score}`
          };
          
          const response = await reportFraud(reportData);
          
          newReportStatus[transactionId] = {
            success: response.data.reporting_acknowledged,
            message: response.data.reporting_acknowledged 
              ? "Automatically reported" 
              : "Failed to report"
          };
        } catch (error) {
          console.error(`Error reporting transaction ${transactionId}:`, error);
          newReportStatus[transactionId] = {
            success: false,
            message: "Error reporting"
          };
        }
      })
    );
    
    setReportStatus(newReportStatus);
  };

  const handleManualReport = async (transactionId, reason, score) => {
    try {
      const reportData = {
        transaction_id: transactionId,
        reporting_entity_id: "MANUAL_BATCH_REPORT",
        fraud_details: `Manually reported from batch. Reason: ${reason}. Score: ${score}`
      };
      
      const response = await reportFraud(reportData);
      
      setReportStatus(prev => ({
        ...prev,
        [transactionId]: {
          success: response.data.reporting_acknowledged,
          message: response.data.reporting_acknowledged 
            ? "Manually reported" 
            : "Failed to report"
        }
      }));
    } catch (error) {
      console.error(`Error reporting transaction ${transactionId}:`, error);
      setReportStatus(prev => ({
        ...prev,
        [transactionId]: {
          success: false,
          message: "Error reporting"
        }
      }));
    }
  };

  return (
    <div className="max-w-full mx-auto">
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <p className="text-sm text-gray-300">
            Upload a JSON file containing transaction data for batch processing
          </p>
          <div className="flex items-center">
            <label className="inline-flex items-center cursor-pointer">
              <input 
                type="checkbox" 
                checked={autoReport} 
                onChange={() => setAutoReport(!autoReport)}
                className="sr-only peer" 
              />
              <div className="relative w-11 h-6 bg-gray-600 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-400 rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-500"></div>
              <span className="ms-3 text-sm font-medium text-gray-300">Auto-report fraud</span>
            </label>
          </div>
        </div>

        {/* File Upload Section */}
        <div className="mb-4">
          <label 
            htmlFor="file-upload" 
            className="block w-full p-4 border border-gray-600 border-dashed rounded-md text-center cursor-pointer hover:bg-gray-800"
          >
            <div className="flex flex-col items-center justify-center">
              <span className="text-gray-400 mb-2">Drop JSON file here or click to browse</span>
              <span className="text-sm text-gray-500">(.json)</span>
            </div>
            <input
              id="file-upload"
              type="file"
              accept=".json"
              onChange={handleFileUpload}
              className="hidden"
            />
          </label>
        </div>

        {/* File Info */}
        {transactions.length > 0 && (
          <div className="mb-4 p-3 bg-blue-900 text-blue-200 rounded-md text-sm">
            {transactions.length} transactions loaded and ready for processing
          </div>
        )}

        {/* Submit Button */}
        <div className="text-center">
        <button
          onClick={handleSubmit}
          disabled={!transactions.length || isLoading}
          className={`w-1/3 px-4 py-2 rounded-md text-sm transition-colors ${
            transactions.length && !isLoading
              ? "bg-blue-600 text-white hover:bg-blue-700"
              : "bg-gray-700 text-gray-400 cursor-not-allowed"
          }`}
        >
          {isLoading ? "Processing..." : "Process Transactions"}
        </button>
        </div>
      </div>

      {/* Results Table */}
      {results && Object.keys(results).length > 0 && (
        <div className="mt-6 overflow-x-auto">
          <table className="w-full text-sm text-gray-300 border-collapse">
            <thead>
              <tr className="bg-gray-800">
                <th className="px-4 py-2 border-b border-gray-700 text-left font-medium">ID</th>
                <th className="px-4 py-2 border-b border-gray-700 text-left font-medium">Status</th>
                <th className="px-4 py-2 border-b border-gray-700 text-left font-medium">Reason</th>
                <th className="px-4 py-2 border-b border-gray-700 text-left font-medium">Score</th>
                <th className="px-4 py-2 border-b border-gray-700 text-left font-medium">Report Status</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(results).map(([id, result]) => (
                <tr key={id} className="hover:bg-gray-800">
                  <td className="px-4 py-3 border-b border-gray-700">{id}</td>
                  <td className="px-4 py-3 border-b border-gray-700">
                    <span
                      className={`inline-flex px-2 py-1 text-xs rounded-full ${
                        result.is_fraud 
                          ? "bg-red-900 text-red-200" 
                          : "bg-green-900 text-green-200"
                      }`}
                    >
                      {result.is_fraud ? "Flagged" : "Clear"}
                    </span>
                  </td>
                  <td className="px-4 py-3 border-b border-gray-700">{result.fraud_reason || "—"}</td>
                  <td className="px-4 py-3 border-b border-gray-700">{result.fraud_score}</td>
                  <td className="px-4 py-3 border-b border-gray-700">
                    {result.is_fraud ? (
                      reportStatus[id] ? (
                        <span className={`inline-flex px-2 py-1 text-xs rounded-full ${
                          reportStatus[id].success 
                            ? "bg-green-900 text-green-200" 
                            : "bg-red-900 text-red-200"
                        }`}>
                          {reportStatus[id].message}
                        </span>
                      ) : (
                        <button
                          onClick={() => handleManualReport(id, result.fraud_reason, result.fraud_score)}
                          className="px-2 py-1 text-xs bg-blue-700 text-white rounded hover:bg-blue-800 transition-colors"
                        >
                          Report
                        </button>
                      )
                    ) : (
                      <span className="text-gray-500">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
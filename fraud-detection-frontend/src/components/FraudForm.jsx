import { useState } from "react";
import { detectFraud, reportFraud } from "../api";

export default function FraudForm() {
  const [transaction, setTransaction] = useState({
    transaction_id: "",
    transaction_amount: "",
    transaction_date: "",
    transaction_channel: "",
    payer_id: "",
    payee_id: "",
  });
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [autoReport, setAutoReport] = useState(true);
  const [reportStatus, setReportStatus] = useState(null);

  const handleChange = (e) => {
    setTransaction({ ...transaction, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setReportStatus(null);
    try {
      const transactionData = {
        ...transaction,
        transaction_amount: parseFloat(transaction.transaction_amount)
      };
      const response = await detectFraud(transactionData);
      setResult(response.data);
      
      // Auto-report if enabled and transaction is detected as fraudulent
      if (autoReport && response.data.is_fraud) {
        await handleAutoReport(response.data);
      }
    } catch (error) {
      console.error("Error detecting fraud:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAutoReport = async (fraudResult) => {
    try {
      const reportData = {
        transaction_id: transaction.transaction_id,
        reporting_entity_id: "SYSTEM_AUTO_REPORT",
        fraud_details: `Auto-reported fraud. Reason: ${fraudResult.fraud_reason}. Score: ${fraudResult.fraud_score}`
      };
      
      const response = await reportFraud(reportData);
      
      if (response.data.reporting_acknowledged) {
        setReportStatus({
          success: true,
          message: "Transaction automatically reported as fraud"
        });
      } else {
        setReportStatus({
          success: false,
          message: "Failed to auto-report transaction"
        });
      }
    } catch (error) {
      console.error("Error auto-reporting fraud:", error);
      setReportStatus({
        success: false,
        message: "Error while auto-reporting fraud"
      });
    }
  };

  const handleManualReport = async () => {
    if (!result || !transaction.transaction_id) return;
    
    setIsLoading(true);
    try {
      const reportData = {
        transaction_id: transaction.transaction_id,
        reporting_entity_id: "MANUAL_USER_REPORT",
        fraud_details: `Manually reported fraud after detection. Reason: ${result.fraud_reason}. Score: ${result.fraud_score}`
      };
      
      const response = await reportFraud(reportData);
      
      if (response.data.reporting_acknowledged) {
        setReportStatus({
          success: true,
          message: "Transaction manually reported as fraud"
        });
      } else {
        setReportStatus({
          success: false,
          message: "Failed to report transaction"
        });
      }
    } catch (error) {
      console.error("Error reporting fraud:", error);
      setReportStatus({
        success: false,
        message: "Error while reporting fraud"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const fieldLabels = {
    transaction_id: "Transaction ID",
    transaction_amount: "Amount",
    transaction_date: "Transaction Date",
    transaction_channel: "Channel",
    payer_id: "Payer ID",
    payee_id: "Payee ID",
  };

  const fieldTypes = {
    transaction_id: "text",
    transaction_amount: "number",
    transaction_date: "date",
    transaction_channel: "text",
    payer_id: "text",
    payee_id: "text",
  };

  return (
    <div className="max-w-full mx-auto">
      <div className="flex justify-between items-center mb-4">
        <p className="text-sm text-gray-300">
          Enter transaction details to check for potential fraud
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
      
      <form onSubmit={handleSubmit} className="space-y-3">
        {Object.keys(transaction).map((field) => (
          <div key={field} className="w-full">
            <label htmlFor={field} className="block text-xs text-gray-300 mb-1">
              {fieldLabels[field]}
            </label>
            <input
              id={field}
              type={fieldTypes[field]}
              name={field}
              value={transaction[field]}
              onChange={handleChange}
              className="w-full p-2 bg-gray-700 text-gray-100 rounded-md border border-gray-600 focus:border-blue-400 focus:outline-none focus:ring-1 focus:ring-blue-400"
              required={field === 'transaction_id' || field === 'transaction_amount'}
            />
          </div>
        ))}
        
        <div className="text-center">
          <button
            type="submit"
            disabled={isLoading}
            className="w-1/3 text-center mt-4 px-4 py-2 text-md rounded-md bg-blue-600 text-white hover:bg-blue-700 transition-colors focus:outline-none disabled:bg-blue-800 disabled:text-gray-200"
          >
            {isLoading ? "Processing..." : "Check Transaction"}
          </button>
        </div>
      </form>

      {result && (
        <div className="mt-6 p-4 rounded-md bg-gray-800 border border-gray-700">
          <div className="flex items-center justify-between mb-3">
            <span
              className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${
                result.is_fraud 
                  ? "bg-red-900 text-red-200" 
                  : "bg-green-900 text-green-200"
              }`}
            >
              {result.is_fraud ? "Potentially Fraudulent" : "Likely Legitimate"}
            </span>
            
            {result.is_fraud && !reportStatus && (
              <button
                onClick={handleManualReport}
                disabled={isLoading}
                className="px-3 py-1 text-xs rounded bg-red-700 text-white hover:bg-red-800 transition-colors"
              >
                Report Manually
              </button>
            )}
          </div>
          
          {reportStatus && (
            <div className={`mb-3 px-3 py-2 rounded text-sm ${
              reportStatus.success ? "bg-green-900 text-green-200" : "bg-red-900 text-red-200"
            }`}>
              {reportStatus.message}
            </div>
          )}
          
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-300">Reason:</span>
              <span className="text-gray-300">{result.fraud_reason || "—"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-300">Risk Score:</span>
              <span className="text-gray-300">{result.fraud_score}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-300">Detection Source:</span>
              <span className="text-gray-300">{result.fraud_source}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
import { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";
import FraudForm from "../components/FraudForm";
import FraudStats from "../components/FraudStats";
import FraudGraphs from "../components/FraudGraphs";
import RuleConfig from "../components/RuleConfig";
import BatchUpload from "../components/BatchUpload";
import FraudReport from "../components/FraudReport";
import { getFraudTrend, getFraudByDimension, getModelConfig, updateModelConfig, getTransactions } from "../api";

export default function Dashboard() {
  const location = useLocation();
  const [filters, setFilters] = useState({ 
    date: "", 
    payerId: "", 
    payeeId: "",
    dimension: "transaction_channel",
    transactionId: ""
  });
  const [activeTab, setActiveTab] = useState("stats");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState([]);
  const [trendData, setTrendData] = useState([]);
  const [modelConfig, setModelConfig] = useState(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [modelChangeStatus, setModelChangeStatus] = useState(null);
  const [transactionData, setTransactionData] = useState(null);
  const [transactionLoading, setTransactionLoading] = useState(false);
  const [samplingMethod, setSamplingMethod] = useState("random");
  const [evaluationPeriod, setEvaluationPeriod] = useState("30d");
  
  const [evaluationMetrics, setEvaluationMetrics] = useState({
    confusion_matrix: {
      true_positive: 78,
      false_positive: 12,
      true_negative: 889,
      false_negative: 21
    },
    precision: 0.867,
    recall: 0.788,
    f1_score: 0.825,
    accuracy: 0.967
  });

  const dimensions = [
    { value: "transaction_channel", label: "Transaction Channel" },
    { value: "transaction_payment_mode", label: "Transaction Payment Mode" },
    { value: "payment_gateway", label: "Payment Gateway" },
    { value: "bank", label: "Bank" },
    { value: "payer_id", label: "Payer ID" },
    { value: "payee_id", label: "Payee ID" }
  ];

  const today = new Date();
  const thirtyDaysAgo = new Date(today);
  thirtyDaysAgo.setDate(today.getDate() - 30);
  today.setDate(today.getDate() + 1);
  
  const formattedToday = today.toISOString().split('T')[0];
  const formattedThirtyDaysAgo = thirtyDaysAgo.toISOString().split('T')[0];

  useEffect(() => {
    if (location.state?.refreshData) {
      console.log("Dashboard - Refreshing data based on navigation state");
      window.history.replaceState({}, document.title);
      setRefreshTrigger(prev => prev + 1);
      setActiveTab("stats");
    }
  }, [location]);

  useEffect(() => {
    const fetchTransactionById = async () => {
      if (!filters.transactionId) {
        setTransactionData(null);
        return;
      }
      
      setTransactionLoading(true);
      try {
        const response = await getTransactions({ 
          transaction_id: filters.transactionId 
        });
        
        console.log("Transaction search response:", response);
        
        if (response.data && response.data.transactions && response.data.transactions.length > 0) {
          setTransactionData(response.data.transactions[0]);
        } else {
          setTransactionData(null);
        }
      } catch (error) {
        console.error("Error fetching transaction:", error);
        setTransactionData(null);
      } finally {
        setTransactionLoading(false);
      }
    };

    if (filters.transactionId) {
      fetchTransactionById();
    }
  }, [filters.transactionId, refreshTrigger]);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        console.log("Dashboard - Fetching data with filters:", filters);
        
        try {
          const dimensionResponse = await getFraudByDimension(filters.dimension, {
            start_date: filters.date || formattedThirtyDaysAgo,
            end_date: formattedToday
          });
          
          console.log(`${filters.dimension} API response:`, dimensionResponse);
          
          if (dimensionResponse.data && dimensionResponse.data.length > 0) {
            const statData = dimensionResponse.data.map(item => ({
              label: item.dimension_value || "Unknown",
              value: item.predicted_fraud || 0,
              reportedValue: item.reported_fraud || 0,
              total: item.total || 0
            }));
            setStats(statData);
          } else {
            console.warn(`No ${filters.dimension} data available`);
            setStats([]);
          }
        } catch (dimError) {
          console.error(`Error fetching ${filters.dimension} data:`, dimError);
          setStats([]);
        }

        console.log(formattedThirtyDaysAgo, formattedToday);
        try {
          const trendResponse = await getFraudTrend(
            filters.date || formattedThirtyDaysAgo,
            formattedToday,
            'day'
          );
          
          console.log("Trend API response:", trendResponse);
          
          if (trendResponse.data && trendResponse.data.length > 0) {
            setTrendData(trendResponse.data);
          } else {
            console.warn("No trend data available");
            setTrendData([]);
          }
        } catch (trendError) {
          console.error("Error fetching trend data:", trendError);
          setTrendData([]);
        }

        try {
          const modelResponse = await getModelConfig();
          if (modelResponse.data) {
            setModelConfig(modelResponse.data);
          }
        } catch (modelError) {
          console.error("Error fetching model config:", modelError);
        }
      } catch (err) {
        console.error("Error fetching dashboard data:", err);
        setError("Failed to load dashboard data. Please try again later.");
      } finally {
        setLoading(false);
      }
    };

    if (!filters.transactionId) {
      fetchData();
    }
  }, [filters.date, filters.dimension, refreshTrigger]);

  const handleRefreshData = () => {
    console.log("Manual refresh triggered");
    setRefreshTrigger(prev => prev + 1);
  };

  const handleApplyFilters = () => {
    console.log("Applying filters:", filters);
    setRefreshTrigger(prev => prev + 1);
  };

  const handleClearTransactionSearch = () => {
    setFilters(prev => ({ ...prev, transactionId: "" }));
    setTransactionData(null);
  };

  const handleChangeModel = async (modelType) => {
    // try {
    //   setModelChangeStatus({ message: "Switching model...", type: "info" });
    //   const response = await updateModelConfig({ model_type: modelType });
      
    //   if (response.data && response.data.status === "success") {
    //     setModelChangeStatus({ message: response.data.message, type: "success" });
    //     const modelResponse = await getModelConfig();
    //     if (modelResponse.data) {
    //       setModelConfig(modelResponse.data);
    //     }
        
    //     setTimeout(() => {
    //       setModelChangeStatus(null);
    //     }, 3000);
    //   } else {
    //     setModelChangeStatus({ 
    //       message: response.data?.message || "Failed to switch model", 
    //       type: "error" 
    //     });
    //   }
    // } catch (error) {
    //   console.error("Error changing model:", error);
    //   setModelChangeStatus({ 
    //     message: "Error changing model. Please try again.", 
    //     type: "error" 
    //   });
    // }
  };

  const handleSamplingMethodChange = (method) => {
    setSamplingMethod(method);
    console.log("Sampling method changed to:", method);
  };

  const samplingMethods = [
    { value: "random", label: "Random Sampling" },
    { value: "stratified", label: "Stratified Sampling" },
    { value: "systematic", label: "Systematic Sampling" }
  ];

  const tabs = [
    { id: "stats", label: "Overview" },
    { id: "single", label: "Single Transaction" },
    { id: "bulk", label: "Batch Processing" },
    { id: "report", label: "Report Fraud" },
    { id: "rules", label: "Rule Engine" },
  ];

  const renderTransactionDetails = () => {
    if (transactionLoading) {
      return <div className="py-6 text-center text-gray-300">Searching for transaction...</div>;
    }
    
    if (!transactionData) {
      return <div className="py-6 text-center text-gray-400">No transaction found with ID: {filters.transactionId}</div>;
    }
    
    const transaction = JSON.parse(transactionData.transaction_data || "{}");
    const isFraud = transactionData.is_fraud_predicted;
    const isFraudReported = transactionData.is_fraud_reported;
    
    return (
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <h3 className="text-base font-medium text-white">Transaction Details</h3>
          <span className={`px-2 py-1 text-xs font-medium rounded-full ${
            isFraud ? "bg-red-900 text-red-200" : "bg-green-900 text-green-200"
          }`}>
            {isFraud ? "Flagged as Fraud" : "Legitimate Transaction"}
          </span>
        </div>
        
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <p className="text-gray-400">Transaction ID:</p>
            <p className="text-white">{transactionData.transaction_id}</p>
          </div>
          <div>
            <p className="text-gray-400">Date:</p>
            <p className="text-white">{new Date(transactionData.created_at).toLocaleString()}</p>
          </div>
          <div>
            <p className="text-gray-400">Amount:</p>
            <p className="text-white">${transaction.transaction_amount || 'N/A'}</p>
          </div>
          <div>
            <p className="text-gray-400">Channel:</p>
            <p className="text-white">{transaction.transaction_channel || 'N/A'}</p>
          </div>
          <div>
            <p className="text-gray-400">Payment Mode:</p>
            <p className="text-white">{transaction.transaction_payment_mode || 'N/A'}</p>
          </div>
          <div>
            <p className="text-gray-400">Payment Gateway:</p>
            <p className="text-white">{transaction.payment_gateway || 'N/A'}</p>
          </div>
          <div>
            <p className="text-gray-400">Payer ID:</p>
            <p className="text-white">{transaction.payer_id || 'N/A'}</p>
          </div>
          <div>
            <p className="text-gray-400">Payee ID:</p>
            <p className="text-white">{transaction.payee_id || 'N/A'}</p>
          </div>
        </div>
        
        <div className="bg-gray-700 p-3 rounded-md mt-4">
          <div className="flex justify-between items-center mb-2">
            <h4 className="text-sm font-medium text-gray-200">Fraud Analysis</h4>
            {isFraudReported && (
              <span className="px-2 py-1 text-xs rounded-full bg-red-900 text-red-200">
                Reported as Fraud
              </span>
            )}
          </div>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <p className="text-gray-400">ML Prediction:</p>
              <p className="text-white">{isFraud ? "Fraud" : "Not Fraud"}</p>
            </div>
            <div>
              <p className="text-gray-400">Source:</p>
              <p className="text-white">{transactionData.fraud_source || 'N/A'}</p>
            </div>
            <div>
              <p className="text-gray-400">Score:</p>
              <p className="text-white">{transactionData.fraud_score?.toFixed(2) || 'N/A'}</p>
            </div>
            <div>
              <p className="text-gray-400">Reason:</p>
              <p className="text-white">{transactionData.fraud_reason || 'N/A'}</p>
            </div>
          </div>
        </div>
        
        <button
          onClick={handleClearTransactionSearch}
          className="mt-4 px-4 py-2 bg-gray-700 text-gray-200 rounded-md text-sm hover:bg-gray-600 focus:outline-none transition-colors"
        >
          Back to Statistics
        </button>
      </div>
    );
  };

  const calculateMetrics = () => {
    const { true_positive, false_positive, true_negative, false_negative } = 
      evaluationMetrics.confusion_matrix;
    
    const total = true_positive + false_positive + true_negative + false_negative;
    const precision = true_positive / (true_positive + false_positive);
    const recall = true_positive / (true_positive + false_negative);
    const f1 = 2 * (precision * recall) / (precision + recall);
    const accuracy = (true_positive + true_negative) / total;
    
    return {
      precision: precision.toFixed(3),
      recall: recall.toFixed(3),
      f1: f1.toFixed(3),
      accuracy: accuracy.toFixed(3)
    };
  };

  const metrics = calculateMetrics();

  const renderTabContent = () => {
    switch (activeTab) {
      case "stats":
        return (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-gray-800 rounded-lg shadow-md border border-gray-700 p-6">
              {filters.transactionId ? (
                <h2 className="text-lg font-medium text-gray-100 mb-4">
                  Transaction Search Result
                </h2>
              ) : (
                <h2 className="text-lg font-medium text-gray-100 mb-4">
                  Fraud Statistics by {dimensions.find(d => d.value === filters.dimension)?.label}
                </h2>
              )}
              
              {filters.transactionId ? (
                renderTransactionDetails()
              ) : loading ? (
                <p className="text-gray-300">Loading statistics...</p>
              ) : error ? (
                <p className="text-red-400">{error}</p>
              ) : stats.length === 0 ? (
                <div className="text-center py-6">
                  <p className="text-gray-400 mb-4">No fraud statistics available yet</p>
                  <p className="text-sm text-gray-500">Use the "Report Fraud" tab to register suspicious transactions</p>
                  <button
                    onClick={handleRefreshData}
                    className="mt-4 px-4 py-2 bg-gray-700 text-white rounded-md text-sm hover:bg-gray-600 transition-colors"
                  >
                    Refresh Data
                  </button>
                </div>
              ) : (
                <FraudStats data={stats} />
              )}
            </div>
            <div className="bg-gray-800 rounded-lg shadow-md border border-gray-700 p-6">
              <h2 className="text-lg font-medium text-gray-100 mb-4">Fraud Analysis</h2>
              {filters.transactionId ? (
                <div className="flex items-center justify-center h-full">
                  <p className="text-gray-400">Transaction search active. Clear search to view trend analysis.</p>
                </div>
              ) : (
                <>
                  <FraudGraphs fraudData={{
                    trends: trendData,
                    categories: stats,
                    dimensionType: filters.dimension
                  }} isLoading={loading} />
                  {!loading && trendData.length === 0 && (
                    <div className="text-center mt-4">
                      <button
                        onClick={handleRefreshData}
                        className="px-4 py-2 bg-gray-700 text-white rounded-md text-sm hover:bg-gray-600 transition-colors"
                      >
                        Refresh Data
                      </button>
                    </div>
                  )}
                </>
              )}
            </div>
            
            <div className="bg-gray-800 rounded-lg shadow-md border border-gray-700 p-6 lg:col-span-2">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-medium text-gray-100">Model Evaluation</h2>
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-300">Time Period:</span>
                  <select
                    value={evaluationPeriod}
                    onChange={(e) => setEvaluationPeriod(e.target.value)}
                    className="bg-gray-900 text-gray-200 text-sm rounded-md px-2 py-1 border border-gray-600 focus:outline-none"
                  >
                    <option value="7d">Last 7 Days</option>
                    <option value="30d">Last 30 Days</option>
                    <option value="90d">Last 90 Days</option>
                  </select>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-gray-700 rounded-lg p-4">
                  <h3 className="text-sm font-medium text-gray-200 mb-3">Confusion Matrix</h3>
                  <div className="flex justify-center">
                    <div className="max-w-xs">
                      <div className="mb-3 text-center text-xs text-gray-400">
                        <span>Predicted vs Actual</span>
                      </div>
                      
                      <div className="grid grid-cols-3 text-center">
                        <div className="col-span-1"></div>
                        <div className="col-span-1 text-xs text-gray-400 p-2">Fraud</div>
                        <div className="col-span-1 text-xs text-gray-400 p-2">Not Fraud</div>
                        
                        <div className="text-xs text-gray-400 p-2 flex items-center justify-end">Fraud</div>
                        <div className="bg-green-800 text-green-100 p-3 flex flex-col items-center justify-center rounded-tl-md">
                          <span className="text-lg font-semibold">{evaluationMetrics.confusion_matrix.true_positive}</span>
                          <span className="text-xs mt-1">True Positive</span>
                        </div>
                        <div className="bg-red-800 text-red-100 p-3 flex flex-col items-center justify-center rounded-tr-md">
                          <span className="text-lg font-semibold">{evaluationMetrics.confusion_matrix.false_negative}</span>
                          <span className="text-xs mt-1">False Negative</span>
                        </div>
                        
                        <div className="text-xs text-gray-400 p-2 flex items-center justify-end">Not Fraud</div>
                        <div className="bg-red-800 text-red-100 p-3 flex flex-col items-center justify-center rounded-bl-md">
                          <span className="text-lg font-semibold">{evaluationMetrics.confusion_matrix.false_positive}</span>
                          <span className="text-xs mt-1">False Positive</span>
                        </div>
                        <div className="bg-green-800 text-green-100 p-3 flex flex-col items-center justify-center rounded-br-md">
                          <span className="text-lg font-semibold">{evaluationMetrics.confusion_matrix.true_negative}</span>
                          <span className="text-xs mt-1">True Negative</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="bg-gray-700 rounded-lg p-4">
                  <h3 className="text-sm font-medium text-gray-200 mb-3">Performance Metrics</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-800 rounded-lg p-3">
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400 text-xs">Precision</span>
                        <span className="text-sm font-medium text-white">{metrics.precision}</span>
                      </div>
                      <div className="w-full bg-gray-600 h-2 mt-2 rounded-full overflow-hidden">
                        <div 
                          className="bg-blue-500 h-full rounded-full" 
                          style={{width: `${evaluationMetrics.precision * 100}%`}}
                        ></div>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">True Positives / (True Positives + False Positives)</p>
                    </div>
                    
                    <div className="bg-gray-800 rounded-lg p-3">
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400 text-xs">Recall</span>
                        <span className="text-sm font-medium text-white">{metrics.recall}</span>
                      </div>
                      <div className="w-full bg-gray-600 h-2 mt-2 rounded-full overflow-hidden">
                        <div 
                          className="bg-purple-500 h-full rounded-full" 
                          style={{width: `${evaluationMetrics.recall * 100}%`}}
                        ></div>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">True Positives / (True Positives + False Negatives)</p>
                    </div>
                    
                    <div className="bg-gray-800 rounded-lg p-3">
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400 text-xs">F1 Score</span>
                        <span className="text-sm font-medium text-white">{metrics.f1}</span>
                      </div>
                      <div className="w-full bg-gray-600 h-2 mt-2 rounded-full overflow-hidden">
                        <div 
                          className="bg-green-500 h-full rounded-full" 
                          style={{width: `${evaluationMetrics.f1_score * 100}%`}}
                        ></div>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">2 × (Precision × Recall) / (Precision + Recall)</p>
                    </div>
                    
                    <div className="bg-gray-800 rounded-lg p-3">
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400 text-xs">Accuracy</span>
                        <span className="text-sm font-medium text-white">{metrics.accuracy}</span>
                      </div>
                      <div className="w-full bg-gray-600 h-2 mt-2 rounded-full overflow-hidden">
                        <div 
                          className="bg-yellow-500 h-full rounded-full" 
                          style={{width: `${evaluationMetrics.accuracy * 100}%`}}
                        ></div>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">(True Positives + True Negatives) / Total</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );
      case "single":
        return (
          <div className="bg-gray-800 rounded-lg shadow-md border border-gray-700 p-6">
            <h2 className="text-lg font-medium text-gray-100 mb-4">Check Transaction</h2>
            <FraudForm />
          </div>
        );
      case "bulk":
        return (
          <div className="bg-gray-800 rounded-lg shadow-md border border-gray-700 p-6">
            <h2 className="text-lg font-medium text-gray-100 mb-4">Process Multiple Transactions</h2>
            <BatchUpload />
          </div>
        );
      case "report":
        return (
          <div className="bg-gray-800 rounded-lg shadow-md border border-gray-700 p-6">
            <h2 className="text-lg font-medium text-gray-100 mb-4">Report Suspicious Activity</h2>
            <FraudReport />
          </div>
        );
      case "rules":
        return (
          <div className="bg-gray-800 rounded-lg shadow-md border border-gray-700 p-6">
            <h2 className="text-lg font-medium text-gray-100 mb-4">Configure Detection Rules</h2>
            <RuleConfig />
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div>
      <header className="mb-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <h1 className="text-xl font-medium text-white">
            Dashboard <span className="text-sm font-normal text-gray-400">/ Fraud Detection System</span>
          </h1>
          <div className="flex space-x-2">
            <button
              onClick={handleRefreshData}
              className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm hover:bg-blue-700 focus:outline-none transition duration-150 ease-in-out shadow-md"
            >
              Refresh Data
            </button>
          </div>
        </div>
        
        {modelConfig && (
          <div className="mt-4 p-3 bg-gray-800 rounded-md border border-gray-700">
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <span className="text-sm text-gray-300">ML Model:</span>
                <div className="relative">
                  <select 
                    className="bg-gray-900 text-gray-200 text-sm rounded-md px-3 py-1 border border-gray-600 focus:outline-none focus:ring focus:ring-blue-500 focus:ring-opacity-25"
                    onChange={(e) => {handleChangeModel(e.target.value)}}
                    value={modelConfig.model_type}
                    disabled={false}
                  >
                    <option value="knn">KNN</option>
                    <option value="knn">Gradient Boosting</option>
                  </select>
                  <div className="absolute -top-2 -right-2">
                    <span className="px-1.5 py-0.5 text-xs rounded-full bg-blue-900 text-blue-200">
                      Active
                    </span>
                  </div>
                </div>
              </div>
              
              <div className="flex items-center gap-3">
                <span className="text-sm text-gray-300">Sampling Method:</span>
                <select 
                  className="bg-gray-900 text-gray-200 text-sm rounded-md px-3 py-1 border border-gray-600 focus:outline-none focus:ring focus:ring-blue-500 focus:ring-opacity-25"
                  onChange={(e) => handleSamplingMethodChange(e.target.value)}
                  value={samplingMethod}
                >
                  {samplingMethods.map(method => (
                    <option key={method.value} value={method.value}>{method.label}</option>
                  ))}
                </select>
              </div>

              {modelChangeStatus && (
                <div className={`text-xs rounded p-2 ${
                  modelChangeStatus.type === "success" 
                    ? "bg-green-800 text-green-200" 
                    : modelChangeStatus.type === "error"
                      ? "bg-red-800 text-red-200"
                      : "bg-blue-800 text-blue-200"
                }`}>
                  {modelChangeStatus.message}
                </div>
              )}
            </div>
          </div>
        )}
      </header>

      <div className="mb-8 p-4 bg-gray-800 rounded-lg shadow-md border border-gray-700">
        <div className="flex flex-col sm:flex-row flex-wrap gap-4">
          <div className="flex-1 min-w-[200px]">
            <label className="block text-sm font-medium text-gray-300 mb-1">Transaction ID Search</label>
            <div className="flex">
              <input
                type="text"
                placeholder="Enter transaction ID"
                value={filters.transactionId}
                onChange={(e) => setFilters({ ...filters, transactionId: e.target.value })}
                className="w-full p-2 bg-gray-900 text-gray-100 rounded-l-md border border-gray-700 focus:border-blue-500 focus:ring focus:ring-blue-500 focus:ring-opacity-25 transition duration-150 ease-in-out"
              />
              <button 
                onClick={handleApplyFilters}
                className="px-3 py-2 bg-blue-600 text-white rounded-r-md text-sm hover:bg-blue-700 focus:outline-none transition duration-150 ease-in-out"
              >
                Search
              </button>
            </div>
          </div>
          
          <div className="flex-1 min-w-[200px]">
            <label className="block text-sm font-medium text-gray-300 mb-1">Date</label>
            <input
              type="date"
              value={filters.date}
              onChange={(e) => setFilters({ ...filters, date: e.target.value })}
              className="w-full p-2 bg-gray-900 text-gray-100 rounded-md border border-gray-700 focus:border-blue-500 focus:ring focus:ring-blue-500 focus:ring-opacity-25 transition duration-150 ease-in-out"
            />
          </div>
          <div className="flex-1 min-w-[200px]">
            <label className="block text-sm font-medium text-gray-300 mb-1">Dimension</label>
            <select
              value={filters.dimension}
              onChange={(e) => setFilters({ ...filters, dimension: e.target.value })}
              className="w-full p-2 bg-gray-900 text-gray-100 rounded-md border border-gray-700 focus:border-blue-500 focus:ring focus:ring-blue-500 focus:ring-opacity-25 transition duration-150 ease-in-out"
            >
              {dimensions.map(dimension => (
                <option key={dimension.value} value={dimension.value}>
                  {dimension.label}
                </option>
              ))}
            </select>
          </div>
          <div className="flex-1 min-w-[200px]">
            <label className="block text-sm font-medium text-gray-300 mb-1">Payer ID</label>
            <input
              type="text"
              placeholder="Enter payer ID"
              value={filters.payerId}
              onChange={(e) => setFilters({ ...filters, payerId: e.target.value })}
              className="w-full p-2 bg-gray-900 text-gray-100 rounded-md border border-gray-700 focus:border-blue-500 focus:ring focus:ring-blue-500 focus:ring-opacity-25 transition duration-150 ease-in-out"
            />
          </div>
          <div className="flex-1 min-w-[200px]">
            <label className="block text-sm font-medium text-gray-300 mb-1">Payee ID</label>
            <input
              type="text"
              placeholder="Enter payee ID"
              value={filters.payeeId}
              onChange={(e) => setFilters({ ...filters, payeeId: e.target.value })}
              className="w-full p-2 bg-gray-900 text-gray-100 rounded-md border border-gray-700 focus:border-blue-500 focus:ring focus:ring-blue-500 focus:ring-opacity-25 transition duration-150 ease-in-out"
            />
          </div>
          <div className="flex items-end">
            <button 
              onClick={handleApplyFilters}
              className="px-4 py-2 bg-gray-700 text-gray-200 rounded-md text-sm hover:bg-gray-600 focus:outline-none transition duration-150 ease-in-out"
              disabled={!!filters.transactionId}
            >
              Apply Filters
            </button>
          </div>
        </div>
      </div>

      <div className="mb-6">
        <div className="border-b border-gray-700">
          <nav className="flex space-x-8" aria-label="Tabs">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  py-4 px-1 border-b-2 font-medium text-sm transition duration-150 ease-in-out
                  ${
                    activeTab === tab.id
                      ? "border-blue-500 text-blue-400"
                      : "border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-600"
                  }
                `}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </div>

      <main>{renderTabContent()}</main>
    </div>
  );
}
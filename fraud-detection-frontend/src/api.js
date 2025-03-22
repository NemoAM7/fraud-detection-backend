import axios from "axios";

const API_BASE_URL = "http://localhost:8000/api"; // Update to match the server port (8000)

// Fraud Detection API (Single Transaction)
export const detectFraud = async (transaction) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/fraud/detect`, transaction);
    return response;
  } catch (error) {
    console.error("Error detecting fraud:", error);
    throw error;
  }
};

// Batch Fraud Detection API (Multiple Transactions)
export const batchDetectFraud = async (transactions) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/fraud/detect/batch`, { transactions });
    return response;
  } catch (error) {
    console.error("Error in batch detection:", error);
    throw error;
  }
};

// Fraud Reporting API (Report Fraud)
export const reportFraud = async (data) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/fraud/report`, data);
    return response;
  } catch (error) {
    console.error("Error reporting fraud:", error);
    throw error;
  }
};

// Get Rules API
export const getRules = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/rules`);
    return response;
  } catch (error) {
    console.error("Error fetching rules:", error);
    throw error;
  }
};

// Create Rule API
export const createRule = async (rule) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/rules`, rule);
    return response;
  } catch (error) {
    console.error("Error creating rule:", error);
    throw error;
  }
};

// Update Rule API
export const updateRule = async (ruleId, rule) => {
  try {
    const response = await axios.put(`${API_BASE_URL}/rules/${ruleId}`, rule);
    return response;
  } catch (error) {
    console.error("Error updating rule:", error);
    throw error;
  }
};

// Delete Rule API
export const deleteRule = async (ruleId) => {
  try {
    const response = await axios.delete(`${API_BASE_URL}/rules/${ruleId}`);
    return response;
  } catch (error) {
    console.error("Error deleting rule:", error);
    throw error;
  }
};

// Analytics APIs
export const getTransactions = async (filters) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/analytics/transactions`, { params: filters });
    return response;
  } catch (error) {
    console.error("Error fetching transactions:", error);
    throw error;
  }
};

export const getFraudByDimension = async (dimension, filters) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/analytics/fraud-by-dimension`, { 
      params: { dimension, ...filters } 
    });
    return response;
  } catch (error) {
    console.error("Error fetching fraud by dimension:", error);
    throw error;
  }
};

export const getFraudTrend = async (startDate, endDate, granularity = "day") => {
  try {
    const response = await axios.get(`${API_BASE_URL}/analytics/fraud-trend`, { 
      params: { start_date: startDate, end_date: endDate, granularity } 
    });
    return response;
  } catch (error) {
    console.error("Error fetching fraud trend:", error);
    throw error;
  }
};

export const getEvaluationMetrics = async (startDate, endDate) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/analytics/evaluation`, { 
      params: { start_date: startDate, end_date: endDate } 
    });
    return response;
  } catch (error) {
    console.error("Error fetching evaluation metrics:", error);
    throw error;
  }
};

// Model Configuration API
export const getModelConfig = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/config/model`);
    return response;
  } catch (error) {
    console.error("Error fetching model config:", error);
    throw error;
  }
};

export const updateModelConfig = async (config) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/config/model`, config);
    return response;
  } catch (error) {
    console.error("Error updating model config:", error);
    throw error;
  }
};

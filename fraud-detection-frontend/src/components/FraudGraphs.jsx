import { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

export default function FraudGraphs({ fraudData = { trends: [], categories: [] }, isLoading = false }) {
  const [timeRange, setTimeRange] = useState("7d");
  const [timeGranularity, setTimeGranularity] = useState("day");

  // Debug logging to see what data is received
  useEffect(() => {
    console.log("FraudGraphs - Received data:", fraudData);
  }, [fraudData]);

  // Update time granularity when time range changes
  useEffect(() => {
    // Set appropriate granularity based on selected time range
    switch (timeRange) {
      case "7d":
        setTimeGranularity("day");
        break;
      case "30d":
        setTimeGranularity("day");
        break;
      case "6m":
        setTimeGranularity("week");
        break;
      default:
        setTimeGranularity("day");
    }
  }, [timeRange]);

  // Chart color scheme
  const colors = {
    predicted: "#3B82F6", // blue-500
    reported: "#F43F5E",  // rose-500
    axis: "#9CA3AF",      // gray-400 for better visibility on dark bg
    background: "transparent"
  };

  const timeRangeOptions = [
    { value: "7d", label: "Last 7 Days" },
    { value: "30d", label: "Last 30 Days" },
    { value: "6m", label: "Last 6 Months" }
  ];

  const granularityOptions = [
    { value: "hour", label: "Hourly" },
    { value: "day", label: "Daily" },
    { value: "week", label: "Weekly" },
    { value: "month", label: "Monthly" }
  ];

  // Format the X axis ticks based on granularity
  const formatXAxis = (tickItem) => {
    const date = new Date(tickItem);
    
    switch (timeGranularity) {
      case "hour":
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      case "day":
        return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
      case "week":
        return `Week ${Math.ceil(date.getDate() / 7)} ${date.toLocaleDateString([], { month: 'short' })}`;
      case "month":
        return date.toLocaleDateString([], { month: 'short', year: '2-digit' });
      default:
        return tickItem;
    }
  };

  // Safe access to data with fallbacks
  const trendData = Array.isArray(fraudData?.trends) ? fraudData.trends : [];

  // Create formatted data for the trend chart
  const formattedTrendData = trendData.map(item => ({
    date: item.date,
    "Predicted Fraud": item.predicted_fraud || 0,
    "Reported Fraud": item.reported_fraud || 0
  }));

  const NoDataMessage = () => (
    <div className="flex items-center justify-center h-40 w-full">
      <div className="text-center">
        <p className="text-gray-400 mb-2">No fraud data available</p>
        <p className="text-sm text-gray-500">Use the "Report Fraud" tab to register suspicious transactions</p>
      </div>
    </div>
  );

  if (isLoading) {
    return (
      <div className="animate-pulse space-y-4">
        <div className="h-48 bg-gray-700 rounded-md"></div>
      </div>
    );
  }

  return (
    <div className="w-full">
      <div className="flex flex-wrap items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div>
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="px-3 py-1 text-sm bg-gray-700 text-white rounded-md border border-gray-600 focus:outline-none focus:ring focus:ring-blue-500 focus:ring-opacity-25"
            >
              {timeRangeOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <select
              value={timeGranularity}
              onChange={(e) => setTimeGranularity(e.target.value)}
              className="px-3 py-1 text-sm bg-gray-700 text-white rounded-md border border-gray-600 focus:outline-none focus:ring focus:ring-blue-500 focus:ring-opacity-25"
            >
              {granularityOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="space-y-6">
        <div>
          <h3 className="text-sm font-medium text-gray-300 mb-3">Fraud Trends Over Time</h3>
          {formattedTrendData.length === 0 ? (
            <NoDataMessage />
          ) : (
            <div className="h-48 md:h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={formattedTrendData}
                  margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="date" 
                    stroke="#9CA3AF"
                    tick={{ fill: '#9CA3AF' }}
                    tickFormatter={formatXAxis}
                  />
                  <YAxis 
                    stroke="#9CA3AF"
                    tick={{ fill: '#9CA3AF' }}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      color: '#E5E7EB' 
                    }}
                    labelFormatter={(label) => new Date(label).toLocaleDateString([], {
                      weekday: 'short',
                      year: 'numeric',
                      month: 'short',
                      day: 'numeric'
                    })}
                  />
                  <Legend wrapperStyle={{ color: '#E5E7EB' }} />
                  <Line
                    type="monotone"
                    dataKey="Predicted Fraud"
                    stroke="#3B82F6"
                    strokeWidth={2}
                    dot={{ r: 4 }}
                    activeDot={{ r: 6, stroke: '#2563EB', strokeWidth: 2 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="Reported Fraud"
                    stroke="#EF4444"
                    strokeWidth={2}
                    dot={{ r: 4 }}
                    activeDot={{ r: 6, stroke: '#DC2626', strokeWidth: 2 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}